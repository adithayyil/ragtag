use anyhow::Result;
use chrono::Local;
use clap::Parser;
use crossterm::event::{self, Event as CrosstermEvent, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::{cursor, ExecutableCommand};
use ragtag_proto::pb::{ClusterStats, Empty};
use ragtag_proto::CoordinatorClient;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::*;
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, Cell, List, ListItem, Paragraph, Row, Table};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{self, stdout, Write};
use std::time::Duration;
use tokio::sync::mpsc;
use tonic::transport::Channel;
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "ragtag-chat")]
#[command(about = "TUI chat client for ragtag")]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:3000")]
    url: String,

    #[arg(long, default_value = "http://127.0.0.1:50051")]
    grpc: String,

    #[arg(long, default_value = "tinyllama")]
    model: String,

    #[arg(long)]
    system: Option<String>,

    #[arg(long, default_value_t = 128)]
    max_tokens: u32,
}

#[derive(Clone, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: AssistantMessage,
}

#[derive(Deserialize)]
struct AssistantMessage {
    content: String,
}

enum AppEvent {
    Key(CrosstermEvent),
    Stats(ClusterStats),
    ChatResponse(Message),
    Error(String),
    Connected,
}

struct App {
    url: String,
    grpc: String,
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
    input: String,
    stats: Option<ClusterStats>,
    activity: Vec<String>,
    status: &'static str,
    client: Client,
    generating: bool,
}

impl App {
    fn new(args: Args, client: Client) -> Self {
        let mut messages = Vec::new();
        if let Some(s) = args.system {
            messages.push(Message {
                role: "system".into(),
                content: s,
            });
        }

        Self {
            url: args.url,
            grpc: args.grpc,
            model: args.model,
            max_tokens: args.max_tokens,
            messages,
            input: String::new(),
            stats: None,
            activity: vec!["ragtag-chat started".into()],
            status: "idle",
            client,
            generating: false,
        }
    }

    fn log(&mut self, msg: String) {
        self.activity
            .push(format!("{} {}", Local::now().format("%H:%M:%S"), msg));
        if self.activity.len() > 100 {
            self.activity.remove(0);
        }
    }

    fn draw(&self, f: &mut Frame) {
        let area = f.area();

        // Title bar at the top
        let title_area = Rect::new(0, 0, area.width, 1);
        let main_area = Rect::new(0, 1, area.width, area.height.saturating_sub(1));

        let title = Block::default()
            .title(" ragtag ")
            .title_alignment(Alignment::Center)
            .borders(Borders::TOP);
        f.render_widget(title, title_area);

        let main = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(9)])
            .split(main_area);

        let top = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(main[0]);

        let bottom = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(main[1]);

        self.draw_cluster(f, top[0]);
        self.draw_chat(f, top[1]);
        self.draw_activity(f, bottom[0]);
        self.draw_input(f, bottom[1]);
    }

    fn draw_cluster(&self, f: &mut Frame, area: Rect) {
        let block = Block::default().title(" cluster ").borders(Borders::ALL);
        let inner = block.inner(area);
        f.render_widget(block, area);

        let rows: Vec<Row> = self
            .stats
            .as_ref()
            .map(|s| {
                s.nodes
                    .iter()
                    .map(|n| {
                        let name = if n.hostname.is_empty() {
                            name_from_uuid(&n.node_id)
                        } else {
                            n.hostname.clone()
                        };
                        let layers = if n.start_layer == 0 && n.end_layer == 0
                            || n.start_layer > n.end_layer
                        {
                            "—".into()
                        } else {
                            format!("{}-{}", n.start_layer, n.end_layer)
                        };
                        let used = format_size(n.used_ram_bytes);
                        let free = format_size(n.free_ram_bytes);
                        let prof = format!("{:.1}", n.tokens_per_sec);
                        let status = if n.reachable { "●" } else { "○" };
                        let row = Row::new(vec![
                            Cell::from(format!("{} {}", status, name)),
                            Cell::from(layers),
                            Cell::from(used),
                            Cell::from(free),
                            Cell::from(prof),
                        ]);
                        row
                    })
                    .collect()
            })
            .unwrap_or_default();

        let table = Table::new(
            rows,
            [
                Constraint::Percentage(30),
                Constraint::Percentage(18),
                Constraint::Percentage(15),
                Constraint::Percentage(15),
                Constraint::Percentage(22),
            ],
        )
        .header(
            Row::new(vec!["NODE", "LAYERS", "USED", "FREE", "GFLOPS"])
                .style(Style::default().add_modifier(Modifier::BOLD)),
        );

        let line = if let Some(stats) = &self.stats {
            let mut spans: Vec<Span> = Vec::new();
            spans.push(Span::raw(status_text(stats.llama_ready)));

            if !stats.coordinator_hostname.is_empty() {
                spans.push(Span::raw(" | "));
                spans.push(Span::styled(
                    format!("coord: {}", stats.coordinator_hostname),
                    Style::default().fg(Color::Magenta),
                ));
            }

            spans.push(Span::raw(format!(" | layers: {}", stats.total_layers)));

            if !stats.model.is_empty() {
                spans.push(Span::raw(format!(" | model: {}", stats.model)));
            }

            if !stats.llama_server_url.is_empty() {
                spans.push(Span::raw(format!(" | engine: {}", stats.llama_server_url)));
            }

            Line::from(spans)
        } else {
            Line::from("connecting to coordinator...")
        };
        let split = Paragraph::new(line);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(1)])
            .split(inner);

        f.render_widget(table, chunks[0]);
        f.render_widget(split, chunks[1]);
    }

    fn draw_chat(&self, f: &mut Frame, area: Rect) {
        let block = Block::default().title(" chat ").borders(Borders::ALL);

        let mut text = Vec::new();
        for msg in &self.messages {
            let (label, style) = match msg.role.as_str() {
                "user" => ("you", Style::default().fg(Color::Cyan)),
                "assistant" => ("ragtag", Style::default().fg(Color::Green)),
                _ => ("system", Style::default().fg(Color::Yellow)),
            };
            text.push(Line::from(vec![
                Span::styled(format!("{}: ", label), style.add_modifier(Modifier::BOLD)),
                Span::raw(&msg.content),
            ]));
        }

        let paragraph = Paragraph::new(Text::from(text)).block(block);
        f.render_widget(paragraph, area);
    }

    fn draw_activity(&self, f: &mut Frame, area: Rect) {
        let block = Block::default().title(" activity ").borders(Borders::ALL);
        let mut lines = self.activity.clone();
        if let Some(stats) = &self.stats {
            lines.extend(stats.events.iter().cloned());
        }
        let items: Vec<ListItem> = lines
            .iter()
            .rev()
            .map(|l| ListItem::new(l.clone()))
            .collect();
        let list = List::new(items).block(block);
        f.render_widget(list, area);
    }

    fn draw_input(&self, f: &mut Frame, area: Rect) {
        let title = if self.generating {
            " input (generating...) "
        } else {
            " input "
        };
        let block = Block::default().title(title).borders(Borders::ALL);
        let input = Paragraph::new(format!("> {}", self.input)).block(block);
        f.render_widget(input, area);
    }
}

fn name_from_uuid(bytes: &[u8]) -> String {
    Uuid::from_slice(bytes)
        .map(|u| u.to_string().split('-').next().unwrap_or("?").to_string())
        .unwrap_or_else(|_| "?".into())
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1}MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1}KB", bytes as f64 / 1_000.0)
    } else {
        format!("{}B", bytes)
    }
}

fn status_text(ready: bool) -> &'static str {
    if ready {
        "● ready"
    } else {
        "○ standby"
    }
}

async fn run_stats_poll(mut client: CoordinatorClient<Channel>, tx: mpsc::Sender<AppEvent>) {
    let _ = tx.try_send(AppEvent::Connected).ok();
    loop {
        match client.get_stats(Empty {}).await {
            Ok(resp) => {
                let _ = tx.try_send(AppEvent::Stats(resp.into_inner())).ok();
            }
            Err(e) => {
                let _ = tx.try_send(AppEvent::Error(format!("stats: {}", e))).ok();
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

async fn send_chat(
    client: Client,
    url: String,
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
    tx: mpsc::Sender<AppEvent>,
) {
    let request = json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    });

    match client
        .post(format!("{}/v1/chat/completions", url.trim_end_matches('/')))
        .json(&request)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => match resp.json::<ChatCompletionResponse>().await
        {
            Ok(body) => {
                if let Some(choice) = body.choices.first() {
                    let _ = tx
                        .try_send(AppEvent::ChatResponse(Message {
                            role: "assistant".into(),
                            content: choice.message.content.clone(),
                        }))
                        .ok();
                } else {
                    let _ = tx
                        .try_send(AppEvent::Error("no choices in response".into()))
                        .ok();
                }
            }
            Err(e) => {
                let _ = tx
                    .try_send(AppEvent::Error(format!("parse error: {}", e)))
                    .ok();
            }
        },
        Ok(resp) => {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            let _ = tx
                .try_send(AppEvent::Error(format!("HTTP {}: {}", status, text)))
                .ok();
        }
        Err(e) => {
            let _ = tx
                .try_send(AppEvent::Error(format!("request failed: {}", e)))
                .ok();
        }
    }
}

async fn run_app(mut app: App) -> Result<()> {
    let (tx, mut rx) = mpsc::channel::<AppEvent>(128);

    // Spawn crossterm reader.
    let event_tx = tx.clone();
    tokio::task::spawn_blocking(move || loop {
        if let Ok(ev) = event::read() {
            if event_tx.blocking_send(AppEvent::Key(ev)).is_err() {
                break;
            }
        }
    });

    // Spawn gRPC stats poller.
    let grpc_url = app.grpc.clone();
    let stats_tx = tx.clone();
    tokio::spawn(async move {
        match CoordinatorClient::connect(grpc_url).await {
            Ok(client) => {
                run_stats_poll(client, stats_tx).await;
            }
            Err(e) => {
                let _ = stats_tx
                    .try_send(AppEvent::Error(format!("grpc connect: {}", e)))
                    .ok();
            }
        }
    });

    let mut terminal = {
        let mut stdout = stdout();
        stdout.execute(EnterAlternateScreen)?;
        stdout.execute(cursor::Hide)?;
        enable_raw_mode()?;
        Terminal::new(CrosstermBackend::new(stdout))?
    };

    let mut running = true;

    while running {
        let _ = terminal.draw(|f| app.draw(f));

        let ev = rx.recv().await;

        match ev {
            Some(AppEvent::Key(CrosstermEvent::Key(key))) if key.kind == KeyEventKind::Press => {
                match key.code {
                    KeyCode::Char('c') if key.modifiers == KeyModifiers::CONTROL => running = false,
                    KeyCode::Esc => running = false,
                    KeyCode::Char(c) => app.input.push(c),
                    KeyCode::Backspace => {
                        app.input.pop();
                    }
                    KeyCode::Enter => {
                        let input = app.input.trim().to_string();
                        if input.is_empty() {
                            continue;
                        }
                        if input == "/quit" {
                            running = false;
                            continue;
                        }
                        app.input.clear();

                        app.messages.push(Message {
                            role: "user".into(),
                            content: input.clone(),
                        });
                        app.log(format!("user: {}", input));

                        app.generating = true;
                        app.status = "generating";

                        let client = app.client.clone();
                        let url = app.url.clone();
                        let model = app.model.clone();
                        let max_tokens = app.max_tokens;
                        let messages = app.messages.clone();
                        let tx = tx.clone();
                        tokio::spawn(send_chat(client, url, model, max_tokens, messages, tx));
                    }
                    _ => {}
                }
            }
            Some(AppEvent::Stats(stats)) => {
                let previous = app.stats.as_ref().map(|s| s.nodes.len()).unwrap_or(0);
                if stats.nodes.len() != previous {
                    app.log(format!("cluster updated: {} node(s)", stats.nodes.len()));
                }
                if stats.llama_ready && app.status == "idle" {
                    app.log("engine ready".into());
                    app.status = "ready";
                }
                app.stats = Some(stats);
            }
            Some(AppEvent::ChatResponse(msg)) => {
                app.messages.push(msg.clone());
                app.log(format!(
                    "assistant: {}",
                    msg.content
                        .split_whitespace()
                        .take(8)
                        .collect::<Vec<_>>()
                        .join(" ")
                ));
                app.generating = false;
                if !app.generating {
                    app.status = "ready";
                }
            }
            Some(AppEvent::Error(e)) => {
                app.log(format!("error: {}", e));
                app.generating = false;
                app.status = "ready";
            }
            Some(AppEvent::Connected) => app.log("gRPC connected".into()),
            _ => {}
        }
    }

    disable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(LeaveAlternateScreen)?;
    stdout.execute(cursor::Show)?;
    stdout.flush()?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let client = Client::new();
    let app = App::new(args, client);
    run_app(app).await
}
