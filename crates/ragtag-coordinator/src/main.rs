//! The `ragtag-coordinator` binary.
//!
//! It maintains a live node table, re-solves the pipeline partition whenever
//! the cluster changes, pushes plans via gRPC server-streaming, and exposes an
//! OpenAI-compatible `/v1/chat/completions` HTTP endpoint that proxies to a
//! local `llama-server` process.

use anyhow::{Context, Result};
use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use chrono::Local;
use clap::Parser;
use dashmap::DashMap;
use futures::{Stream, StreamExt};
use mdns_sd::{ServiceDaemon, ServiceEvent};
use ragtag_core::{NodeCapability, PartitionPlan};
use ragtag_proto::pb::{
    coordinator_server::{Coordinator, CoordinatorServer},
    ring_transport_client::RingTransportClient,
    ActivationChunk, ClusterStats, Empty, NodeStats, Stability as PbStability,
};
use ragtag_scheduler::{solve, LinearCostModel, ScheduledInput};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio::time::{interval, MissedTickBehavior};
use tokio_stream::wrappers::WatchStream;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{info, warn};
use uuid::Uuid;

mod llama_server;

const DEFAULT_SERVICE_TYPE: &str = "_ragtag._tcp.local.";

fn system_hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|h| !h.is_empty())
        .or_else(|| {
            std::process::Command::new("hostname")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

fn service_type() -> String {
    std::env::var("RAGTAG_SERVICE_TYPE").unwrap_or_else(|_| DEFAULT_SERVICE_TYPE.to_string())
}

/// Coordinator CLI.
#[derive(Parser, Debug)]
#[command(name = "ragtag-coordinator", version)]
pub struct Args {
    /// Address the gRPC coordinator server binds to.
    #[arg(long, default_value = "127.0.0.1:0")]
    grpc: SocketAddr,
    /// Address the OpenAI-compatible HTTP server binds to.
    #[arg(long, default_value = "127.0.0.1:3000")]
    http: SocketAddr,
    /// Total transformer layers to partition.
    #[arg(long, default_value_t = 6)]
    total_layers: u32,
    /// Bytes of RAM per layer (used by the scheduler cost model).
    #[arg(long, default_value_t = 512 * 1024 * 1024)]
    bytes_per_layer: u64,
    /// Baseline seconds per layer (used by the scheduler cost model).
    #[arg(long, default_value_t = 1.0)]
    seconds_per_layer: f32,
    /// Seconds without a heartbeat before a node is marked unreachable.
    #[arg(long, default_value_t = 15)]
    heartbeat_timeout_secs: u64,
    /// Optional mDNS daemon port.
    #[arg(long)]
    mdns_port: Option<u16>,
    /// The coordinator's own node id. Any heartbeat with this id is rejected so
    /// the coordinator is never included in the schedulable node table, which in
    /// turn keeps the llama.cpp `--tensor-split` vector aligned with the number
    /// of RPC workers.
    #[arg(long)]
    coordinator_id: Option<Uuid>,
    /// Path to the GGUF model. When set, the coordinator starts `llama-server`
    /// with `--rpc` and `--tensor-split` derived from the partition plan.
    #[arg(long)]
    model: Option<PathBuf>,
    /// llama.cpp `llama-server` binary name or path.
    #[arg(long, default_value = "llama-server")]
    llama_server_binary: String,
    /// Host for the local `llama-server` HTTP API.
    #[arg(long, default_value = "127.0.0.1")]
    llama_host: String,
    /// Port for the local `llama-server` HTTP API.
    #[arg(long, default_value_t = 8081)]
    llama_port: u16,
}

/// Shared coordinator state.
pub struct CoordinatorState {
    coordinator_id: Uuid,
    nodes: DashMap<Uuid, NodeCapability>,
    addrs: DashMap<Uuid, SocketAddr>,
    rpc_addrs: DashMap<Uuid, SocketAddr>,
    hostnames: DashMap<Uuid, String>,
    plan_tx: watch::Sender<PartitionPlan>,
    llama_ready: AtomicBool,
    events: Mutex<VecDeque<String>>,
    llama_client: Client,
    total_layers: u32,
    bytes_per_layer: u64,
    seconds_per_layer: f32,
    heartbeat_timeout: Duration,
    model: Option<PathBuf>,
    coordinator_hostname: String,
    llama_host: String,
    llama_port: u16,
}

impl CoordinatorState {
    fn new(args: &Args) -> (Self, watch::Receiver<PartitionPlan>) {
        let (plan_tx, plan_rx) = watch::channel(PartitionPlan {
            assignments: Vec::new(),
        });
        let state = Self {
            coordinator_id: args.coordinator_id.unwrap_or_else(Uuid::new_v4),
            nodes: DashMap::new(),
            addrs: DashMap::new(),
            rpc_addrs: DashMap::new(),
            hostnames: DashMap::new(),
            events: Mutex::new(VecDeque::new()),
            plan_tx,
            llama_ready: AtomicBool::new(false),
            llama_client: Client::new(),
            total_layers: args.total_layers,
            bytes_per_layer: args.bytes_per_layer,
            seconds_per_layer: args.seconds_per_layer,
            heartbeat_timeout: Duration::from_secs(args.heartbeat_timeout_secs),
            model: args.model.clone(),
            coordinator_hostname: system_hostname(),
            llama_host: args.llama_host.clone(),
            llama_port: args.llama_port,
        };
        (state, plan_rx)
    }

    fn recompute_plan(&self) -> Result<PartitionPlan, ragtag_scheduler::ScheduleError> {
        let mut devices = Vec::new();
        let mut addresses = HashMap::new();

        for entry in self.nodes.iter() {
            let node = entry.value();
            if !node.reachable {
                continue;
            }
            let addr = self
                .addrs
                .get(&node.node_id)
                .map(|a| *a)
                .unwrap_or_else(|| SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 0));
            devices.push(node.clone());
            addresses.insert(node.node_id, addr);
        }

        let input = ScheduledInput::new(self.coordinator_id, &devices, &addresses)?;
        let cost = LinearCostModel::new(self.bytes_per_layer, self.seconds_per_layer);
        let plan = solve(self.total_layers, &input, &cost)?;

        let _ = self.plan_tx.send(plan.clone());
        let summary = plan
            .assignments
            .iter()
            .map(|a| {
                let host = self.node_hostname(a.node_id);
                format!("{}={}-{}", host, a.start_layer, a.end_layer)
            })
            .collect::<Vec<_>>()
            .join(", ");
        info!(
            "recomputed plan: {} assignments, {} layers",
            plan.assignments.len(),
            plan.total_layers()
        );
        self.log_event(format!("plan({}): {}", plan.total_layers(), summary));
        Ok(plan)
    }

    fn recompute_plan_or_empty(&self) {
        match self.recompute_plan() {
            Ok(_) => {}
            Err(e) => {
                warn!("scheduler error: {}", e);
                self.log_event(format!("plan infeasible: {}", e));
                let _ = self.plan_tx.send(PartitionPlan {
                    assignments: Vec::new(),
                });
            }
        }
    }

    /// Build the ordered `(rpc_address, layer_count)` vector used to construct
    /// the `llama-server` `--rpc` and `--tensor-split` arguments.
    ///
    /// The vector is built from a single iteration over `plan.assignments`, so
    /// the two CLI arguments are guaranteed to share the same order.
    pub fn rpc_plan(
        &self,
        plan: &PartitionPlan,
    ) -> Result<Vec<(SocketAddr, u32)>, ragtag_scheduler::ScheduleError> {
        let mut args = Vec::with_capacity(plan.assignments.len());

        for a in &plan.assignments {
            if a.node_id == self.coordinator_id {
                // The scheduler already excludes the coordinator, but this
                // function is called from a separate task and is a second line
                // of defence against the tensor-split shape bug.
                continue;
            }

            let count = a.layer_count();
            if count == 0 {
                continue;
            }

            let addr = self
                .rpc_addrs
                .get(&a.node_id)
                .ok_or(ragtag_scheduler::ScheduleError::MissingAddress(a.node_id))?;

            args.push((*addr, count));
        }

        if args.is_empty() {
            return Err(ragtag_scheduler::ScheduleError::NoDevices);
        }

        Ok(args)
    }

    fn node_hostname(&self, node_id: Uuid) -> String {
        self.hostnames
            .get(&node_id)
            .map(|h| h.clone())
            .unwrap_or_else(|| {
                node_id
                    .to_string()
                    .split('-')
                    .next()
                    .unwrap_or("?")
                    .to_string()
            })
    }

    pub fn log_event(&self, msg: impl Into<String>) {
        let text = format!("{} {}", Local::now().format("%H:%M:%S"), msg.into());
        let mut events = self.events.lock().unwrap();
        events.push_back(text);
        while events.len() > 50 {
            events.pop_front();
        }
    }

    fn llama_server_url(&self) -> Option<String> {
        if self.llama_ready.load(Ordering::Relaxed) {
            Some(format!("http://{}:{}", self.llama_host, self.llama_port))
        } else {
            None
        }
    }
}

/// gRPC service wrapper. Newtype needed because of Rust's orphan rules.
#[derive(Clone)]
pub struct CoordinatorService(Arc<CoordinatorState>);

impl CoordinatorService {
    /// Create a coordinator service around the shared state.
    pub fn new(state: Arc<CoordinatorState>) -> Self {
        Self(state)
    }
}

#[tonic::async_trait]
impl Coordinator for CoordinatorService {
    type WatchPlanStream = Pin<
        Box<dyn Stream<Item = std::result::Result<ragtag_proto::pb::PartitionPlan, Status>> + Send>,
    >;

    async fn heartbeat(
        &self,
        request: Request<ragtag_proto::pb::NodeCapability>,
    ) -> std::result::Result<Response<ragtag_proto::pb::Ack>, Status> {
        let remote_addr = request.remote_addr();
        let ring_addr = request
            .metadata()
            .get("ring-addr")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok());
        let rpc_addr = request
            .metadata()
            .get("rpc-addr")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok());
        let hostname = request
            .metadata()
            .get("hostname")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let inner = request.into_inner();
        let cap =
            NodeCapability::try_from(inner).map_err(|e| Status::invalid_argument(e.to_string()))?;

        let node_id = cap.node_id;
        if node_id == self.0.coordinator_id {
            warn!(
                %node_id,
                "heartbeat from coordinator node id rejected; the coordinator must not be in the node table"
            );
            return Ok(Response::new(ragtag_proto::pb::Ack {}));
        }

        let is_new = !self.0.nodes.contains_key(&node_id);
        let mut stability_changed = false;

        if let Some(old) = self.0.nodes.get(&node_id) {
            if old.stability != cap.stability {
                stability_changed = true;
            }
        }

        let hostname = hostname.unwrap_or_else(|| self.0.node_hostname(node_id));
        let was_unreachable = self
            .0
            .nodes
            .get(&node_id)
            .map(|n| !n.reachable)
            .unwrap_or(false);

        let mut cap = cap;
        cap.reachable = true;
        cap.last_heartbeat = Instant::now();
        self.0.nodes.insert(node_id, cap);

        if let Some(addr) = ring_addr {
            self.0.addrs.insert(node_id, addr);
        } else if let Some(addr) = remote_addr {
            self.0.addrs.insert(node_id, addr);
        }

        if let Some(addr) = rpc_addr {
            self.0.rpc_addrs.insert(node_id, addr);
        } else if let Some(addr) = self.0.addrs.get(&node_id) {
            // Fall back to the ring address when no explicit RPC address is
            // supplied, so older nodes can still be scheduled until the
            // RingTransport path is removed.
            self.0.rpc_addrs.insert(node_id, *addr);
        }

        self.0.hostnames.insert(node_id, hostname.clone());

        if is_new {
            self.0.log_event(format!("node {} joined", hostname));
        } else if was_unreachable {
            self.0.log_event(format!("node {} reachable", hostname));
        }

        if is_new || stability_changed {
            self.0.recompute_plan_or_empty();
        }

        Ok(Response::new(ragtag_proto::pb::Ack {}))
    }

    async fn watch_plan(
        &self,
        request: Request<ragtag_proto::pb::NodeId>,
    ) -> std::result::Result<Response<Self::WatchPlanStream>, Status> {
        let _node_id = Uuid::try_from(request.into_inner())
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        let rx = self.0.plan_tx.subscribe();
        let stream =
            WatchStream::new(rx).map(|plan| Ok(ragtag_proto::pb::PartitionPlan::from(plan)));

        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_stats(
        &self,
        _request: Request<Empty>,
    ) -> std::result::Result<Response<ClusterStats>, Status> {
        let state = &self.0;
        let plan = state.plan_tx.borrow().clone();

        let mut nodes = Vec::with_capacity(state.nodes.len());

        for entry in state.nodes.iter() {
            let cap = entry.value();
            let (start_layer, end_layer) = plan
                .assignments
                .iter()
                .find(|a| a.node_id == cap.node_id)
                .map(|a| (a.start_layer, a.end_layer))
                .unwrap_or((0, 0));

            let rpc_addr = state
                .rpc_addrs
                .get(&cap.node_id)
                .map(|a| a.to_string())
                .unwrap_or_default();

            let hostname = state
                .hostnames
                .get(&cap.node_id)
                .map(|h| h.clone())
                .unwrap_or_else(|| {
                    cap.node_id
                        .to_string()
                        .split('-')
                        .next()
                        .unwrap_or("?")
                        .to_string()
                });

            let used_ram = (end_layer.saturating_sub(start_layer) as u64)
                .saturating_mul(state.bytes_per_layer);
            nodes.push(NodeStats {
                node_id: cap.node_id.as_bytes().to_vec(),
                tokens_per_sec: cap.tokens_per_sec,
                free_ram_bytes: cap.free_ram_bytes,
                used_ram_bytes: used_ram,
                reachable: cap.reachable,
                stability: PbStability::from(cap.stability) as i32,
                rpc_addr,
                start_layer,
                end_layer,
                hostname,
            });
        }

        let events: Vec<_> = state.events.lock().unwrap().iter().cloned().collect();

        Ok(Response::new(ClusterStats {
            llama_ready: state.llama_ready.load(Ordering::Relaxed),
            model: state
                .model
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            total_layers: state.total_layers,
            llama_server_url: state.llama_server_url().unwrap_or_default(),
            coordinator_hostname: state.coordinator_hostname.clone(),
            bytes_per_layer: state.bytes_per_layer,
            events,
            nodes,
        }))
    }
}

async fn timeout_checker(state: Arc<CoordinatorState>) {
    let mut timer = interval(Duration::from_secs(1));
    timer.set_missed_tick_behavior(MissedTickBehavior::Delay);

    loop {
        timer.tick().await;

        let mut changed = false;
        for mut entry in state.nodes.iter_mut() {
            let elapsed = {
                let node = entry.value();
                if !node.reachable {
                    continue;
                }
                node.last_heartbeat.elapsed()
            };
            if elapsed > state.heartbeat_timeout {
                let id = entry.value().node_id;
                let host = state.node_hostname(id);
                entry.value_mut().reachable = false;
                changed = true;
                warn!("node {} heartbeat timed out", id);
                state.log_event(format!("node {} timed out", host));
            }
        }

        if changed {
            state.recompute_plan_or_empty();
        }
    }
}

fn run_mdns_browse(state: Arc<CoordinatorState>, port: Option<u16>) -> Result<()> {
    let mdns = match port {
        Some(p) => ServiceDaemon::new_with_port(p).context("create mdns daemon with port")?,
        None => ServiceDaemon::new().context("create mdns daemon")?,
    };

    let receiver = mdns.browse(&service_type()).context("browse mdns")?;

    std::thread::spawn(move || {
        while let Ok(event) = receiver.recv() {
            match event {
                ServiceEvent::ServiceResolved(info) => {
                    if let Some(id_str) = info.get_property_val_str("node_id") {
                        if let Ok(id) = Uuid::parse_str(id_str) {
                            if id == state.coordinator_id {
                                continue;
                            }
                            let ip = info.get_addresses().iter().next().map(|ip| ip.to_ip_addr());
                            if let Some(ip) = ip {
                                let port = info.get_port();
                                let ring_addr = SocketAddr::new(ip, port);
                                info!(%id, %ring_addr, "mdns service resolved");
                                state.addrs.insert(id, ring_addr);

                                if let Some(rpc_str) = info.get_property_val_str("rpc_addr") {
                                    if let Ok(rpc) = rpc_str.parse() {
                                        state.rpc_addrs.insert(id, rpc);
                                    }
                                } else {
                                    state.rpc_addrs.insert(id, ring_addr);
                                }

                                let host = info
                                    .get_property_val_str("hostname")
                                    .map(|h| h.to_string())
                                    .or_else(|| {
                                        info.get_fullname().split('.').next().map(|h| h.to_string())
                                    })
                                    .unwrap_or_else(|| {
                                        id.to_string().split('-').next().unwrap_or("?").to_string()
                                    });
                                state.hostnames.insert(id, host.clone());

                                state.log_event(format!("node {} discovered", host));
                                state.recompute_plan_or_empty();
                            }
                        }
                    }
                }
                ServiceEvent::ServiceRemoved(fullname, _) => {
                    let prefix = fullname.split('.').next().unwrap_or("");
                    if let Ok(id) = Uuid::parse_str(prefix) {
                        let host = state.node_hostname(id);
                        state.addrs.remove(&id);
                        state.rpc_addrs.remove(&id);
                        state.hostnames.remove(&id);
                        if let Some(mut entry) = state.nodes.get_mut(&id) {
                            entry.value_mut().reachable = false;
                        }
                        state.log_event(format!("node {} left", host));
                        state.recompute_plan_or_empty();
                    }
                }
                _ => {}
            }
        }
    });

    Ok(())
}

async fn run_grpc_server(listener: TcpListener, state: Arc<CoordinatorState>) -> Result<()> {
    Server::builder()
        .add_service(CoordinatorServer::new(CoordinatorService::new(state)))
        .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
        .await
        .context("coordinator gRPC server")?;
    Ok(())
}

#[derive(Debug, Deserialize, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
}

async fn chat_completions(
    State(state): State<Arc<CoordinatorState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> std::result::Result<Json<serde_json::Value>, StatusCode> {
    if req.stream {
        return Err(StatusCode::NOT_IMPLEMENTED);
    }

    // If a model is configured and llama-server is ready, proxy to it.
    if state.model.is_some() {
        if let Some(url) = state.llama_server_url() {
            state.log_event("generation started");

            let body = serde_json::json!({
                "model": req.model,
                "messages": req.messages,
                "stream": false,
            });

            let response = state
                .llama_client
                .post(format!("{}/v1/chat/completions", url))
                .json(&body)
                .send()
                .await
                .map_err(|_| {
                    state.log_event("generation failed");
                    StatusCode::BAD_GATEWAY
                })?;

            if !response.status().is_success() {
                state.log_event("generation failed");
                return Err(StatusCode::BAD_GATEWAY);
            }

            let json = response.json::<serde_json::Value>().await.map_err(|_| {
                state.log_event("generation failed");
                StatusCode::BAD_GATEWAY
            })?;
            state.log_event("generation completed");
            return Ok(Json(json));
        }
    }

    // Fallback to the legacy ring-transport path when no model is configured
    // or llama-server is not ready.
    state.log_event("request routed to ring");
    let plan = state.plan_tx.borrow().clone();
    let entry = plan.entry_node().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    let prompt = req
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let client_addr = format!("http://{}", entry.next_hop);
    let mut client = RingTransportClient::connect(client_addr)
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?;

    let chunk = ActivationChunk {
        request_id: Uuid::new_v4().as_bytes().to_vec(),
        sequence_no: 0,
        payload: prompt.into_bytes(),
    };

    let mut stream = client
        .forward(Request::new(futures::stream::iter(vec![chunk])))
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?
        .into_inner();

    let mut output = Vec::new();
    while let Some(item) = stream.next().await {
        let item = item.map_err(|_| StatusCode::BAD_GATEWAY)?;
        output.extend(item.payload);
    }

    let content = String::from_utf8_lossy(&output).to_string();
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Ok(Json(serde_json::json!({
        "id": format!("chatcmpl-{}", Uuid::new_v4()),
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop"
        }]
    })))
}

async fn run_http_server(listener: TcpListener, state: Arc<CoordinatorState>) -> Result<()> {
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    axum::serve(listener, app).await.context("http server")?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_ansi(false).init();
    let args = Args::parse();

    let (state, plan_rx) = CoordinatorState::new(&args);
    let state = Arc::new(state);

    let grpc_listener = TcpListener::bind(args.grpc)
        .await
        .context("bind gRPC socket")?;
    let grpc_addr = grpc_listener.local_addr()?;

    let http_listener = TcpListener::bind(args.http)
        .await
        .context("bind HTTP socket")?;
    let http_addr = http_listener.local_addr()?;

    info!(%grpc_addr, %http_addr, "coordinator starting");
    info!(coordinator_id = %state.coordinator_id, "coordinator id");

    run_mdns_browse(state.clone(), args.mdns_port)?;

    let grpc = tokio::spawn(run_grpc_server(grpc_listener, state.clone()));
    let http = tokio::spawn(run_http_server(http_listener, state.clone()));
    let timeouts = tokio::spawn(timeout_checker(state.clone()));

    let llama = if let Some(model) = args.model {
        let manager = llama_server::LlamaServerManager::new(
            state.clone(),
            model,
            args.llama_server_binary,
            args.llama_host,
            args.llama_port,
            plan_rx,
        );
        tokio::spawn(manager.run())
    } else {
        tokio::spawn(async { std::future::pending::<()>().await })
    };

    tokio::select! {
        _ = tokio::signal::ctrl_c() => info!("shutting down"),
        _ = grpc => {},
        _ = http => {},
        _ = timeouts => {},
        _ = llama => {},
    }

    Ok(())
}
