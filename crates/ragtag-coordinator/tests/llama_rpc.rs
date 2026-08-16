//! End-to-end integration test for llama.cpp RPC orchestration.
//!
//! This test requires `llama-server` and `llama-rpc-server` on `PATH` and the
//! workspace `models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf` model. Enable it with
//! `cargo test -p ragtag-coordinator --test llama_rpc --features llama-e2e`.

#![cfg(feature = "llama-e2e")]
//! Spawns two real `llama-rpc-server` workers, starts the coordinator with a
//! tiny GGUF model, registers two fake nodes with `rpc-addr` metadata, and
//! asserts that `POST /v1/chat/completions` returns a real generated response
//! rather than the no-op ring fallback.

use anyhow::{bail, Context, Result};
use ragtag_core::{NodeCapability, Stability};
use ragtag_proto::pb::coordinator_client::CoordinatorClient;
use std::net::{SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tonic::metadata::MetadataValue;
use tonic::Request;
use uuid::Uuid;

const COORDINATOR_EXE: &str = env!("CARGO_BIN_EXE_ragtag-coordinator");

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn default_model() -> PathBuf {
    workspace_root().join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf")
}

fn find_free_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").context("bind ephemeral port")?;
    let port = listener.local_addr()?.port();
    Ok(port)
}

async fn read_lines<R: AsyncRead + Unpin + Send + 'static>(reader: R, tx: mpsc::Sender<String>) {
    let mut lines = BufReader::new(reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        if tx.send(line).await.is_err() {
            break;
        }
    }
}

async fn spawn_worker(port: u16) -> Result<Child> {
    let mut child = Command::new("llama-rpc-server")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--device")
        .arg("CPU")
        .arg("--cache")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("spawn llama-rpc-server; is it in PATH?")?;

    // Give the worker a moment to fail if the port is in use.
    tokio::time::sleep(Duration::from_millis(500)).await;
    if let Ok(Some(status)) = child.try_wait() {
        bail!("llama-rpc-server exited early: {}", status);
    }

    Ok(child)
}

async fn spawn_coordinator(
    model: &Path,
    llama_port: u16,
) -> Result<(Child, SocketAddr, SocketAddr, mpsc::Receiver<String>)> {
    let mut child = Command::new(COORDINATOR_EXE)
        .arg("--total-layers")
        .arg("22")
        .arg("--bytes-per-layer")
        .arg("30000000")
        .arg("--seconds-per-layer")
        .arg("1.0")
        .arg("--heartbeat-timeout-secs")
        .arg("60")
        .arg("--grpc")
        .arg("127.0.0.1:0")
        .arg("--http")
        .arg("127.0.0.1:0")
        .arg("--llama-port")
        .arg(llama_port.to_string())
        .arg("--model")
        .arg(model)
        .arg("--llama-server-binary")
        .arg("llama-server")
        .arg("--coordinator-id")
        .arg("00000000-0000-0000-0000-000000000000")
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawn coordinator binary")?;

    let (tx, mut rx) = mpsc::channel::<String>(1024);
    tokio::spawn(read_lines(
        child.stdout.take().context("coordinator stdout missing")?,
        tx.clone(),
    ));
    tokio::spawn(read_lines(
        child.stderr.take().context("coordinator stderr missing")?,
        tx,
    ));

    let addr_re = regex::Regex::new(r"grpc_addr=(\S+)\s+http_addr=(\S+)").unwrap();

    let (grpc, http) = loop {
        let line = timeout(Duration::from_secs(10), rx.recv())
            .await
            .context("timeout waiting for coordinator startup")?
            .context("coordinator output closed")?;

        if let Some(caps) = addr_re.captures(&line) {
            break (
                caps[1].parse::<SocketAddr>()?,
                caps[2].parse::<SocketAddr>()?,
            );
        }
    };

    Ok((child, grpc, http, rx))
}

async fn wait_for_llama_ready(rx: &mut mpsc::Receiver<String>, max_wait: Duration) -> Result<()> {
    let ready_re = regex::Regex::new(r"llama-server ready").unwrap();
    timeout(max_wait, async {
        while let Some(line) = rx.recv().await {
            if ready_re.is_match(&line) {
                return Ok(());
            }
        }
        bail!("coordinator output closed before llama-server became ready")
    })
    .await
    .context("timeout waiting for llama-server to become ready")?
}

async fn wait_for_plan_count(
    rx: &mut mpsc::Receiver<String>,
    count: usize,
    max_wait: Duration,
) -> Result<()> {
    let re = regex::Regex::new(&format!(r"recomputed plan: {} assignments", count)).unwrap();
    timeout(max_wait, async {
        while let Some(line) = rx.recv().await {
            if re.is_match(&line) {
                return Ok(());
            }
        }
        bail!(
            "coordinator output closed before plan with {} assignments",
            count
        )
    })
    .await
    .context("timeout waiting for plan")?
}

async fn send_heartbeat(
    client: &mut CoordinatorClient<tonic::transport::Channel>,
    node_id: Uuid,
    ring_addr: SocketAddr,
    rpc_addr: SocketAddr,
) -> Result<()> {
    let cap = NodeCapability {
        node_id,
        tokens_per_sec: 10.0,
        free_ram_bytes: 3_000_000_000,
        reachable: true,
        stability: Stability::Stable,
        last_heartbeat: Instant::now(),
    };

    let mut req = Request::new(ragtag_proto::pb::NodeCapability::from(cap));

    let ring_value = MetadataValue::try_from(ring_addr.to_string().as_str())
        .map_err(|_| anyhow::anyhow!("invalid ring-addr metadata"))?;
    req.metadata_mut().insert("ring-addr", ring_value);

    let rpc_value = MetadataValue::try_from(rpc_addr.to_string().as_str())
        .map_err(|_| anyhow::anyhow!("invalid rpc-addr metadata"))?;
    req.metadata_mut().insert("rpc-addr", rpc_value);

    client.heartbeat(req).await.context("send heartbeat")?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn llama_server_rpc_split_and_completion() -> Result<()> {
    let model = default_model();
    if !model.exists() {
        bail!(
            "model not found at {}; run this test from the nix dev shell",
            model.display()
        );
    }

    let rpc_port_a = find_free_port()?;
    let rpc_port_b = find_free_port()?;
    let llama_port = find_free_port()?;

    let rpc_addr_a = SocketAddr::from(([127, 0, 0, 1], rpc_port_a));
    let rpc_addr_b = SocketAddr::from(([127, 0, 0, 1], rpc_port_b));

    let mut worker_a = spawn_worker(rpc_port_a).await?;
    let mut worker_b = spawn_worker(rpc_port_b).await?;

    let (mut coordinator, grpc_addr, http_addr, mut rx) =
        spawn_coordinator(&model, llama_port).await?;

    let channel = tonic::transport::Channel::from_shared(format!("http://{}", grpc_addr))
        .context("build endpoint")?
        .connect_timeout(Duration::from_secs(2))
        .connect()
        .await
        .context("connect to coordinator")?;
    let mut client = CoordinatorClient::new(channel);

    // Register two fake nodes. Ring addresses are dummies because this test
    // exercises the llama-server proxy path, not the legacy RingTransport.
    let node_a = Uuid::parse_str("11111111-1111-1111-1111-111111111111").unwrap();
    let node_b = Uuid::parse_str("22222222-2222-2222-2222-222222222222").unwrap();

    send_heartbeat(&mut client, node_a, "127.0.0.1:9001".parse()?, rpc_addr_a).await?;
    send_heartbeat(&mut client, node_b, "127.0.0.1:9002".parse()?, rpc_addr_b).await?;

    // Wait for the coordinator to recompute a two-worker plan and start
    // llama-server with the new split.
    wait_for_plan_count(&mut rx, 2, Duration::from_secs(30)).await?;
    wait_for_llama_ready(&mut rx, Duration::from_secs(90)).await?;

    let body = serde_json::json!({
        "model": "tinyllama",
        "messages": [
            {"role": "user", "content": "Hello"},
        ],
        "max_tokens": 16,
    });

    let client = reqwest::Client::new();
    let mut content = None;

    // The first requests may arrive before llama-server has finished its RPC
    // handshake, so retry briefly until we get a real generated response.
    for _ in 0..40 {
        let response = client
            .post(format!("http://{}/v1/chat/completions", http_addr))
            .json(&body)
            .timeout(Duration::from_secs(30))
            .send()
            .await;

        match response {
            Ok(resp) if resp.status().is_success() => {
                let json: serde_json::Value = resp.json().await.unwrap_or_default();
                if let Some(c) = json["choices"][0]["message"]["content"].as_str() {
                    if !c.starts_with("user: Hello") && c.len() > 20 {
                        content = Some(c.to_string());
                        break;
                    }
                }
            }
            _ => {}
        }

        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    let content = content.context("never got a real generated response")?;

    // The no-op ring fallback echoes the prompt as "user: Hello". A real
    // llama-server response is a generated continuation.
    assert!(
        !content.starts_with("user: Hello"),
        "got fallback/no-op response: {}",
        content
    );
    assert!(
        content.len() > 20,
        "response too short to be a real generation: {}",
        content
    );

    worker_a.kill().await.ok();
    worker_b.kill().await.ok();
    coordinator.kill().await.ok();

    Ok(())
}
