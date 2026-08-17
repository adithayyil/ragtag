//! Multi-process mDNS discovery + ring forwarding integration test.
//!
//! Spawns a real `ragtag-coordinator` and two real `ragtag-node` processes,
//! lets them discover each other over loopback multicast, and asserts that:
//!
//! - the coordinator resolves both nodes via mDNS
//! - a two-node partition plan is pushed to the coordinator's `WatchPlan`
//! - an OpenAI-compatible HTTP request actually traverses the ring through
//!   both independent node processes and returns the echoed prompt.

use anyhow::{bail, Context, Result};
use futures::StreamExt;
use ragtag_core::PartitionPlan;
use ragtag_proto::pb::{coordinator_client::CoordinatorClient, NodeId};
use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::{ChildStderr, ChildStdout, Command};
use tokio::sync::mpsc;
use tokio::time::timeout;
use uuid::Uuid;

const COORDINATOR_EXE: &str = env!("CARGO_BIN_EXE_ragtag-coordinator");

async fn read_lines<R: AsyncRead + Unpin + Send + 'static>(reader: R, tx: mpsc::Sender<String>) {
    let mut lines = BufReader::new(reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        if tx.send(line).await.is_err() {
            break;
        }
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

async fn build_node() -> Result<PathBuf> {
    let ws = workspace_root();
    let default_bin = ws.join("target/debug/ragtag-node");
    if default_bin.exists() {
        return Ok(default_bin);
    }

    let e2e_target = ws.join("target/e2e-node");
    let e2e_bin = e2e_target.join("debug/ragtag-node");

    let cargo = std::env::var("CARGO").context("CARGO env var not set")?;
    let status = Command::new(cargo)
        .args(["build", "-p", "ragtag-node", "-q"])
        .current_dir(&ws)
        .env("CARGO_TARGET_DIR", &e2e_target)
        .status()
        .await
        .context("spawn cargo build for ragtag-node")?;
    if !status.success() {
        bail!("cargo build -p ragtag-node failed");
    }

    Ok(e2e_bin)
}

/// Spawn the coordinator binary, returning its gRPC/HTTP addresses and the
/// channel for its stdout/stderr lines.
async fn spawn_coordinator(
    service_type: &str,
) -> Result<(
    tokio::process::Child,
    SocketAddr,
    SocketAddr,
    mpsc::Receiver<String>,
)> {
    let mut child = Command::new(COORDINATOR_EXE)
        .arg("--total-layers")
        .arg("6")
        .arg("--bytes-per-layer")
        .arg("1")
        .arg("--seconds-per-layer")
        .arg("1.0")
        .arg("--heartbeat-timeout-secs")
        .arg("60")
        .arg("--grpc")
        .arg("127.0.0.1:0")
        .arg("--http")
        .arg("127.0.0.1:0")
        .arg("--advertise-host")
        .arg("127.0.0.1")
        .env("RUST_LOG", "info")
        .env("RAGTAG_SERVICE_TYPE", service_type)
        .env("RAGTAG_COORD_SERVICE_TYPE", service_type)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawn coordinator binary")?;

    let (tx, mut rx) = mpsc::channel::<String>(1024);
    tokio::spawn(read_lines::<ChildStdout>(
        child.stdout.take().context("coordinator stdout missing")?,
        tx.clone(),
    ));
    tokio::spawn(read_lines::<ChildStderr>(
        child.stderr.take().context("coordinator stderr missing")?,
        tx,
    ));

    let re = regex::Regex::new(r"grpc_addr=(\S+)\s+http_addr=(\S+)").unwrap();

    let (grpc, http) = loop {
        let line = timeout(Duration::from_secs(10), rx.recv())
            .await
            .context("timeout waiting for coordinator startup")?
            .context("coordinator output closed")?;

        if let Some(caps) = re.captures(&line) {
            break (
                caps[1].parse::<SocketAddr>()?,
                caps[2].parse::<SocketAddr>()?,
            );
        }
    };

    Ok((child, grpc, http, rx))
}

/// Spawn a real `ragtag-node` binary and wait for it to log its listen address
/// and a first partition plan.
async fn spawn_node(
    exe: &Path,
    coordinator: SocketAddr,
    node_id: Uuid,
    service_type: &str,
) -> Result<(tokio::process::Child, SocketAddr)> {
    let mut child = Command::new(exe)
        .arg("--coordinator")
        .arg(coordinator.to_string())
        .arg("--node-id")
        .arg(node_id.to_string())
        .arg("--listen")
        .arg("127.0.0.1:0")
        .arg("--advertise-host")
        .arg("127.0.0.1")
        .arg("--heartbeat-secs")
        .arg("1")
        .arg("--stability")
        .arg("stable")
        .arg("--tokens-per-sec")
        .arg("10.0")
        .arg("--free-ram-bytes")
        .arg("3")
        .env("RUST_LOG", "info")
        .env("RAGTAG_SERVICE_TYPE", service_type)
        .env("RAGTAG_COORD_SERVICE_TYPE", service_type)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawn node binary")?;

    let (tx, mut rx) = mpsc::channel::<String>(1024);
    tokio::spawn(read_lines::<ChildStdout>(
        child.stdout.take().context("node stdout missing")?,
        tx.clone(),
    ));
    tokio::spawn(read_lines::<ChildStderr>(
        child.stderr.take().context("node stderr missing")?,
        tx,
    ));

    let listen_re = regex::Regex::new(r"listen_addr=(\S+)").unwrap();
    let plan_re = regex::Regex::new(r"received new partition plan").unwrap();

    let mut listen_addr = None;
    let mut got_plan = false;
    loop {
        let line = timeout(Duration::from_secs(15), rx.recv())
            .await
            .context("timeout waiting for node startup")?
            .context("node output closed")?;

        if let Some(caps) = listen_re.captures(&line) {
            listen_addr = Some(caps[1].parse::<SocketAddr>()?);
        }
        if plan_re.is_match(&line) {
            got_plan = true;
        }
        if listen_addr.is_some() && got_plan {
            break;
        }
    }

    // Keep the log drain alive for the remainder of the test.
    tokio::spawn(async move { while rx.recv().await.is_some() {} });

    Ok((child, listen_addr.context("node listen_addr not found")?))
}

async fn wait_for_plan(
    client: &mut CoordinatorClient<tonic::transport::Channel>,
    predicate: impl Fn(&PartitionPlan) -> bool,
    max_wait: Duration,
) -> Result<PartitionPlan> {
    let req = NodeId::from(Uuid::new_v4());
    let mut stream = client
        .watch_plan(req)
        .await
        .context("watch plan")?
        .into_inner();

    timeout(max_wait, async {
        while let Some(msg) = stream.next().await {
            let msg = msg.context("watch plan stream")?;
            let plan = PartitionPlan::try_from(msg).context("convert plan")?;
            if predicate(&plan) {
                return Ok(plan);
            }
        }
        bail!("watch plan stream ended before matching plan")
    })
    .await
    .context("timeout waiting for plan")?
}

#[tokio::test(flavor = "multi_thread")]
async fn mdns_two_node_ring_round_trip() -> Result<()> {
    // Unique mDNS service type per run so parallel test invocations do not
    // discover each other's nodes. mDNS service names must be <= 15 bytes
    // (including the leading underscore), so keep this short.
    let short_id = Uuid::new_v4()
        .simple()
        .to_string()
        .chars()
        .take(6)
        .collect::<String>();
    let service_type = format!("_ragtag-{}._tcp.local.", short_id);

    let (mut coordinator, grpc_addr, http_addr, mut coord_rx) =
        spawn_coordinator(&service_type).await?;
    let node_exe = build_node().await?;

    let id_a = Uuid::new_v4();
    let id_b = Uuid::new_v4();

    let (mut node_a, addr_a) = spawn_node(&node_exe, grpc_addr, id_a, &service_type).await?;
    let (mut node_b, addr_b) = spawn_node(&node_exe, grpc_addr, id_b, &service_type).await?;

    // Wait for the coordinator to resolve both nodes over mDNS.
    let mut resolved = HashSet::new();
    while resolved.len() < 2 {
        let line = timeout(Duration::from_secs(20), coord_rx.recv())
            .await
            .context("timeout waiting for mDNS resolution")?
            .context("coordinator output closed")?;

        if line.contains("mdns service resolved") {
            for id in [id_a, id_b] {
                if line.contains(&id.to_string()) {
                    resolved.insert(id);
                }
            }
        }
    }

    // Drain coordinator logs so the child never writes to a broken pipe.
    tokio::spawn(async move { while coord_rx.recv().await.is_some() {} });

    let channel = tonic::transport::Channel::from_shared(format!("http://{}", grpc_addr))
        .context("build endpoint")?
        .connect_timeout(Duration::from_secs(2))
        .connect()
        .await
        .context("connect to coordinator")?;
    let mut client = CoordinatorClient::new(channel);

    let plan = wait_for_plan(
        &mut client,
        |p| {
            p.total_layers() == 6
                && p.assignments.len() == 2
                && p.assignments.iter().any(|a| a.node_id == id_a)
                && p.assignments.iter().any(|a| a.node_id == id_b)
        },
        Duration::from_secs(10),
    )
    .await?;

    // The ring must point at real node addresses (both addresses appear as
    // next hops in the two-node ring).
    let next_hops: HashSet<_> = plan.assignments.iter().map(|a| a.next_hop).collect();
    assert_eq!(next_hops.len(), 2);
    assert!(next_hops.contains(&addr_a));
    assert!(next_hops.contains(&addr_b));

    let body = serde_json::json!({
        "model": "mock",
        "messages": [
            {"role": "user", "content": "hello from the ring"},
        ],
    });

    let response = reqwest::Client::new()
        .post(format!("http://{}/v1/chat/completions", http_addr))
        .json(&body)
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .context("post to http endpoint")?;

    assert!(
        response.status().is_success(),
        "http status: {}",
        response.status()
    );

    let json: serde_json::Value = response.json().await.context("parse http response")?;
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .context("missing content in response")?;

    assert_eq!(content, "user: hello from the ring");

    node_a.kill().await.ok();
    node_b.kill().await.ok();
    coordinator.kill().await.ok();
    Ok(())
}
