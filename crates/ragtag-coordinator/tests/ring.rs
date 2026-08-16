//! End-to-end integration test for the coordinator.
//!
//! Spawns the `ragtag-coordinator` binary, sends heartbeats from three fake
//! nodes, waits for partition plans to be pushed, then marks one node
//! opportunistic and asserts the scheduler can exclude it under a tight
//! combined-RAM scenario.

use anyhow::{bail, Context, Result};
use futures::StreamExt;
use ragtag_core::{NodeCapability, PartitionPlan, Stability};
use ragtag_proto::pb::{coordinator_client::CoordinatorClient, NodeId};
use ragtag_scheduler::{is_valid_plan, LinearCostModel};
use std::net::SocketAddr;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::{ChildStderr, ChildStdout, Command};
use tokio::sync::mpsc;
use tokio::time::timeout;
use uuid::Uuid;

const EXE: &str = env!("CARGO_BIN_EXE_ragtag-coordinator");

async fn read_lines<R: AsyncRead + Unpin + Send + 'static>(reader: R, tx: mpsc::Sender<String>) {
    let mut lines = BufReader::new(reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        if tx.send(line).await.is_err() {
            break;
        }
    }
}

/// Spawn the coordinator binary, returning its gRPC and HTTP addresses.
async fn spawn_coordinator() -> Result<(tokio::process::Child, SocketAddr, SocketAddr)> {
    let mut child = Command::new(EXE)
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
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawn coordinator binary")?;

    let (tx, mut rx) = mpsc::channel::<String>(64);
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

    // Keep the log drain alive for the remainder of the test so the
    // coordinator's tracing subscriber never writes to a broken pipe.
    tokio::spawn(async move { while rx.recv().await.is_some() {} });

    Ok((child, grpc, http))
}

fn make_capability(
    node_id: Uuid,
    tokens_per_sec: f32,
    free_ram_bytes: u64,
    stability: Stability,
) -> NodeCapability {
    NodeCapability {
        node_id,
        tokens_per_sec,
        free_ram_bytes,
        reachable: true,
        stability,
        last_heartbeat: Instant::now(),
    }
}

async fn next_plan(
    stream: &mut tonic::Streaming<ragtag_proto::pb::PartitionPlan>,
    max_wait: Duration,
) -> Result<PartitionPlan> {
    timeout(max_wait, async {
        while let Some(msg) = stream.next().await {
            let msg = msg.context("watch plan stream")?;
            let plan = PartitionPlan::try_from(msg)?;
            if !plan.assignments.is_empty() {
                return Ok(plan);
            }
        }
        bail!("watch plan stream ended before a non-empty plan")
    })
    .await
    .context("timeout waiting for partition plan")?
}

#[tokio::test]
async fn heartbeat_schedule_plan_push_and_exclusion() -> Result<()> {
    let (mut child, grpc_addr, _http_addr) = spawn_coordinator().await?;

    let channel = tonic::transport::Channel::from_shared(format!("http://{}", grpc_addr))
        .context("build endpoint")?
        .connect_timeout(Duration::from_secs(2))
        .connect()
        .await
        .context("connect to coordinator")?;
    let mut client = CoordinatorClient::new(channel);

    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();

    // Two fast stable nodes with 3 bytes each, one slow opportunistic node
    // with 6 bytes. The model requires 6 layers x 1 byte = 6 bytes, so the
    // two fast nodes can cover it and the slow node should be excluded.
    let caps = [
        make_capability(a, 10.0, 3, Stability::Stable),
        make_capability(b, 5.0, 3, Stability::Stable),
        make_capability(c, 1.0, 3, Stability::Stable),
    ];

    for cap in &caps {
        client
            .heartbeat(ragtag_proto::pb::NodeCapability::from(cap.clone()))
            .await
            .context("send heartbeat")?;
    }

    // Subscribe to plan updates for each node.
    let mut streams = Vec::new();
    for cap in &caps {
        let stream = client
            .watch_plan(NodeId::from(cap.node_id))
            .await
            .context("watch plan request")?
            .into_inner();
        streams.push(stream);
    }

    // All three nodes should receive the same initial plan.
    let mut plans = Vec::new();
    for (i, stream) in streams.iter_mut().enumerate() {
        let plan = next_plan(stream, Duration::from_secs(2))
            .await
            .with_context(|| format!("wait for initial plan for node {}", i))?;
        plans.push(plan);
    }

    let first_plan = plans[0].clone();
    assert_eq!(first_plan.total_layers(), 6);
    for plan in &plans {
        assert_eq!(plan, &first_plan);
    }
    assert!(is_valid_plan(
        &first_plan,
        6,
        &LinearCostModel::new(1, 1.0),
        &caps
    ));

    // Now mark the third node opportunistic and give it enough RAM for the
    // full model. It is much slower, so the scheduler should still prefer the
    // two fast stable nodes.
    let updated = make_capability(c, 1.0, 6, Stability::Opportunistic);
    client
        .heartbeat(ragtag_proto::pb::NodeCapability::from(updated))
        .await
        .context("send opportunistic heartbeat")?;

    for (i, stream) in streams.iter_mut().enumerate() {
        let new_plan = next_plan(stream, Duration::from_secs(2))
            .await
            .with_context(|| format!("wait for updated plan for node {}", i))?;
        assert!(
            !new_plan.assignments.iter().any(|r| r.node_id == c),
            "opportunistic node should be excluded"
        );
        assert!(is_valid_plan(
            &new_plan,
            6,
            &LinearCostModel::new(1, 1.0),
            &[
                make_capability(a, 10.0, 3, Stability::Stable),
                make_capability(b, 5.0, 3, Stability::Stable),
            ]
        ));
    }

    child.kill().await.ok();
    Ok(())
}
