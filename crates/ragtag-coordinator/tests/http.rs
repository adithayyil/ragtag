//! End-to-end integration test for the OpenAI-compatible HTTP endpoint.
//!
//! Spawns the `ragtag-coordinator` binary, registers a mock ring-transport
//! node by sending a heartbeat with a `ring-addr` metadata header, and then
//! POSTs to `/v1/chat/completions` and checks that the response contains the
//! prompt echoed back by the mock engine.

use anyhow::{Context, Result};
use futures::{Stream, StreamExt};
use ragtag_core::{NodeCapability, Stability};
use ragtag_proto::pb::{
    coordinator_client::CoordinatorClient,
    ring_transport_server::{RingTransport, RingTransportServer},
    ActivationChunk,
};
use std::net::SocketAddr;
use std::pin::Pin;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::net::TcpListener;
use tokio::process::{ChildStderr, ChildStdout, Command};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_stream::wrappers::ReceiverStream;
use tonic::metadata::MetadataValue;
use tonic::{transport::Server, Request, Response, Status, Streaming};
use tracing::warn;
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

/// A mock ring-transport node. It acts like a single-node ring with a no-op
/// engine: each incoming `ActivationChunk` is returned with its `sequence_no`
/// incremented.
struct MockRing;

#[tonic::async_trait]
impl RingTransport for MockRing {
    type ForwardStream = Pin<Box<dyn Stream<Item = Result<ActivationChunk, Status>> + Send>>;

    async fn forward(
        &self,
        request: Request<Streaming<ActivationChunk>>,
    ) -> std::result::Result<Response<Self::ForwardStream>, Status> {
        let mut input = request.into_inner();
        let (tx, rx) = mpsc::channel::<Result<ActivationChunk, Status>>(16);

        tokio::spawn(async move {
            while let Some(result) = input.next().await {
                match result {
                    Ok(mut chunk) => {
                        chunk.sequence_no += 1;
                        if tx.send(Ok(chunk)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                }
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

/// Start a mock ring-transport gRPC server on an ephemeral loopback port.
async fn spawn_mock_ring() -> Result<(SocketAddr, tokio::task::JoinHandle<()>)> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind mock ring server")?;
    let addr = listener.local_addr()?;

    let handle = tokio::spawn(async move {
        if let Err(e) = Server::builder()
            .add_service(RingTransportServer::new(MockRing))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
        {
            warn!("mock ring server error: {}", e);
        }
    });

    Ok((addr, handle))
}

#[tokio::test(flavor = "multi_thread")]
async fn http_chat_completion_round_trip() -> Result<()> {
    let (mut child, grpc_addr, http_addr) = spawn_coordinator().await?;
    let (ring_addr, _ring_server) = spawn_mock_ring().await?;

    let channel = tonic::transport::Channel::from_shared(format!("http://{}", grpc_addr))
        .context("build endpoint")?
        .connect_timeout(Duration::from_secs(2))
        .connect()
        .await
        .context("connect to coordinator")?;
    let mut client = CoordinatorClient::new(channel);

    let node_id = Uuid::new_v4();
    let cap = NodeCapability {
        node_id,
        tokens_per_sec: 10.0,
        free_ram_bytes: 1024,
        reachable: true,
        stability: Stability::Stable,
        last_heartbeat: Instant::now(),
    };

    let mut req = Request::new(ragtag_proto::pb::NodeCapability::from(cap));
    let ring_addr_str = ring_addr.to_string();
    let value = MetadataValue::try_from(ring_addr_str.as_str())
        .map_err(|_| anyhow::anyhow!("invalid ring-addr metadata"))?;
    req.metadata_mut().insert("ring-addr", value);

    client
        .heartbeat(req)
        .await
        .context("send heartbeat with ring-addr")?;

    let body = serde_json::json!({
        "model": "mock",
        "messages": [
            {"role": "user", "content": "hello from the ring"},
        ],
    });

    let response = reqwest::Client::new()
        .post(format!("http://{}/v1/chat/completions", http_addr))
        .json(&body)
        .timeout(Duration::from_secs(5))
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

    child.kill().await.ok();
    Ok(())
}
