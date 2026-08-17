//! The `ragtag-node` agent binary.
//!
//! It advertises itself over mDNS, heartbeats its capabilities to a
//! coordinator, and runs a `RingTransport` server that executes its assigned
//! transformer layers.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use clap::{Parser, ValueEnum};
use futures::{Stream, StreamExt};
use mdns_sd::{ServiceDaemon, ServiceInfo};
use ragtag_core::{NodeCapability, PartitionPlan, Stability};
use ragtag_proto::pb::{
    coordinator_client::CoordinatorClient,
    ring_transport_client::RingTransportClient,
    ring_transport_server::{RingTransport, RingTransportServer},
    ActivationChunk,
};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::System;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch};
use tokio::time::{interval, MissedTickBehavior};
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::metadata::MetadataValue;
use tonic::{transport::Server, Request, Response, Status, Streaming};
use tracing::{info, warn};
use uuid::Uuid;

mod llama_engine;

const DEFAULT_SERVICE_TYPE: &str = "_ragtag._tcp.local.";

fn service_type() -> String {
    std::env::var("RAGTAG_SERVICE_TYPE").unwrap_or_else(|_| DEFAULT_SERVICE_TYPE.to_string())
}

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

/// Command-line configuration for a `ragtag-node` instance.
#[derive(Parser, Debug)]
#[command(name = "ragtag-node", version)]
struct Args {
    /// Socket address the ring transport server should bind to.
    #[arg(long, default_value = "127.0.0.1:0")]
    listen: SocketAddr,
    /// Coordinator gRPC address.
    #[arg(long)]
    coordinator: SocketAddr,
    /// Hostname or IP to advertise via mDNS.
    #[arg(long, default_value = "127.0.0.1")]
    advertise_host: String,
    /// Hostname to report to the coordinator for display.
    #[arg(long)]
    hostname: Option<String>,
    /// URL of an external llama.cpp HTTP server (e.g. http://127.0.0.1:8080).
    /// When set, the node sends prompts to this server instead of using the
    /// no-op engine. The server must be started separately with the desired
    /// GGUF model.
    #[arg(long)]
    llama_server: Option<String>,
    /// Address of this node's llama.cpp RPC worker. The coordinator uses this
    /// to build the `--rpc` and `--tensor-split` arguments for its local
    /// `llama-server`.
    #[arg(long, default_value = "127.0.0.1:50052")]
    rpc_addr: SocketAddr,
    /// Path or name of the `llama-rpc-server` binary. When set, the node will
    /// spawn and supervise a worker on `--rpc-addr` before sending heartbeats.
    #[arg(long)]
    llama_rpc_server: Option<String>,
    /// Device string passed to `llama-rpc-server --device`. Examples: `CPU`,
    /// `BLAS`, `CUDA0`, `Vulkan0`. Defaults to `CPU`.
    #[arg(long, default_value = "CPU")]
    llama_device: String,
    /// Override the automatically generated node id.
    #[arg(long)]
    node_id: Option<Uuid>,
    /// Self-reported stability class.
    #[arg(long, value_enum, default_value = "stable")]
    stability: StabilityArg,
    /// Override the matmul-derived tokens-per-second value.
    #[arg(long)]
    tokens_per_sec: Option<f32>,
    /// Override the detected free RAM value (bytes).
    #[arg(long)]
    free_ram_bytes: Option<u64>,
    /// Seconds between heartbeats.
    #[arg(long, default_value_t = 5)]
    heartbeat_secs: u64,
    /// Matrix side length for the startup benchmark.
    #[arg(long, default_value_t = 128)]
    matmul_size: usize,
    /// Number of matmul iterations for the startup benchmark.
    #[arg(long, default_value_t = 5)]
    matmul_iters: usize,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum StabilityArg {
    #[default]
    Stable,
    Opportunistic,
}

impl From<StabilityArg> for Stability {
    fn from(v: StabilityArg) -> Self {
        match v {
            StabilityArg::Stable => Stability::Stable,
            StabilityArg::Opportunistic => Stability::Opportunistic,
        }
    }
}

/// Trait for the layer-execution engine. This is the integration point for a
/// real llama.cpp backend; the default implementation is a pass-through used in
/// tests and when no model is supplied.
#[async_trait]
trait Engine: Send + Sync {
    async fn forward(&self, chunk: ActivationChunk) -> Result<ActivationChunk>;
}

#[derive(Clone)]
struct NoopEngine;

#[async_trait]
impl Engine for NoopEngine {
    async fn forward(&self, chunk: ActivationChunk) -> Result<ActivationChunk> {
        Ok(chunk)
    }
}

/// Shared node state used by the ring-transport gRPC handler.
#[derive(Clone)]
struct NodeState {
    node_id: Uuid,
    engine: Arc<dyn Engine + Send + Sync>,
    plan: Arc<watch::Receiver<Option<PartitionPlan>>>,
}

impl NodeState {
    async fn process_and_forward(
        &self,
        chunk: ActivationChunk,
        tx: mpsc::Sender<Result<ActivationChunk, Status>>,
    ) -> Result<()> {
        let plan = self
            .plan
            .borrow()
            .clone()
            .context("no partition plan available")?;

        let my_idx = plan
            .assignments
            .iter()
            .position(|a| a.node_id == self.node_id)
            .context("this node is not in the current plan")?;

        let plan_len = plan.assignments.len();
        let is_entry = my_idx == 0;
        let next_hop = plan.assignments[my_idx].next_hop;

        // A chunk that has completed the full ring returns to the entry node.
        if is_entry && chunk.sequence_no >= plan_len as u64 {
            tx.send(Ok(chunk))
                .await
                .map_err(|_| anyhow::anyhow!("output channel closed"))?;
            return Ok(());
        }

        let mut out = self.engine.forward(chunk).await?;
        out.sequence_no += 1;

        // Single-node ring: just return after executing.
        if plan_len == 1 {
            tx.send(Ok(out))
                .await
                .map_err(|_| anyhow::anyhow!("output channel closed"))?;
            return Ok(());
        }

        // Forward to the next hop and stream its responses back upstream.
        let client_addr = format!("http://{}", next_hop);
        let mut client = RingTransportClient::connect(client_addr)
            .await
            .context("connect to next hop")?;
        let req = Request::new(futures::stream::iter(vec![out]));
        let mut stream = client.forward(req).await?.into_inner();

        while let Some(item) = stream.next().await {
            tx.send(item)
                .await
                .map_err(|_| anyhow::anyhow!("output channel closed"))?;
        }

        Ok(())
    }
}

#[async_trait]
impl RingTransport for NodeState {
    type ForwardStream = Pin<Box<dyn Stream<Item = Result<ActivationChunk, Status>> + Send>>;

    async fn forward(
        &self,
        request: Request<Streaming<ActivationChunk>>,
    ) -> std::result::Result<Response<Self::ForwardStream>, Status> {
        let mut input = request.into_inner();
        let (tx, rx) = mpsc::channel::<Result<ActivationChunk, Status>>(16);
        let state = self.clone();

        tokio::spawn(async move {
            while let Some(result) = input.next().await {
                match result {
                    Ok(chunk) => {
                        let chunk_tx = tx.clone();
                        if let Err(e) = state.process_and_forward(chunk, chunk_tx).await {
                            let _ = tx.send(Err(Status::internal(e.to_string()))).await;
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

/// Run a short matmul benchmark and read the available RAM from `sysinfo`.
fn profile(matmul_size: usize, iters: usize) -> Result<(f32, u64)> {
    let mut sys = System::new_all();
    sys.refresh_all();
    let free_ram = sys.free_memory();

    let n = matmul_size;
    let mut a = vec![0.0_f32; n * n];
    let mut b = vec![0.0_f32; n * n];
    let mut c = vec![0.0_f32; n * n];

    for i in 0..n * n {
        a[i] = rand::random::<f32>();
        b[i] = rand::random::<f32>();
    }

    let start = Instant::now();
    for _ in 0..iters {
        for i in 0..n {
            for k in 0..n {
                let a_ik = a[i * n + k];
                for j in 0..n {
                    c[i * n + j] += a_ik * b[k * n + j];
                }
            }
        }
    }
    let elapsed = start.elapsed().as_secs_f32().max(f32::EPSILON);

    let flops = 2.0_f64 * (n as f64).powi(3) * iters as f64;
    let gflops = (flops / 1e9) / elapsed as f64;

    Ok((gflops as f32, free_ram))
}

fn build_capability(args: &Args, node_id: Uuid) -> Result<NodeCapability> {
    let (tokens_per_sec, free_ram_bytes) =
        if let (Some(tps), Some(ram)) = (args.tokens_per_sec, args.free_ram_bytes) {
            (tps, ram)
        } else {
            profile(args.matmul_size, args.matmul_iters)?
        };

    Ok(NodeCapability {
        node_id,
        tokens_per_sec,
        free_ram_bytes,
        reachable: true,
        stability: args.stability.into(),
        last_heartbeat: Instant::now(),
    })
}

fn build_engine(args: &Args) -> Result<Arc<dyn Engine + Send + Sync>> {
    if let Some(url) = &args.llama_server {
        info!(%url, "using llama.cpp HTTP server");
        return Ok(Arc::new(llama_engine::LlamaEngine::new(url.clone())));
    }
    Ok(Arc::new(NoopEngine))
}

fn advertise(
    node_id: Uuid,
    capability: &NodeCapability,
    listen_ip: IpAddr,
    port: u16,
    rpc_addr: SocketAddr,
    hostname: &str,
) -> Result<()> {
    let mdns = ServiceDaemon::new().context("create mdns daemon")?;

    let stability = format!("{:?}", capability.stability).to_lowercase();
    let properties: HashMap<String, String> = [
        ("node_id", node_id.to_string()),
        ("tokens_per_sec", capability.tokens_per_sec.to_string()),
        ("free_ram_bytes", capability.free_ram_bytes.to_string()),
        ("stability", stability),
        ("rpc_addr", rpc_addr.to_string()),
        ("hostname", hostname.to_string()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();

    let host = format!("{}.local.", hostname);
    let info = ServiceInfo::new(
        &service_type(),
        hostname,
        &host,
        listen_ip,
        port,
        properties,
    )
    .context("create mdns service info")?;

    mdns.register(info).context("register mdns service")?;

    // The mDNS daemon stops when its handle is dropped, so keep it alive in a
    // background thread.
    std::thread::spawn(move || {
        let _keep = mdns;
        loop {
            std::thread::sleep(Duration::from_secs(60));
        }
    });

    Ok(())
}

async fn run_heartbeats(
    mut client: CoordinatorClient<tonic::transport::Channel>,
    capability: NodeCapability,
    node_id: Uuid,
    listen_addr: SocketAddr,
    rpc_addr: SocketAddr,
    heartbeat_secs: u64,
    hostname: String,
) -> Result<()> {
    let mut timer = interval(Duration::from_secs(heartbeat_secs));
    timer.set_missed_tick_behavior(MissedTickBehavior::Delay);

    loop {
        timer.tick().await;

        let mut cap = capability.clone();
        cap.last_heartbeat = Instant::now();

        let mut req = Request::new(ragtag_proto::pb::NodeCapability::from(cap));

        let ring_addr = listen_addr.to_string();
        let ring_value = MetadataValue::try_from(ring_addr.as_str())
            .map_err(|_| anyhow::anyhow!("invalid ring-addr metadata"))?;
        req.metadata_mut().insert("ring-addr", ring_value);

        let rpc = rpc_addr.to_string();
        let rpc_value = MetadataValue::try_from(rpc.as_str())
            .map_err(|_| anyhow::anyhow!("invalid rpc-addr metadata"))?;
        req.metadata_mut().insert("rpc-addr", rpc_value);

        let host_value = MetadataValue::try_from(hostname.as_str())
            .map_err(|_| anyhow::anyhow!("invalid hostname metadata"))?;
        req.metadata_mut().insert("hostname", host_value);

        client.heartbeat(req).await.context("send heartbeat")?;

        info!(%node_id, "heartbeat sent");
    }
}

async fn watch_plans(
    mut client: CoordinatorClient<tonic::transport::Channel>,
    node_id: Uuid,
    plan_tx: watch::Sender<Option<PartitionPlan>>,
) -> Result<()> {
    let req = ragtag_proto::pb::NodeId::from(node_id);
    let mut stream = client.watch_plan(req).await?.into_inner();

    while let Some(msg) = stream.next().await {
        let msg = msg.context("watch plan stream")?;
        let plan = PartitionPlan::try_from(msg).context("convert plan")?;
        let _ = plan_tx.send(Some(plan));
        info!(%node_id, "received new partition plan");
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn heartbeat_loop(
    coordinator: SocketAddr,
    node_id: Uuid,
    capability: NodeCapability,
    plan_tx: watch::Sender<Option<PartitionPlan>>,
    listen_addr: SocketAddr,
    rpc_addr: SocketAddr,
    heartbeat_secs: u64,
    hostname: String,
) {
    let mut backoff = Duration::from_secs(1);

    loop {
        let addr = format!("http://{}", coordinator);
        match CoordinatorClient::connect(addr).await {
            Ok(client) => {
                backoff = Duration::from_secs(1);
                let watch_client = client.clone();
                let watch = tokio::spawn(watch_plans(watch_client, node_id, plan_tx.clone()));

                if let Err(e) = run_heartbeats(
                    client,
                    capability.clone(),
                    node_id,
                    listen_addr,
                    rpc_addr,
                    heartbeat_secs,
                    hostname.clone(),
                )
                .await
                {
                    warn!(%node_id, "heartbeat loop ended: {}", e);
                }

                watch.abort();
            }
            Err(e) => {
                warn!(%node_id, "failed to connect to coordinator {}: {}", coordinator, e);
            }
        }

        tokio::time::sleep(backoff).await;
        backoff = std::cmp::min(backoff * 2, Duration::from_secs(60));
    }
}

async fn run_rpc_server(
    binary: &str,
    host: IpAddr,
    port: u16,
    device: &str,
) -> Result<tokio::process::Child> {
    let mut child = tokio::process::Command::new(binary)
        .arg("--host")
        .arg(host.to_string())
        .arg("--port")
        .arg(port.to_string())
        .arg("--device")
        .arg(device)
        .arg("--cache")
        .spawn()
        .context("spawn llama-rpc-server")?;

    // Give the worker a moment to fail if the port is in use.
    tokio::time::sleep(Duration::from_millis(500)).await;
    if let Ok(Some(status)) = child.try_wait() {
        bail!("llama-rpc-server exited early: {}", status);
    }

    Ok(child)
}

async fn run_server(listener: TcpListener, state: NodeState) -> Result<()> {
    Server::builder()
        .add_service(RingTransportServer::new(state))
        .serve_with_incoming(TcpListenerStream::new(listener))
        .await
        .context("ring transport server")?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_ansi(false).init();
    let args = Args::parse();

    let node_id = args.node_id.unwrap_or_else(Uuid::new_v4);
    let hostname = args.hostname.clone().unwrap_or_else(system_hostname);
    let capability = build_capability(&args, node_id)?;

    let (plan_tx, plan_rx) = watch::channel(None);
    let engine = build_engine(&args)?;

    let listener = TcpListener::bind(args.listen)
        .await
        .context("bind ring transport socket")?;
    let listen_addr = listener.local_addr()?;

    let state = NodeState {
        node_id,
        engine,
        plan: Arc::new(plan_rx),
    };

    let advertise_ip: IpAddr = args
        .advertise_host
        .parse()
        .context("parse advertise-host as IP")?;
    advertise(
        node_id,
        &capability,
        advertise_ip,
        listen_addr.port(),
        args.rpc_addr,
        &hostname,
    )?;

    info!(%node_id, %listen_addr, rpc_addr = %args.rpc_addr, "node starting");

    let mut rpc_child = None;
    if let Some(binary) = &args.llama_rpc_server {
        rpc_child = Some(run_rpc_server(binary, advertise_ip, args.rpc_addr.port(), &args.llama_device).await?);
    }

    let heartbeat = tokio::spawn(heartbeat_loop(
        args.coordinator,
        node_id,
        capability,
        plan_tx,
        listen_addr,
        args.rpc_addr,
        args.heartbeat_secs,
        hostname,
    ));
    let server = tokio::spawn(run_server(listener, state));

    tokio::select! {
        _ = tokio::signal::ctrl_c() => info!("shutting down"),
        _ = heartbeat => {},
        _ = server => {},
    }

    if let Some(mut child) = rpc_child {
        let _ = child.kill().await;
        let _ = child.wait().await;
    }

    Ok(())
}
