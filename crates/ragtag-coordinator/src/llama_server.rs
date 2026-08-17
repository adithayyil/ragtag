//! llama.cpp `llama-server` process manager for the coordinator.
//!
//! This module watches the coordinator's [`PartitionPlan`] and restarts the
//! local `llama-server` process whenever the plan changes, passing it
//! `--rpc` and `--tensor-split` values derived from a single ordered iteration
//! over the plan. The `--tensor-split` vector therefore has exactly one entry
//! per RPC worker, with no leading coordinator slot.

use crate::CoordinatorState;
use anyhow::{Context, Result};
use ragtag_core::PartitionPlan;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::watch;
use tokio::time::{interval, Instant};
use tracing::{info, warn};

/// Manages the lifecycle of the `llama-server` child process.
pub struct LlamaServerManager {
    state: Arc<CoordinatorState>,
    model: PathBuf,
    binary: String,
    host: String,
    port: u16,
    n_gpu_layers: Option<String>,
    plan_rx: watch::Receiver<PartitionPlan>,
}

impl LlamaServerManager {
    /// Create a new manager.
    pub fn new(
        state: Arc<CoordinatorState>,
        model: PathBuf,
        binary: String,
        host: String,
        port: u16,
        n_gpu_layers: Option<String>,
        plan_rx: watch::Receiver<PartitionPlan>,
    ) -> Self {
        Self {
            state,
            model,
            binary,
            host,
            port,
            n_gpu_layers,
            plan_rx,
        }
    }

    /// Run the manager loop until shutdown.
    ///
    /// On each new plan, an existing `llama-server` process is killed and a new
    /// one is started with the correct `--rpc` and `--tensor-split` values.
    pub async fn run(mut self) {
        let mut child: Option<Child> = None;
        let mut last_args: Option<Vec<(SocketAddr, u32)>> = None;

        loop {
            let plan = self.plan_rx.borrow().clone();

            match self.state.rpc_plan(&plan) {
                Ok(args) if !args.is_empty() => {
                    if last_args.as_ref() == Some(&args) {
                        // Same RPC worker set as the running process; keep it.
                    } else {
                        if let Some(mut old) = child.take() {
                            let _ = old.kill().await;
                            let _ = old.wait().await;
                            self.state.log_event("engine stopped");
                        }

                        self.state.log_event("engine restarting...");

                        match self.spawn(&args).await {
                            Ok(mut new_child) => {
                                last_args = Some(args);

                                match Self::wait_ready(self.state.llama_client.clone(), self.url())
                                    .await
                                {
                                    Ok(()) => {
                                        self.set_ready(true);
                                        child = Some(new_child);
                                        info!(url = %self.url(), "llama-server ready");
                                        self.state.log_event("engine ready");
                                    }
                                    Err(e) => {
                                        warn!("llama-server did not become ready: {}", e);
                                        let _ = new_child.kill().await;
                                        let _ = new_child.wait().await;
                                        self.set_ready(false);
                                        last_args = None;
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("failed to start llama-server: {}", e);
                                self.set_ready(false);
                                last_args = None;
                            }
                        }
                    }
                }
                Ok(_) => {
                    if last_args.is_some() || child.is_some() {
                        warn!("partition plan has no RPC workers; stopping llama-server");
                        if let Some(mut old) = child.take() {
                            let _ = old.kill().await;
                            let _ = old.wait().await;
                            self.state.log_event("engine stopped");
                        }
                        last_args = None;
                        self.set_ready(false);
                    }
                }
                Err(e) => {
                    warn!("cannot build RPC args from plan: {}", e);
                    if let Some(mut old) = child.take() {
                        let _ = old.kill().await;
                        let _ = old.wait().await;
                        self.state.log_event("engine stopped");
                    }
                    last_args = None;
                    self.set_ready(false);
                }
            }

            if self.plan_rx.changed().await.is_err() {
                break;
            }

            // A change was signalled. If the new plan yields the same RPC args,
            // keep the current llama-server and wait again.
            let new_plan = self.plan_rx.borrow().clone();
            match self.state.rpc_plan(&new_plan) {
                Ok(new_args) if !new_args.is_empty() => {
                    if last_args.as_ref() == Some(&new_args) {
                        continue;
                    }
                    // Different workers: kill the current process and respawn.
                    if let Some(mut old) = child.take() {
                        let _ = old.kill().await;
                        let _ = old.wait().await;
                        self.state.log_event("engine stopped");
                    }
                    self.set_ready(false);
                    last_args = None;
                    continue;
                }
                _ => {
                    if let Some(mut old) = child.take() {
                        let _ = old.kill().await;
                        let _ = old.wait().await;
                        self.state.log_event("engine stopped");
                    }
                    self.set_ready(false);
                    last_args = None;
                    continue;
                }
            }
        }

        if let Some(mut old) = child {
            let _ = old.kill().await;
            let _ = old.wait().await;
        }
        self.set_ready(false);
    }

    async fn spawn(&self, args: &[(SocketAddr, u32)]) -> Result<Child> {
        let rpc = args
            .iter()
            .map(|(addr, _)| addr.to_string())
            .collect::<Vec<_>>()
            .join(",");

        let tensor_split = args
            .iter()
            .map(|(_, count)| count.to_string())
            .collect::<Vec<_>>()
            .join("/");

        info!(%rpc, %tensor_split, "restarting llama-server");

        let mut cmd = Command::new(&self.binary);
        cmd.arg("-m")
            .arg(&self.model)
            .arg("--rpc")
            .arg(&rpc)
            .arg("--tensor-split")
            .arg(&tensor_split)
            .arg("--split-mode")
            .arg("layer")
            .arg("--host")
            .arg(&self.host)
            .arg("--port")
            .arg(self.port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(n) = &self.n_gpu_layers {
            cmd.arg("--n-gpu-layers").arg(n);
        }

        let mut child = cmd.spawn().context("spawn llama-server")?;

        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            tokio::spawn(async move {
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!(target: "llama-server", "{}", line);
                }
            });
        }

        if let Some(stderr) = child.stderr.take() {
            let reader = BufReader::new(stderr);
            tokio::spawn(async move {
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!(target: "llama-server", "{}", line);
                }
            });
        }

        Ok(child)
    }

    async fn wait_ready(client: reqwest::Client, url: String) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut tick = interval(Duration::from_millis(250));

        loop {
            tick.tick().await;
            match client.get(format!("{}/health", url)).send().await {
                Ok(resp) if resp.status().is_success() => return Ok(()),
                Ok(resp) => {
                    if resp.status().as_u16() != 503 {
                        // Still loading; keep polling.
                        return Err(anyhow::anyhow!(
                            "llama-server /health returned {}",
                            resp.status()
                        ));
                    }
                }
                Err(_) => {}
            }
            if Instant::now() >= deadline {
                return Err(anyhow::anyhow!("llama-server readiness timeout"));
            }
        }
    }

    fn url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    fn set_ready(&self, ready: bool) {
        self.state
            .llama_ready
            .store(ready, std::sync::atomic::Ordering::Relaxed);
    }
}
