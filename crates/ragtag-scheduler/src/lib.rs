//! DP-based pipeline layer partition scheduler.
//!
//! The scheduler takes a ring-ordered slice of [`NodeCapability`] values and
//! decides which contiguous slice of transformer layers each device should own.
//! A device may receive zero layers, which is how the scheduler can exclude a
//! slow or opportunistic node when doing so lowers the bottleneck time.
#![warn(missing_docs)]

use ragtag_core::{LayerRange, NodeCapability, PartitionPlan, Stability};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Instant;
use thiserror::Error;
use uuid::Uuid;

pub mod cost;
pub use cost::{CostModel, LinearCostModel};

/// Validated scheduler input.
///
/// Constructing one enforces that the coordinator's own node id is not in the
/// schedulable set. This prevents a silent off-by-one in the llama.cpp
/// `--tensor-split` vector: if the coordinator were present, the first entry
/// would be consumed as a leading zero and all layers would land on a single
/// worker (see the `CoordinatorInNodeTable` error and the manual b10273
/// tensor-split tests).
#[derive(Debug, Clone)]
pub struct ScheduledInput {
    devices: Vec<NodeCapability>,
    addresses: HashMap<Uuid, SocketAddr>,
}

impl ScheduledInput {
    /// Build a validated input.
    ///
    /// `coordinator_id` is filtered from the schedulable set. If it appears in
    /// `ring_devices` the constructor returns `CoordinatorInNodeTable`, making
    /// the misconfiguration fail loudly instead of silently mis-splitting the
    /// model.
    pub fn new(
        coordinator_id: Uuid,
        ring_devices: &[NodeCapability],
        addresses: &HashMap<Uuid, SocketAddr>,
    ) -> Result<Self, ScheduleError> {
        if ring_devices.iter().any(|d| d.node_id == coordinator_id) {
            return Err(ScheduleError::CoordinatorInNodeTable(coordinator_id));
        }

        let device_ids: std::collections::HashSet<_> = ring_devices
            .iter()
            .filter(|d| d.reachable)
            .map(|d| d.node_id)
            .collect();

        let devices: Vec<_> = ring_devices
            .iter()
            .filter(|d| d.reachable && d.node_id != coordinator_id)
            .cloned()
            .collect();

        let addresses: HashMap<_, _> = addresses
            .iter()
            .filter(|(id, _)| **id != coordinator_id && device_ids.contains(id))
            .map(|(id, addr)| (*id, *addr))
            .collect();

        Ok(Self { devices, addresses })
    }

    /// Devices that are allowed to receive layers.
    pub fn devices(&self) -> &[NodeCapability] {
        &self.devices
    }

    /// Address map for the devices in `devices()`.
    pub fn addresses(&self) -> &HashMap<Uuid, SocketAddr> {
        &self.addresses
    }
}

/// Errors returned by the scheduler.
#[derive(Debug, Error)]
pub enum ScheduleError {
    /// There are no reachable devices to schedule onto.
    #[error("no reachable devices")]
    NoDevices,
    /// The combined memory (or token rate) of reachable devices cannot fit the
    /// full model.
    #[error("no feasible partition fits all {0} layers on reachable devices")]
    Infeasible(u32),
    /// A node that was assigned layers does not have a known ring address.
    #[error("missing ring address for node {0}")]
    MissingAddress(Uuid),
    /// The coordinator's own node id was present in the schedulable set. If it
    /// is not excluded, llama.cpp's `--tensor-split` vector would silently get
    /// an extra leading entry and place all layers on one worker.
    #[error("coordinator node {0} must not be in the schedulable node table")]
    CoordinatorInNodeTable(Uuid),
}

/// Solve the layer partition problem with a PipeEdge-style DP recurrence.
///
/// `dp[i][j] = min(dp[i][j-1], min_{0 <= k < i} max(dp[k][j-1], cost(k+1..i, dev_j)))`
///
/// where `i` is the number of layers already assigned and `j` is the number of
/// devices considered so far. `dp[i][j-1]` represents giving device `j` zero
/// layers, which is how a device can be excluded.
///
/// # Arguments
///
/// * `layer_count` - total number of transformer layers (`0..layer_count`).
/// * `input` - validated scheduler input (see [`ScheduledInput`]).
/// * `cost_model` - the cost model that encodes memory and speed constraints.
///
/// # Example
///
/// ```
/// use ragtag_core::{NodeCapability, Stability};
/// use ragtag_scheduler::{solve, LinearCostModel, ScheduledInput};
/// use std::collections::HashMap;
/// use std::net::SocketAddr;
/// use std::time::Instant;
/// use uuid::Uuid;
///
/// let a = Uuid::new_v4();
/// let b = Uuid::new_v4();
/// let c = Uuid::new_v4();
///
/// let devices = vec![
///     NodeCapability {
///         node_id: a,
///         tokens_per_sec: 10.0,
///         free_ram_bytes: 1024,
///         reachable: true,
///         stability: Stability::Stable,
///         last_heartbeat: Instant::now(),
///     },
///     NodeCapability {
///         node_id: b,
///         tokens_per_sec: 5.0,
///         free_ram_bytes: 1024,
///         reachable: true,
///         stability: Stability::Stable,
///         last_heartbeat: Instant::now(),
///     },
///     NodeCapability {
///         node_id: c,
///         tokens_per_sec: 1.0,
///         free_ram_bytes: 1024,
///         reachable: true,
///         stability: Stability::Opportunistic,
///         last_heartbeat: Instant::now(),
///     },
/// ];
///
/// let mut addresses = HashMap::new();
/// for (id, port) in [(a, 8001), (b, 8002), (c, 8003)] {
///     addresses.insert(id, SocketAddr::from(([127, 0, 0, 1], port)));
/// }
///
/// let input = ScheduledInput::new(Uuid::nil(), &devices, &addresses).unwrap();
/// let plan = solve(6, &input, &LinearCostModel::new(1, 1.0)).unwrap();
///
/// // The slow opportunistic node should be excluded.
/// assert!(!plan.assignments.iter().any(|r| r.node_id == c));
/// ```
pub fn solve(
    layer_count: u32,
    input: &ScheduledInput,
    cost_model: &dyn CostModel,
) -> Result<PartitionPlan, ScheduleError> {
    let devices: Vec<&NodeCapability> = input.devices.iter().collect();

    if layer_count == 0 {
        return Ok(PartitionPlan {
            assignments: Vec::new(),
        });
    }

    if devices.is_empty() {
        return Err(ScheduleError::NoDevices);
    }

    let n = layer_count as usize;
    let m = devices.len();

    // dp[i][j] = bottleneck for the first i layers using the first j devices.
    let mut dp = vec![vec![f32::INFINITY; m + 1]; n + 1];
    let mut choice = vec![vec![None; m + 1]; n + 1];

    // With zero devices, only zero layers can be scheduled.
    dp[0][0] = 0.0;
    for row in dp.iter_mut().take(n + 1).skip(1) {
        row[0] = f32::INFINITY;
    }

    // With any positive number of devices, zero layers have zero cost.
    for row in dp[0].iter_mut().take(m + 1).skip(1) {
        *row = 0.0;
    }

    // choice[i][j] = Some(k) means device j owns layers k+1..=i in the optimal
    // sub-problem. None means device j receives zero layers.

    for j in 1..=m {
        let dev = devices[j - 1];
        for i in 0..=n {
            let mut best = dp[i][j - 1];
            let mut best_k = None;

            for (k, row) in dp.iter().take(i).enumerate() {
                let prev = row[j - 1];
                if prev.is_infinite() {
                    continue;
                }

                let layers = (k as u32)..(i as u32);
                let c = cost_model.cost(layers, dev);
                if c.is_infinite() {
                    continue;
                }

                let candidate = prev.max(c);
                // Strict `<` means ties go to "assign zero layers", which keeps
                // the plan minimal and deterministic.
                if candidate < best {
                    best = candidate;
                    best_k = Some(k);
                }
            }

            dp[i][j] = best;
            choice[i][j] = best_k;
        }
    }

    if dp[n][m].is_infinite() {
        return Err(ScheduleError::Infeasible(layer_count));
    }

    let mut assignments = Vec::with_capacity(m);
    let mut i = n;
    let mut j = m;

    while i > 0 && j > 0 {
        match choice[i][j] {
            None => {
                // Device j-1 got zero layers.
                j -= 1;
            }
            Some(k) => {
                let dev = devices[j - 1];
                assignments.push(LayerRange {
                    node_id: dev.node_id,
                    start_layer: k as u32,
                    end_layer: i as u32,
                    next_hop: SocketAddr::from(([0, 0, 0, 0], 0)),
                });
                i = k;
                j -= 1;
            }
        }
    }

    if i != 0 {
        // Reconstruction failed even though dp said the problem was feasible.
        return Err(ScheduleError::Infeasible(layer_count));
    }

    // Reconstruct backwards; restore ring order.
    assignments.reverse();

    // Fill in next_hop from the address map.
    for idx in 0..assignments.len() {
        let next_idx = (idx + 1) % assignments.len();
        let next_id = assignments[next_idx].node_id;
        let addr = input
            .addresses()
            .get(&next_id)
            .ok_or(ScheduleError::MissingAddress(next_id))?;
        assignments[idx].next_hop = *addr;
    }

    Ok(PartitionPlan { assignments })
}

/// Returns `true` if the plan's layer ranges are contiguous, non-overlapping,
/// cover `[0, total_layers)`, and each assigned range respects the device's
/// available RAM.
///
/// This is primarily exposed for property- and integration-test assertions.
pub fn is_valid_plan(
    plan: &PartitionPlan,
    total_layers: u32,
    cost_model: &dyn CostModel,
    devices: &[NodeCapability],
) -> bool {
    if plan.assignments.is_empty() {
        return total_layers == 0;
    }

    let device_map: HashMap<Uuid, &NodeCapability> =
        devices.iter().map(|d| (d.node_id, d)).collect();

    let mut covered = 0;
    let mut expected = 0;
    for a in &plan.assignments {
        if a.start_layer != expected {
            return false;
        }
        if a.start_layer >= a.end_layer {
            return false;
        }
        covered += a.layer_count();

        if let Some(dev) = device_map.get(&a.node_id) {
            let required = cost_model.ram_required(a.start_layer..a.end_layer);
            if required > dev.free_ram_bytes {
                return false;
            }
        } else {
            return false;
        }

        expected = a.end_layer;
    }

    covered == total_layers
}

/// Convenience helper to build a [`NodeCapability`] in tests without specifying
/// every field.
///
/// # Example
///
/// ```
/// use ragtag_core::Stability;
/// use ragtag_scheduler::node;
/// use std::time::Instant;
/// use uuid::Uuid;
///
/// let n = node(Uuid::new_v4(), 10.0, 1024, Stability::Stable);
/// assert!(n.reachable);
/// ```
pub fn node(
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
