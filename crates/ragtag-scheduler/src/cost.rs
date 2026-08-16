//! Cost models used by the DP scheduler.

use ragtag_core::NodeCapability;
use std::ops::Range;

/// A pluggable model that predicts the compute cost and memory footprint of a
/// layer range on a given device.
pub trait CostModel: std::fmt::Debug {
    /// Predicted execution time for `layers` on `device`.
    ///
    /// Returning `f32::INFINITY` marks the assignment as infeasible, e.g.
    /// because the layers do not fit in `device.free_ram_bytes`.
    fn cost(&self, layers: Range<u32>, device: &NodeCapability) -> f32;

    /// RAM in bytes required to store the weights for `layers`.
    fn ram_required(&self, layers: Range<u32>) -> u64;
}

/// A simple linear model: each layer requires a fixed amount of memory and a
/// fixed amount of compute, scaled by the node's `tokens_per_sec`.
///
/// `tokens_per_sec` is treated as a per-layer processing rate: a device with
/// `tokens_per_sec = 10.0` can execute 10 layers in one time unit when
/// `seconds_per_layer = 1.0`.
///
/// # Example
///
/// ```
/// use ragtag_core::NodeCapability;
/// use ragtag_scheduler::{CostModel, LinearCostModel};
/// use std::time::Instant;
/// use uuid::Uuid;
///
/// let model = LinearCostModel::new(512 * 1024 * 1024, 1.0);
/// let device = NodeCapability {
///     node_id: Uuid::new_v4(),
///     tokens_per_sec: 10.0,
///     free_ram_bytes: 8 * 1024 * 1024 * 1024,
///     reachable: true,
///     stability: ragtag_core::Stability::Stable,
///     last_heartbeat: Instant::now(),
/// };
///
/// // 4 layers on a fast device.
/// let cost = model.cost(0..4, &device);
/// assert!(cost.is_finite());
///
/// // Too many layers to fit in RAM.
/// let infeasible = model.cost(0..100, &device);
/// assert!(infeasible.is_infinite());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearCostModel {
    bytes_per_layer: u64,
    seconds_per_layer: f32,
}

impl LinearCostModel {
    /// Create a new linear cost model.
    pub const fn new(bytes_per_layer: u64, seconds_per_layer: f32) -> Self {
        Self {
            bytes_per_layer,
            seconds_per_layer,
        }
    }

    /// Bytes of memory required by one transformer layer.
    pub const fn bytes_per_layer(&self) -> u64 {
        self.bytes_per_layer
    }

    /// Basetime time per layer on a node with `tokens_per_sec == 1.0`.
    pub const fn seconds_per_layer(&self) -> f32 {
        self.seconds_per_layer
    }
}

impl CostModel for LinearCostModel {
    fn cost(&self, layers: Range<u32>, device: &NodeCapability) -> f32 {
        if layers.is_empty() {
            return 0.0;
        }

        let ram = self.ram_required(layers.clone());
        if ram > device.free_ram_bytes || device.tokens_per_sec <= 0.0 {
            return f32::INFINITY;
        }

        layers.len() as f32 * self.seconds_per_layer / device.tokens_per_sec
    }

    fn ram_required(&self, layers: Range<u32>) -> u64 {
        layers.len() as u64 * self.bytes_per_layer
    }
}
