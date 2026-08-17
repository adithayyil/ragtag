//! Core, I/O-free domain types for the `ragtag` cluster.
//!
//! Nothing in this crate performs network calls, file I/O, or `unsafe` blocks.
//! It exists so that `ragtag-scheduler`, `ragtag-proto`, and the two binaries
//! can share a single source of truth for what a node capability and a layer
//! partition look like.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::Instant;
use uuid::Uuid;

/// Network discovery helpers (mDNS, local address selection).
pub mod net;

/// Describes what a node can currently offer to the pipeline.
///
/// `tokens_per_sec` is a node-local throughput estimate. The concrete unit is
/// left to the chosen [`CostModel`](ragtag_scheduler::CostModel), but it is
/// treated as a per-layer processing rate by the default cost model.
///
/// `stability` is **self-reported** by the node. A node that is `Opportunistic`
/// may be excluded by the scheduler first when resources are tight.
///
/// # Example
///
/// ```
/// use ragtag_core::{NodeCapability, Stability};
/// use std::time::Instant;
/// use uuid::Uuid;
///
/// let node = NodeCapability {
///     node_id: Uuid::new_v4(),
///     tokens_per_sec: 12.5,
///     free_ram_bytes: 8 * 1024 * 1024 * 1024,
///     reachable: true,
///     stability: Stability::Stable,
///     last_heartbeat: Instant::now(),
/// };
///
/// assert!(node.reachable);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeCapability {
    /// Unique node identifier.
    pub node_id: Uuid,
    /// Estimated layer-processing throughput.
    pub tokens_per_sec: f32,
    /// Available RAM in bytes, as reported by the node.
    pub free_ram_bytes: u64,
    /// `false` means the coordinator has timed out this node and will not
    /// schedule work onto it until heartbeats resume.
    pub reachable: bool,
    /// Reliability hint reported by the node itself.
    pub stability: Stability,
    /// Wall-clock independent heartbeat timestamp. Serialized via
    /// `serde_millis` as a millisecond offset relative to the local process
    /// clock; it is intended for local timeout checks, not cross-machine
    /// comparison.
    #[serde(with = "serde_millis")]
    pub last_heartbeat: Instant,
}

/// Whether a node's capacity should be treated as guaranteed or best-effort.
///
/// `Opportunistic` devices are not excluded by the scheduler by default, but
/// they may be assigned zero layers when doing so improves the bottleneck.
///
/// # Example
///
/// ```
/// use ragtag_core::Stability;
///
/// let stable = Stability::Stable;
/// let opportunistic = Stability::Opportunistic;
///
/// assert_ne!(stable, opportunistic);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Stability {
    /// The node expects to remain in the cluster.
    Stable,
    /// The node may disappear; the scheduler should exclude it under
    /// contention.
    Opportunistic,
}

/// One contiguous assignment of transformer layers to a node.
///
/// `start_layer` is inclusive and `end_layer` is exclusive, so the range
/// represents layers `start_layer..end_layer`.
///
/// # Example
///
/// ```
/// use ragtag_core::LayerRange;
/// use std::net::SocketAddr;
/// use uuid::Uuid;
///
/// let assignment = LayerRange {
///     node_id: Uuid::new_v4(),
///     start_layer: 0,
///     end_layer: 8,
///     next_hop: SocketAddr::from(([127, 0, 0, 1], 8080)),
/// };
///
/// assert_eq!(assignment.end_layer - assignment.start_layer, 8);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerRange {
    /// Which node owns this slice.
    pub node_id: Uuid,
    /// Inclusive start layer index.
    pub start_layer: u32,
    /// Exclusive end layer index.
    pub end_layer: u32,
    /// Where to send this node's output activations in the ring.
    pub next_hop: SocketAddr,
}

impl LayerRange {
    /// Number of layers in this range.
    pub fn layer_count(&self) -> u32 {
        self.end_layer - self.start_layer
    }
}

/// A complete partition of the model across a ring of nodes.
///
/// # Example
///
/// ```
/// use ragtag_core::{LayerRange, PartitionPlan};
/// use std::net::SocketAddr;
/// use uuid::Uuid;
///
/// let plan = PartitionPlan {
///     assignments: vec![
///         LayerRange {
///             node_id: Uuid::new_v4(),
///             start_layer: 0,
///             end_layer: 4,
///             next_hop: SocketAddr::from(([127, 0, 0, 1], 8081)),
///         },
///         LayerRange {
///             node_id: Uuid::new_v4(),
///             start_layer: 4,
///             end_layer: 8,
///             next_hop: SocketAddr::from(([127, 0, 0, 1], 8080)),
///         },
///     ],
/// };
///
/// assert_eq!(plan.assignments.iter().map(|a| a.layer_count()).sum::<u32>(), 8);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartitionPlan {
    /// Ordered ring assignments.
    pub assignments: Vec<LayerRange>,
}

impl PartitionPlan {
    /// Total number of transformer layers covered by this plan.
    pub fn total_layers(&self) -> u32 {
        self.assignments.iter().map(|a| a.layer_count()).sum()
    }

    /// Returns the entry node for the ring, i.e. the assignment that starts at
    /// layer 0.
    pub fn entry_node(&self) -> Option<&LayerRange> {
        self.assignments.iter().find(|a| a.start_layer == 0)
    }
}
