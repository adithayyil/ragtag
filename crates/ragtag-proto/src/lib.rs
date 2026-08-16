//! Generated gRPC types and conversions for `ragtag`.
#![warn(missing_docs)]

use ragtag_core::{LayerRange, NodeCapability, PartitionPlan, Stability};
use std::net::SocketAddr;
use std::time::Instant;
use thiserror::Error;
use uuid::Uuid;

/// Errors that can occur while converting between protobuf and core types.
#[derive(Debug, Error)]
pub enum ProtoError {
    /// The wire format did not contain a valid UUID.
    #[error("invalid uuid bytes")]
    Uuid,
    /// The wire format contained an unknown `Stability` value.
    #[error("unknown stability value")]
    Stability,
    /// The wire format contained a socket address that could not be parsed.
    #[error("invalid socket address: {0}")]
    SocketAddr(String),
}

/// The generated protobuf module.
#[allow(missing_docs)]
pub mod pb {
    tonic::include_proto!("ragtag");
}

// Re-export the generated clients and servers to keep imports shallow.
pub use pb::coordinator_client::CoordinatorClient;
pub use pb::coordinator_server::{Coordinator, CoordinatorServer};
pub use pb::ring_transport_client::RingTransportClient;
pub use pb::ring_transport_server::{RingTransport, RingTransportServer};

impl TryFrom<pb::NodeCapability> for NodeCapability {
    type Error = ProtoError;

    fn try_from(v: pb::NodeCapability) -> Result<Self, Self::Error> {
        let stability = match v.stability {
            0 => Stability::Stable,
            1 => Stability::Opportunistic,
            _ => return Err(ProtoError::Stability),
        };

        Ok(NodeCapability {
            node_id: Uuid::from_slice(&v.node_id).map_err(|_| ProtoError::Uuid)?,
            tokens_per_sec: v.tokens_per_sec,
            free_ram_bytes: v.free_ram_bytes,
            reachable: v.reachable,
            stability,
            // The receiver always refreshes its own local heartbeat timer.
            last_heartbeat: Instant::now(),
        })
    }
}

impl From<NodeCapability> for pb::NodeCapability {
    fn from(v: NodeCapability) -> Self {
        Self {
            node_id: v.node_id.as_bytes().to_vec(),
            tokens_per_sec: v.tokens_per_sec,
            free_ram_bytes: v.free_ram_bytes,
            reachable: v.reachable,
            stability: v.stability as i32,
            last_heartbeat_ms: 0,
        }
    }
}

impl From<Stability> for pb::Stability {
    fn from(v: Stability) -> Self {
        match v {
            Stability::Stable => pb::Stability::Stable,
            Stability::Opportunistic => pb::Stability::Opportunistic,
        }
    }
}

impl TryFrom<pb::LayerRange> for LayerRange {
    type Error = ProtoError;

    fn try_from(v: pb::LayerRange) -> Result<Self, Self::Error> {
        Ok(LayerRange {
            node_id: Uuid::from_slice(&v.node_id).map_err(|_| ProtoError::Uuid)?,
            start_layer: v.start_layer,
            end_layer: v.end_layer,
            next_hop: v
                .next_hop
                .parse::<SocketAddr>()
                .map_err(|e| ProtoError::SocketAddr(e.to_string()))?,
        })
    }
}

impl From<LayerRange> for pb::LayerRange {
    fn from(v: LayerRange) -> Self {
        Self {
            node_id: v.node_id.as_bytes().to_vec(),
            start_layer: v.start_layer,
            end_layer: v.end_layer,
            next_hop: v.next_hop.to_string(),
        }
    }
}

impl TryFrom<pb::PartitionPlan> for PartitionPlan {
    type Error = ProtoError;

    fn try_from(v: pb::PartitionPlan) -> Result<Self, Self::Error> {
        let mut assignments = Vec::with_capacity(v.assignments.len());
        for a in v.assignments {
            assignments.push(LayerRange::try_from(a)?);
        }
        Ok(PartitionPlan { assignments })
    }
}

impl From<PartitionPlan> for pb::PartitionPlan {
    fn from(v: PartitionPlan) -> Self {
        Self {
            assignments: v
                .assignments
                .into_iter()
                .map(pb::LayerRange::from)
                .collect(),
        }
    }
}

impl TryFrom<pb::NodeId> for Uuid {
    type Error = ProtoError;

    fn try_from(v: pb::NodeId) -> Result<Self, Self::Error> {
        Uuid::from_slice(&v.node_id).map_err(|_| ProtoError::Uuid)
    }
}

impl From<Uuid> for pb::NodeId {
    fn from(v: Uuid) -> Self {
        Self {
            node_id: v.as_bytes().to_vec(),
        }
    }
}
