use proptest::prelude::*;
use ragtag_core::Stability;
use ragtag_scheduler::{is_valid_plan, node, solve, LinearCostModel, ScheduledInput};
use std::collections::HashMap;
use std::net::SocketAddr;
use uuid::Uuid;

fn make_addrs(nodes: &[ragtag_core::NodeCapability]) -> HashMap<Uuid, SocketAddr> {
    nodes
        .iter()
        .enumerate()
        .map(|(i, n)| {
            (
                n.node_id,
                SocketAddr::from(([127, 0, 0, 1], 8000 + i as u16)),
            )
        })
        .collect()
}

#[test]
fn hand_computed_three_node_equal_split() {
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();

    let devices = vec![
        node(a, 2.0, 1024, Stability::Stable),
        node(b, 2.0, 1024, Stability::Stable),
        node(c, 2.0, 1024, Stability::Stable),
    ];

    let addrs = make_addrs(&devices);
    let input = ScheduledInput::new(Uuid::nil(), &devices, &addrs).unwrap();
    let plan = solve(6, &input, &LinearCostModel::new(1, 1.0)).unwrap();

    assert_eq!(plan.assignments.len(), 3);
    assert_eq!(plan.total_layers(), 6);
    assert_eq!(plan.assignments[0].start_layer, 0);
    assert_eq!(plan.assignments[0].end_layer, 2);
    assert_eq!(plan.assignments[1].start_layer, 2);
    assert_eq!(plan.assignments[1].end_layer, 4);
    assert_eq!(plan.assignments[2].start_layer, 4);
    assert_eq!(plan.assignments[2].end_layer, 6);
}

#[test]
fn slow_opportunistic_node_is_excluded() {
    let fast = Uuid::new_v4();
    let medium = Uuid::new_v4();
    let slow = Uuid::new_v4();

    let devices = vec![
        node(fast, 10.0, 1024, Stability::Stable),
        node(medium, 5.0, 1024, Stability::Stable),
        node(slow, 1.0, 1024, Stability::Opportunistic),
    ];

    let addrs = make_addrs(&devices);
    let input = ScheduledInput::new(Uuid::nil(), &devices, &addrs).unwrap();
    let plan = solve(6, &input, &LinearCostModel::new(1, 1.0)).unwrap();

    assert!(!plan.assignments.iter().any(|r| r.node_id == slow));
    assert_eq!(plan.assignments.len(), 2);
    assert_eq!(plan.total_layers(), 6);
    assert_eq!(plan.assignments[0].node_id, fast);
    assert_eq!(plan.assignments[0].start_layer, 0);
    assert_eq!(plan.assignments[0].end_layer, 4);
    assert_eq!(plan.assignments[1].node_id, medium);
    assert_eq!(plan.assignments[1].start_layer, 4);
    assert_eq!(plan.assignments[1].end_layer, 6);
}

#[test]
fn opportunistic_node_excluded_under_tight_ram() {
    let fast = Uuid::new_v4();
    let medium = Uuid::new_v4();
    let slow = Uuid::new_v4();

    // Combined RAM of fast+medium exactly equals the 6 layers, while slow
    // has enough memory but is far slower. The scheduler should still prefer
    // fast+medium.
    let devices = vec![
        node(fast, 10.0, 3, Stability::Stable),
        node(medium, 5.0, 3, Stability::Stable),
        node(slow, 1.0, 6, Stability::Opportunistic),
    ];

    let addrs = make_addrs(&devices);
    let input = ScheduledInput::new(Uuid::nil(), &devices, &addrs).unwrap();
    let plan = solve(6, &input, &LinearCostModel::new(1, 1.0)).unwrap();

    assert!(!plan.assignments.iter().any(|r| r.node_id == slow));
    assert_eq!(plan.assignments.len(), 2);
    assert_eq!(plan.total_layers(), 6);
    assert_eq!(plan.assignments[0].node_id, fast);
    assert_eq!(
        plan.assignments[0].end_layer - plan.assignments[0].start_layer,
        3
    );
    assert_eq!(plan.assignments[1].node_id, medium);
    assert_eq!(
        plan.assignments[1].end_layer - plan.assignments[1].start_layer,
        3
    );
}

#[test]
fn coordinator_in_node_table_is_rejected() {
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();

    let devices = vec![
        node(a, 10.0, 1024, Stability::Stable),
        node(b, 5.0, 1024, Stability::Stable),
    ];

    let mut addrs = HashMap::new();
    addrs.insert(a, SocketAddr::from(([127, 0, 0, 1], 8001)));
    addrs.insert(b, SocketAddr::from(([127, 0, 0, 1], 8002)));

    let err = ScheduledInput::new(a, &devices, &addrs).unwrap_err();
    assert!(err.to_string().contains("coordinator"));
}

fn arb_node() -> impl Strategy<Value = ragtag_core::NodeCapability> {
    (
        any::<u128>(),
        0.1f32..1000.0f32,
        100_000_000u64..10_000_000_000u64,
        any::<bool>(),
    )
        .prop_map(|(id, tps, ram, stable)| {
            node(
                Uuid::from_u128(id),
                tps,
                ram,
                if stable {
                    Stability::Stable
                } else {
                    Stability::Opportunistic
                },
            )
        })
}

proptest! {
    #[test]
    fn ram_constraint_never_violated(
        layer_count in 1u32..64,
        bytes_per_layer in 1u64..1_000_000,
        seconds_per_layer in 0.01f32..10.0f32,
        devices in prop::collection::vec(arb_node(), 1..8),
    ) {
        let cost_model = LinearCostModel::new(bytes_per_layer, seconds_per_layer);

        let mut addresses = HashMap::new();
        for (i, d) in devices.iter().enumerate() {
            addresses.insert(d.node_id, SocketAddr::from(([127, 0, 0, 1], 8000 + i as u16)));
        }

        let input = ScheduledInput::new(Uuid::nil(), &devices, &addresses).unwrap();
        let plan = solve(layer_count, &input, &cost_model)
            .expect("all generated devices can hold the full model, so a plan should exist");

        prop_assert_eq!(plan.total_layers(), layer_count);
        prop_assert!(is_valid_plan(&plan, layer_count, &cost_model, &devices));
    }

    #[test]
    fn empty_layer_count_yields_empty_plan(
        bytes_per_layer in 1u64..1_000_000,
        seconds_per_layer in 0.01f32..10.0f32,
        devices in prop::collection::vec(arb_node(), 1..4),
    ) {
        let cost_model = LinearCostModel::new(bytes_per_layer, seconds_per_layer);
        let addrs = make_addrs(&devices);
        let input = ScheduledInput::new(Uuid::nil(), &devices, &addrs).unwrap();
        let plan = solve(0, &input, &cost_model).unwrap();
        prop_assert!(plan.assignments.is_empty());
    }
}
