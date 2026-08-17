use anyhow::{Context, Result};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

/// Default mDNS service type for worker nodes.
pub const SERVICE_TYPE: &str = "_ragtag._tcp.local.";
/// Default mDNS service type for coordinator discovery.
pub const COORD_SERVICE_TYPE: &str = "_ragtag-coord._tcp.local.";

/// A coordinator discovered through mDNS.
#[derive(Clone, Debug)]
pub struct DiscoveredCoordinator {
    /// gRPC endpoint of the coordinator.
    pub grpc: SocketAddr,
    /// HTTP endpoint of the coordinator, if advertised.
    pub http: Option<SocketAddr>,
}

/// Service type for node advertisements, overridable via `RAGTAG_SERVICE_TYPE`.
pub fn service_type() -> String {
    std::env::var("RAGTAG_SERVICE_TYPE").unwrap_or_else(|_| SERVICE_TYPE.into())
}

/// Service type for coordinator advertisements, overridable via
/// `RAGTAG_COORD_SERVICE_TYPE`.
pub fn coord_service_type() -> String {
    std::env::var("RAGTAG_COORD_SERVICE_TYPE").unwrap_or_else(|_| COORD_SERVICE_TYPE.into())
}

/// Create an mDNS daemon. Reuse this daemon for both browsing and registering
/// services in the same process; the daemon is kept alive by clones.
pub fn create_mdns_daemon() -> Result<mdns_sd::ServiceDaemon> {
    mdns_sd::ServiceDaemon::new().context("create mdns daemon")
}

fn interfaces() -> Result<Vec<if_addrs::Interface>> {
    if_addrs::get_if_addrs().context("list network interfaces")
}

fn is_usable(iface: &if_addrs::Interface) -> bool {
    if !iface.is_oper_up() || iface.is_loopback() || iface.is_p2p() {
        return false;
    }
    if iface.addr.is_link_local() {
        return false;
    }
    if let if_addrs::IfAddr::V4(v4) = &iface.addr {
        if v4.prefixlen == 0 || v4.prefixlen >= 31 {
            return false;
        }
        let name = iface.name.to_lowercase();
        for prefix in [
            "veth", "docker", "podman", "br-", "virbr", "vmnet", "tun", "tap", "wg", "ppp",
        ] {
            if name.starts_with(prefix) {
                return false;
            }
        }
        true
    } else {
        false
    }
}

/// Pick the first usable local IPv4 address. This is a coarse fallback; prefer
/// `local_ipv4_for` when you know the remote address.
pub fn first_local_ipv4() -> Option<Ipv4Addr> {
    interfaces().ok()?.into_iter().find_map(|iface| {
        if !is_usable(&iface) {
            return None;
        }
        if let if_addrs::IfAddr::V4(v4) = iface.addr {
            Some(v4.ip)
        } else {
            None
        }
    })
}

/// Find the local IPv4 address on the interface whose subnet contains `remote`.
/// This uses the actual netmask, not a hardcoded /24, so it handles odd
/// subnets correctly.
pub fn local_ipv4_for(remote: Ipv4Addr) -> Option<Ipv4Addr> {
    let mut best: Option<Ipv4Addr> = None;
    for iface in interfaces().ok()? {
        if !is_usable(&iface) {
            continue;
        }
        if let if_addrs::IfAddr::V4(v4) = &iface.addr {
            let mask = if v4.prefixlen == 0 {
                0
            } else {
                u32::MAX << (32 - v4.prefixlen)
            };
            let local_net = u32::from(v4.ip) & mask;
            let remote_net = u32::from(remote) & mask;
            if local_net == remote_net && best.is_none() {
                best = Some(v4.ip);
            }
        }
    }
    best
}

/// Returns true if `addr` is on a local, usable subnet.
pub fn is_local_ipv4(addr: Ipv4Addr) -> bool {
    local_ipv4_for(addr).is_some()
}

/// Resolve a configured advertise host string into an IPv4 address.
///
/// - `""`, `"auto"`, or `"0.0.0.0"`: use the local IP that can reach
///   `coord_ip` when one is known, otherwise fall back to `127.0.0.1`.
/// - `"127.0.0.1"`: keep loopback when no local coordinator was found, but if
///   a local coordinator was found prefer the LAN IP so other nodes can reach
///   us directly.
/// - explicit IPv4 address: use that.
pub fn resolve_advertise_ip(host: &str, coord_ip: Option<Ipv4Addr>) -> Option<Ipv4Addr> {
    if host.is_empty() || host.eq_ignore_ascii_case("auto") || host == "0.0.0.0" {
        if let Some(coord) = coord_ip {
            local_ipv4_for(coord)
        } else {
            Some(Ipv4Addr::new(127, 0, 0, 1))
        }
    } else if host == "127.0.0.1" {
        if let Some(coord) = coord_ip {
            local_ipv4_for(coord).or(Some(Ipv4Addr::new(127, 0, 0, 1)))
        } else {
            Some(Ipv4Addr::new(127, 0, 0, 1))
        }
    } else {
        host.parse().ok()
    }
}

/// Discover a ragtag coordinator over mDNS.
///
/// `timeout` caps how long we wait. We prefer a coordinator whose address is on
/// a local subnet (so LAN mode works) and, for determinism, sort by local flag
/// then by stringified address. Returns `None` if no service is seen in time.
pub fn discover_coordinator(
    mdns: &mdns_sd::ServiceDaemon,
    timeout: Duration,
) -> Result<Option<DiscoveredCoordinator>> {
    let receiver = mdns
        .browse(&coord_service_type())
        .context("browse coordinator")?;
    let deadline = std::time::Instant::now() + timeout;
    let mut candidates: Vec<DiscoveredCoordinator> = Vec::new();

    while std::time::Instant::now() < deadline {
        let remaining = deadline - std::time::Instant::now();
        let wait = remaining.min(Duration::from_millis(200));
        match receiver.recv_timeout(wait) {
            Ok(mdns_sd::ServiceEvent::ServiceResolved(info)) => {
                let info = *info;
                if !info.is_valid() {
                    continue;
                }
                // The coordinator advertises its chosen grpc/http endpoints in TXT
                // records; prefer those over A-record addresses which can include
                // bridge/loopback interfaces.
                let grpc = info
                    .get_property_val_str("grpc")
                    .and_then(|s| s.parse::<SocketAddr>().ok())
                    .or_else(|| {
                        let mut v4s: Vec<_> = info
                            .get_addresses_v4()
                            .into_iter()
                            .filter(|ip| is_local_ipv4(*ip))
                            .collect();
                        if v4s.is_empty() {
                            v4s = info
                                .get_addresses_v4()
                                .into_iter()
                                .filter(|ip| !ip.is_loopback() && !ip.is_link_local())
                                .collect();
                        }
                        v4s.sort();
                        v4s.into_iter()
                            .next()
                            .map(|ip| SocketAddr::new(IpAddr::V4(ip), info.get_port()))
                    });
                let Some(grpc) = grpc else { continue };
                let http = info
                    .get_property_val_str("http")
                    .and_then(|s| s.parse::<SocketAddr>().ok());
                candidates.push(DiscoveredCoordinator { grpc, http });
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }

    if candidates.is_empty() {
        return Ok(None);
    }

    candidates.sort_by(|a, b| {
        let a_local = matches!(a.grpc.ip(), IpAddr::V4(v4) if is_local_ipv4(v4));
        let b_local = matches!(b.grpc.ip(), IpAddr::V4(v4) if is_local_ipv4(v4));
        a_local
            .cmp(&b_local)
            .then_with(|| a.grpc.to_string().cmp(&b.grpc.to_string()))
    });

    Ok(Some(candidates[0].clone()))
}
