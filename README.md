# ragtag

A generic, capability-driven, heterogeneous LAN cluster for distributed
transformer inference. Scheduling is based on reported throughput, free RAM,
stability, and reachability — not device type.

- mDNS and gRPC peer discovery
- Dynamic layer partitioning across heterogeneous workers
- Ring-pipeline fallback
- llama.cpp RPC orchestration via `llama-server` + `llama-rpc-server`
- OpenAI-compatible `POST /v1/chat/completions` HTTP API

## Quick start

```bash
nix develop .#
cargo test --workspace
```

## Build a container image

```bash
nix build .#containerImage
podman load -i result
```

Then run the coordinator + two RPC worker nodes:

```bash
podman compose up
# or
docker compose up
```

The compose file expects the TinyLlama GGUF at
`models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf`.
