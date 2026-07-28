# SGL KV Indexer

`sgl-kv-indexer` is an experimental shared metadata index for SGLang KV-cache
blocks. It records which worker and storage tier currently holds each
content-addressed block, allowing a router to query likely cache hits without
moving KV data itself.

## Architecture

```text
SGLang worker ── ZMQ PUB + replay ROUTER ──> bridge ── gRPC ──> indexer ──> Redis
```

- SGLang publishes `BlockStored`, `BlockRemoved`, and `AllBlocksCleared` events.
- One `kv-indexer-bridge` follows each independent worker/rank event stream.
- `kv-indexer-server` applies ordered event batches and serves match queries.
- Redis, Dragonfly, or Redis Cluster stores durable placement metadata.

The bridge detects sequence gaps and requests missing batches from SGLang's
replay endpoint. The indexer uses per-worker sequence gating and incarnation
fencing so duplicate or delayed batches cannot overwrite newer state.

## Recommended deployment

For production, run one bridge as a sidecar alongside each independent SGLang
KV-event stream (for example, per DP rank):

```text
SGLang worker ──local ZMQ──> bridge ──gRPC──> indexer ──> Redis
```

- Co-locate the bridge and publisher in one Pod or container group, keep ZMQ
  traffic local, and give each stream a unique `KV_INDEXER_WORKER_ID`.
- Configure `SGLANG_KV_EVENT_REPLAY_ENDPOINT` in production. Size SGLang's
  replay `buffer_steps` to cover the longest expected bridge or indexer outage.
- Persist `KV_INDEXER_WORKER_INCARNATION_FILE` across bridge-only restarts; a
  shared Kubernetes `emptyDir` is normally sufficient.
- Assign each bridge to exactly one indexer server. Apply traffic for one worker
  must not be distributed across servers; match queries may be load-balanced.
- Keep inference readiness independent of this advisory index, and monitor
  bridge and indexer health separately.
- Prefer a single Redis or Dragonfly instance when sufficient; use Redis Cluster
  when horizontal throughput or shard-level failover is required.

## Best-effort fault tolerance

The index is advisory routing metadata, not the source of truth. Callers must
tolerate misses and stale candidates; workers remain authoritative.

- The bridge replays sequence gaps while SGLang still has the missing batches.
- An unrecoverable gap retires the worker incarnation, drops its placements, and
  resumes the live stream. This favors bounded recovery and temporary
  under-reporting over infinite catch-up or unverifiable state.
- Incarnation checkpointing never blocks startup. If persistence fails, the next
  restart performs a full resync. There is no periodic full-state reconciliation.

## Current status and limitations

This crate is experimental.

- Multiple indexer servers may share one Redis namespace when worker ownership
  is statically partitioned between them. All events for one worker must go to
  exactly one indexer server at a time because per-worker apply serialization is
  process-local.
- Deploy one bridge per independent SGLang KV-event stream (for example, per DP
  rank), each with a unique `KV_INDEXER_WORKER_ID`.
- Configure the replay endpoint in production, and size SGLang's `buffer_steps`
  to cover the longest expected bridge outage.
- Redis Cluster is supported, but an apply batch spans multiple hash slots and
  is not globally atomic. Per-worker sequence and generation fencing preserve
  replay convergence.

### Known gaps

These are accepted for an experimental crate and are listed so operators are not
surprised by them:

- After an incarnation is retired, the bridge resumes from the live event stream
  rather than replaying the new publisher's earlier history. Blocks that SGLang
  cached before the rotation and never re-reports stay absent from the index
  until they are evicted and cached again, so routing may temporarily miss a
  reusable prefix.
- Retired incarnation tokens accumulate as `retired:<token>` fields in each
  worker's meta hash and are never pruned. Growth is one small field per bridge
  restart, so it is slow, but a worker restarted continuously for a long time
  will keep a large meta hash. Deleting the worker's meta key clears it.
- Sequence numbers are compared inside Lua, which uses double-precision
  arithmetic, so values above 2^53 are not exact even though the proto field is
  `uint64`. A publisher would have to emit ~9e15 batches to reach that point.
- There is no periodic full-state reconciliation between SGLang and the index.
  Convergence relies on replay and on incarnation rotation; a placement lost to
  a gap is only restored when the block is reported again.
- SGLang's radix cache namespaces entries by an `extra_key` (LoRA id, cache
  salt) but does not fold it into the block digest, and always publishes
  `lora_id` as null. Blocks belonging to different adapters therefore share a
  hash, and the index cannot tell them apart. This is an upstream property that
  sgl-router's own hash tree shares; routing is advisory, so the cost of a
  wrong guess is a cache miss. `HashSpec.namespace` is reserved for
  partitioning once the engine publishes the key.
- Blocks SGLang reports under the `EXTERNAL` medium — a shared remote pool —
  are not indexed. Placement here is per `(block, worker)`, and a shared pool is
  equally reachable from every worker, so recording it under whichever worker
  wrote it would claim an affinity that does not exist; a router picking by
  prefix length would then prefer a long remote-pool prefix over a short warm
  one. Indexing a shared pool needs a cluster-level record instead. SGLang emits
  only `GPU` and `CPU_PINNED` today, so nothing is lost yet, and the bridge
  reports once per publisher when it does skip such events.

## Horizontal scaling

Indexer servers share no persistent local state, so they can scale horizontally
by statically assigning each worker's bridge to one server while all servers use
the same Redis namespace:

```text
bridge(worker-0) ──> indexer-0 ─┐
bridge(worker-1) ──> indexer-1 ─┼─> shared Redis namespace
bridge(worker-2) ──> indexer-0 ─┤
bridge(worker-3) ──> indexer-1 ─┘
```

Set each bridge's existing `KV_INDEXER_ENDPOINT` to its assigned server. Match
queries can be sent to any indexer server because every server reads the shared
Redis state.

The deployment must guarantee that two indexer servers do not concurrently
process events for the same `KV_INDEXER_WORKER_ID`. Automatic endpoint
selection, rebalancing, and failover are intentionally out of scope; reassign
and restart the affected bridge when changing ownership.

## Build

```bash
cd experimental/sgl-kv-indexer
cargo build --release --features redis-backend
```

This produces:

- `target/release/kv-indexer-server`
- `target/release/kv-indexer-bridge`

The `redis-backend` feature is required to run the Redis-backed server. The
`bridge` feature is on by default and builds the event follower; a routing
client turns it off (see the integration contract below).

## Run locally

Start Redis:

```bash
redis-server --port 6379
```

Start the indexer:

```bash
KV_INDEXER_BACKEND=redis \
KV_INDEXER_REDIS_URL=redis://127.0.0.1:6379 \
KV_INDEXER_LISTEN_ADDR=127.0.0.1:50051 \
RUST_LOG=info \
  cargo run --release --features redis-backend --bin kv-indexer-server
```

Start one bridge for a SGLang worker:

```bash
KV_INDEXER_WORKER_ID=worker-0 \
KV_INDEXER_WORKER_ADDRESS=http://127.0.0.1:30000 \
KV_INDEXER_ENDPOINT=http://127.0.0.1:50051 \
SGLANG_KV_EVENT_ENDPOINT=tcp://127.0.0.1:5567 \
SGLANG_KV_EVENT_REPLAY_ENDPOINT=tcp://127.0.0.1:5590 \
RUST_LOG=info \
  cargo run --release --bin kv-indexer-bridge
```

The corresponding SGLang server must publish KV events at those endpoints, for
example with a `kv-events-config` containing a ZMQ publisher endpoint and replay
endpoint.

## Configuration

### Indexer server

| Variable | Default | Description |
| --- | --- | --- |
| `KV_INDEXER_LISTEN_ADDR` | `[::1]:50051` | gRPC listen address |
| `KV_INDEXER_BACKEND` | required | `redis` for the production backend |
| `KV_INDEXER_REDIS_URL` | none | Single Redis/Dragonfly URL |
| `KV_INDEXER_REDIS_CLUSTER_NODES` | none | Comma-separated Redis Cluster seed URLs; takes precedence over the single URL |
| `KV_INDEXER_REDIS_NAMESPACE` | `kvidx` | Redis key prefix |
| `KV_INDEXER_REDIS_REQUIRED` | `1` | Fail startup if Redis is unavailable; `0` enables degraded lazy connection |
| `KV_INDEXER_WORKER_TTL_SECS` | `120` | Worker liveness TTL; `0` disables expiry |

### Bridge

| Variable | Default | Description |
| --- | --- | --- |
| `KV_INDEXER_WORKER_ID` | required | Unique ID for this worker event stream |
| `KV_INDEXER_WORKER_ADDRESS` | empty | The worker's routing identity, byte-identical to the URL a router registers. Empty means the worker is indexed but never returned by a routing query |
| `KV_INDEXER_ENDPOINT` | `http://[::1]:50051` | Indexer gRPC endpoint |
| `SGLANG_KV_EVENT_ENDPOINT` | `tcp://127.0.0.1:5557` | SGLang event PUB endpoint |
| `SGLANG_KV_EVENT_REPLAY_ENDPOINT` | none | SGLang replay ROUTER endpoint |
| `SGLANG_KV_EVENT_TOPIC` | empty | ZMQ subscription topic |
| `KV_INDEXER_CLEAR_TIERS` | `HBM,DRAM,SSD` | Tiers affected by `AllBlocksCleared` |
| `KV_INDEXER_HEARTBEAT_SECS` | `30` | Worker heartbeat interval; `0` disables it |
| `KV_INDEXER_WORKER_INCARNATION` | generated | Optional observable prefix for generated incarnation tokens |
| `KV_INDEXER_WORKER_INCARNATION_FILE` | `/tmp/sgl-kv-indexer-<worker-id-hex>.incarnation` | Local checkpoint that preserves the publisher incarnation across bridge-only restarts; place it on a sidecar-persistent volume when Bridge and SGLang have different container lifecycles. Best effort: an unwritable location only costs a full resync on restart, it never blocks startup |

## API

The protobuf service in `proto/kv_indexer.proto` provides:

- `ApplyExternalKvBatch`: ordered placement reports, revocations, and clears.
- `MatchExternalKv`: workers and tiers holding requested block hashes.
- `MatchExternalKvPrefix`: how much of an ordered request each worker can serve
  from cache. This is the routing query; see below.
- `GetExternalKvHitCounts`: per-block hit counters.

The server also exposes the standard gRPC health service. Redis readiness is
checked periodically and reflected as `SERVING` or `NOT_SERVING`.

## Router integration contract

A KV-aware router asks one question: which worker already holds this request's
prefix. `MatchExternalKvPrefix` answers it, and this section is the contract it
answers under. Nothing here is wired into a router yet; the crate is built so
that it can be.

### What the query returns

Per worker: its routing address, DP rank, and `matched_prefix_blocks` — the
number of leading blocks of the query it holds contiguously — ordered longest
first. Plus `best_prefix_blocks`, `queried_blocks`, and
`spec_mismatched_workers` for the whole response.

The indexer does not pick a worker. It cannot: it has no view of health, of
circuit breakers, or of in-flight load, and those decide the final choice. It
reports cache evidence and the caller combines it with everything else.

`matched_prefix_blocks` stops at the first block a worker is missing. That is
stricter than "holds the deepest matched block" — the rule sgl-router's local
hash tree uses — so the indexer can under-report a hit but never points a
request at a worker that cannot serve it. `tests/router_contract.rs` pins both
halves of that: equivalence on gap-free placement, and the direction of the
divergence otherwise.

### Worker identity

`worker_address` is the worker's **routing identity**, not its KV-transfer
address. A router intersects it with its own registered worker URLs, so the two
must agree byte for byte. Set `KV_INDEXER_WORKER_ADDRESS` to exactly the URL the
router registers; a worker with no address is omitted from prefix responses,
because there is nothing to route to.

Several DP ranks share one address. `dp_rank` distinguishes their event streams;
the bridge pins it to the first `attn_dp_rank` it observes and warns once if the
stream later carries another, which is how a bridge subscribed to the wrong port
announces itself.

### Hash space

Block hashes are content-addressed, so a caller that hashes with a different
block size or a different page shape produces hashes that match nothing. The
symptom is a cache hit rate of zero and no error anywhere.

`HashSpec` makes that visible. The bridge derives it from the event stream —
page size from `BlockStored.block_size`, unigram vs. bigram from the shape of
`token_ids` — rather than from configuration, so it cannot drift from what the
worker actually publishes, and forgets it when the publisher's incarnation
rotates, since a restarted worker may have restarted with a different page size.
A query carrying a spec drops workers whose recorded spec disagrees and reports
them in `spec_mismatched_workers`; an empty result with a nonzero count there is
the signal that the two sides disagree.

Checking is skipped whenever either side is silent, so a router and its
publishers can be upgraded in either order.

Hashes are decimal strings of SGLang's signed 64-bit block hash. The server
normalizes integer-valued hashes to that form on ingest, so the unsigned
encoding of the same 64 bits lands on the same entry. Use
`client::hash_to_wire`.

### Failure handling

The index is advisory, so a caller must never fail a request because the index
did. `client::ExternalPrefixIndex::match_prefix` returns `PrefixIndexOutcome`,
which has no error variant — every failure is a `NoSignal` the caller falls back
from. `KvIndexerPrefixIndex` connects lazily, bounds each query (10 ms by
default), and trips a breaker after repeated outages so an unreachable index
costs an atomic load rather than a connection attempt. Only index-level trouble
opens the breaker; a rejected query is about that query, and one malformed
request must not cost every other request its routing signal.

Connecting takes longer than a query budget, so a router's first few queries
fail and open the breaker before the channel is up. The connection continues
establishing in the background, so this costs one cooldown of min-load routing
at startup rather than a permanent outage.

`count_as_hit` is off by default. Counting is telemetry: it credits only the
head of the reusable prefix, and a failure to write it is logged rather than
returned, because discarding a computed answer over a counter would look to the
caller like an unreachable index.

The server also bounds its own work. A prefix scan stops after 2048 blocks, so a
longer match is reported as 2048 rather than its true length; `queried_blocks`
shows where the scan stopped. A routing decision does not get better by resolving
a longer prefix exactly, and at SGLang's page sizes this covers a context larger
than any real request.

### Depending on the crate from a router

```toml
sgl-kv-indexer = { path = "../sgl-kv-indexer", default-features = false }
```

This drops the ZMQ event follower, msgpack, and the Redis backend, leaving the
generated types and the gRPC client.

### Cost

The scan reads the first block, then the registry once for the workers holding
it, then walks forward in growing windows until no candidate can extend. The
number of block hashes read tracks the prefix that exists rather than the length
of the request: a cold first block costs one read no matter how long the request
is. `queried_blocks` reports the real figure.

## Tests

Single Redis:

```bash
KV_INDEXER_REDIS_URL=redis://127.0.0.1:6379 \
  cargo test --features redis-backend -- --test-threads=1
```

Static checks:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --features redis-backend -- -D warnings
```

The repository CI additionally starts a six-node Redis Cluster and exercises
normal Cluster routing, ASK redirects, and master failover.
