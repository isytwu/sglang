<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
SPDX-License-Identifier: Apache-2.0
-->

# Component-aware prefix matching

## Why

SGLang's unified radix cache can report, per `BlockStored`, which KV *components*
(`full` / `swa` / `mamba`) are resident for a block at a medium. Reporting is
gated by `--enable-kv-events-component-types` (default **off**). Without
component awareness the indexer counts a block as reusable as soon as its hash is
present anywhere, which can over-report what a worker can actually serve for
hybrid models (sliding-window attention, Mamba). This feature makes the indexer's
`MatchExternalKvPrefix` a **safe lower bound** of the reusable prefix: it may
under-report, but — whenever the index state is accurate — it never over-reports.

## Design in one screen

The indexer treats component names as **opaque labels**. A worker declares how
each of its components participates in matching via a versioned
`WorkerCacheSpec`, carried on every `ApplyExternalKvBatch` (like
`worker_address`, so it self-heals across restarts). Each component has a
`MatchRule`:

| Rule | Meaning | Fits |
|------|---------|------|
| `CONTIGUOUS` | resident on **every** block of the prefix | full attention |
| `TRAILING_WINDOW(window_tokens)` | resident contiguously over ≥ `window_tokens` tokens ending at the boundary (an unbroken run from block 0 is always valid) | sliding-window attention |
| `EXACT_BOUNDARY` | resident on the **boundary block only** | Mamba checkpoints |

The reusable prefix for a worker is the largest boundary `N` at which **every**
required component's rule holds. A component is "available" at a block if it is
resident at a tier that is both declared servable by the rule and servable by the
indexer (**V1: HBM or DRAM; SSD is not counted**); different components may live
on different tiers. The RPC still returns a single `matched_prefix_blocks`
(device vs host-loadable prefixes are not split in V1).

Placement is stored per `(hash, worker, tier)`: the Redis placement HASH keys
each `(worker, tier)` as its own field (`worker_id \x1f tier`) whose value is the
tier's component set. A `BlockStored` is a **REPLACE snapshot** for that
`(hash, tier)` — a partial eviction restates a smaller set and never emits
`BlockRemoved`. Per-tier fields make a REPLACE one `HSET` and a per-tier revoke
one `HDEL`, so **removing one tier never disturbs another**. A reserved field
co-locates the block's token count for trailing-window accumulation.

The prefix rule engine (`service.rs::compute_worker_prefix`) is backend-agnostic
and is the single definition of the semantics; the trait default and the Redis
fast path both feed it, so they cannot drift (guarded by
`prefix_fast_path_matches_default_impl` and `component_prefix_matches_default_impl`).

## Safety / exclusion rules

- A worker that reports component data but has **no spec**, or whose spec carries
  an unknown/unusable rule, is **excluded** from prefix results (only that
  worker; if all candidates are excluded the response is empty → the router maps
  it to `NoSignal`). The indexer never guesses.
- The "never over-report" guarantee covers component semantics **when the index
  is accurate**. It does not cover stale index from dropped events or worker
  restarts — consistent with this build's advisory, unfenced design.

## Backward compatibility

| Scenario | Wire | Behaviour |
|----------|------|-----------|
| Flag off (any cache) | `component_types = None` | legacy whole-block, **identical to before** |
| Flag on, full-only tree | `None` (gated on multi-component) | legacy = before |
| Flag on, hybrid, spec configured | component list | component-aware |
| Flag on, hybrid, **no spec** | component list | worker excluded (NoSignal) |
| Pure legacy deployment | `None` | before |

The bridge accepts both the 7-element (legacy) and 8-element (component-aware)
`BlockStored` wire shapes.

**Deployment constraint:** whenever a worker runs with
`--enable-kv-events-component-types` on a hybrid model, configure a matching
spec, or that worker will not participate in cache-aware routing.

## Configuring a worker (bridge)

The bridge builds the spec from the environment (opaque labels; the indexer never
hardcodes `full/swa/mamba`):

```
KV_INDEXER_CACHE_SPEC="full:contiguous;swa:trailing_window:window=4096:tiers=HBM"
KV_INDEXER_CACHE_SPEC_VERSION=1   # optional, default 1
```

Grammar: `name:rule[:window=<tokens>][:tiers=HBM+DRAM]`, components separated by
`;`. `tiers` defaults to `HBM+DRAM`. Unset `KV_INDEXER_CACHE_SPEC` ⇒ a legacy /
full-only worker.
