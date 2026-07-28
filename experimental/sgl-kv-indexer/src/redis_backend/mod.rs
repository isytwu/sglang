// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Redis storage backend for the KV indexer.
//!
//! Data model (see [`schema`]): placement is a per-block-hash HASH of
//! `worker -> tier bitmask`; a per-worker SET is the reverse index; a per-worker
//! durable HASH holds the registry, sequence, incarnation fencing, retired
//! tokens, and liveness deadline; hit counts live in a per-hash HASH co-located
//! with placement (and are deleted together with the placement
//! when a block is fully revoked, so they never outlive the block). All writes
//! flow through
//! [`RedisKvIndexerBackend::apply`], which is naturally idempotent (bit set/clear,
//! SADD/SREM), so a verbatim batch replay (same `seq`) is a no-op and never
//! double-counts hits (hits are only bumped on match).
//!
//! Ordering / idempotency is anchored by a durable per-worker seq stored in the
//! worker meta hash (`{w:worker}:meta` field `seq`). Each apply first runs a
//! seq gate: a batch whose seq is `<= last_applied` is skipped as a duplicate;
//! otherwise the batch is applied and the stored seq is advanced to
//! `max(last, seq)` *after* the mutations commit. The response carries
//! `last_applied_seq` so the bridge can resume from the durable position after a
//! restart. Crash between mutations and the seq advance is safe: the stored seq
//! stays behind, so the next (idempotent) replay re-applies rather than skips.
//!
//! On Cluster an apply batch spans many slots and is therefore not globally
//! atomic; each block-hash mutation is atomic on its own slot and the batch is
//! idempotent, so a partial failure is corrected by the bridge replaying the
//! whole batch under the same seq.

mod conn;
mod schema;
mod scripts;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::future::try_join_all;
use redis::FromRedisValue;
use tonic::Status;

use crate::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, ExternalKvActionType,
    ExternalKvNodeMatch, ExternalKvPrefixMatch, GetExternalKvHitCountsRequest,
    GetExternalKvHitCountsResponse, HashSpec, HitCountEntry, MatchExternalKvPrefixRequest,
    MatchExternalKvPrefixResponse, MatchExternalKvRequest, MatchExternalKvResponse, TierHashes,
};
use crate::service::{prefix_response, KvIndexerBackend};

use conn::{ClusterConn, RedisConn, SingleConn};
use schema::{
    hit_key, placement_key, tier_bit, tiers_from_mask, worker_blocks_key, worker_meta_key,
};
use scripts::{
    HIT_BUMP, MATCH_HASH, PLACEMENT_CLEAR, PLACEMENT_CLEAR_WORKER, PLACEMENT_SET, RESET_FINISH,
    SEQ_CHECK, SEQ_COMMIT, TOUCH_META, WORKER_VIEW,
};

/// Boxed error for construction paths (env parsing / connect / ping).
type BoxError = Box<dyn std::error::Error + Send + Sync>;

const DEFAULT_NAMESPACE: &str = "kvidx";
/// Per-request bound on concurrent Redis operations. Requests may contain more
/// hashes (up to the service-layer protocol limit), but every read/write path
/// processes them in sequential chunks so one request cannot create unbounded
/// in-flight work against Redis.
const REDIS_FANOUT_CHUNK: usize = 256;
/// `COUNT` hint for reverse-index `SSCAN` iteration. Sized to match the fanout
/// chunk so one scanned page turns into one round of concurrent cleanup.
const REVERSE_SCAN_PAGE: usize = 256;
/// First look-ahead window of the prefix scan, doubling up to
/// [`REDIS_FANOUT_CHUNK`] as the prefix proves itself.
///
/// The window trades over-reading against round trips. Reading the whole
/// request at once wastes work on the common case of a short shared prefix;
/// reading one block at a time turns a long prefix into hundreds of sequential
/// round trips. Doubling from a small window keeps the over-read below one
/// window for short prefixes and the round trips logarithmic for long ones.
const PREFIX_SCAN_CHUNK: usize = 32;
/// How far a prefix scan will read before answering with what it has.
///
/// A routing query runs on a millisecond budget and only needs enough of the
/// prefix to rank workers; resolving a 16k-block prefix exactly would cost
/// dozens of serial Redis rounds to sharpen a decision that was already made.
/// At SGLang's usual page sizes this covers a context far larger than any real
/// request, and `queried_blocks` reports when the cap was reached.
const PREFIX_SCAN_MAX_BLOCKS: usize = 2_048;
/// How many leading blocks a hit-counted prefix query credits.
///
/// Hit counts are popularity telemetry, and the head of a prefix is what
/// carries the signal. Crediting every block of a long match would turn the
/// cheapest read in the system into its largest write.
const PREFIX_HIT_MAX_BLOCKS: usize = 256;

/// Resolved connection target parsed from the environment.
enum Target {
    Single(String),
    Cluster(Vec<String>),
}

struct WorkerTouch {
    reset_needed: bool,
    generation: i64,
}

/// One worker's entry in a block hash's placement hash.
struct PlacementOwner {
    worker: String,
    generation: i64,
    mask: i64,
}

/// A worker's durable registry entry, as read for a match.
struct WorkerView {
    address: String,
    alive: bool,
    generation: i64,
    /// Encoded hash spec, empty when the worker never reported one.
    spec: String,
    dp_rank: u32,
}

/// A worker still being extended (or already stalled) by the prefix scan.
struct PrefixCandidate {
    worker: String,
    address: String,
    dp_rank: u32,
    generation: i64,
    tiers: BTreeSet<i32>,
    matched: usize,
}

struct ReverseMember {
    encoded: String,
    generation: i64,
    hash: String,
}

impl ReverseMember {
    fn parse(encoded: String) -> Self {
        let (generation, hash) = parse_reverse_member(&encoded);
        let hash = hash.to_string();
        Self {
            generation,
            hash,
            encoded,
        }
    }
}

pub struct RedisKvIndexerBackend {
    conn: Arc<dyn RedisConn>,
    ns: String,
    worker_locks: Mutex<HashMap<String, Weak<tokio::sync::Mutex<()>>>>,
    /// When set, every apply refreshes a separate per-worker liveness key with
    /// this TTL, and `match` ignores workers whose liveness key has expired.
    /// The durable meta key never expires. `None` disables liveness filtering.
    worker_ttl: Option<Duration>,
}

impl RedisKvIndexerBackend {
    fn new(conn: Arc<dyn RedisConn>, ns: impl Into<String>) -> Self {
        Self {
            conn,
            ns: ns.into(),
            worker_locks: Mutex::new(HashMap::new()),
            worker_ttl: None,
        }
    }

    /// Connects to a single Redis/Dragonfly instance.
    pub async fn connect_single(url: &str, ns: impl Into<String>) -> Result<Self, BoxError> {
        let conn = SingleConn::connect(url).await?;
        Ok(Self::new(Arc::new(conn), ns))
    }

    /// Connects to a Redis Cluster from a list of seed node URLs.
    pub async fn connect_cluster(
        nodes: Vec<String>,
        ns: impl Into<String>,
    ) -> Result<Self, BoxError> {
        let conn = ClusterConn::connect(nodes).await?;
        Ok(Self::new(Arc::new(conn), ns))
    }

    /// Builds a single-instance backend without connecting: the connection is
    /// established lazily on first use. Used for degraded startup so the server
    /// can come up before Redis is reachable and self-heal once it is.
    pub fn connect_single_deferred(url: &str, ns: impl Into<String>) -> Self {
        Self::new(Arc::new(SingleConn::deferred(url)), ns)
    }

    /// Cluster counterpart to [`connect_single_deferred`].
    pub fn connect_cluster_deferred(nodes: Vec<String>, ns: impl Into<String>) -> Self {
        Self::new(Arc::new(ClusterConn::deferred(nodes)), ns)
    }

    /// Sets the per-worker liveness TTL (0 / `None` disables it).
    pub fn with_worker_ttl(mut self, ttl: Option<Duration>) -> Self {
        self.worker_ttl = ttl;
        self
    }

    /// Builds the backend from the environment:
    ///   * `KV_INDEXER_REDIS_NAMESPACE` (default `kvidx`)
    ///   * `KV_INDEXER_REDIS_CLUSTER_NODES` (comma-separated) → Cluster, else
    ///   * `KV_INDEXER_REDIS_URL` → single instance (required)
    ///   * `KV_INDEXER_REDIS_REQUIRED` (default `1`): connect + PING on startup
    ///     and fail fast (bounded, with a clear error) if Redis is unreachable;
    ///     set `0` to start degraded — the server comes up immediately and the
    ///     connection is established lazily / retried on demand.
    ///   * `KV_INDEXER_WORKER_TTL_SECS` (default `120`): per-worker liveness TTL;
    ///     a worker that stops applying/heartbeating for this long is dropped
    ///     from `match`. `0` disables liveness (keys never expire).
    pub async fn from_env() -> Result<Self, BoxError> {
        let ns = std::env::var("KV_INDEXER_REDIS_NAMESPACE")
            .unwrap_or_else(|_| DEFAULT_NAMESPACE.into());
        let required = std::env::var("KV_INDEXER_REDIS_REQUIRED")
            .map(|v| v != "0")
            .unwrap_or(true);
        let worker_ttl = parse_worker_ttl()?;

        let target = if let Ok(nodes) = std::env::var("KV_INDEXER_REDIS_CLUSTER_NODES") {
            let nodes: Vec<String> = nodes
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            if nodes.is_empty() {
                return Err("KV_INDEXER_REDIS_CLUSTER_NODES is empty".into());
            }
            Target::Cluster(nodes)
        } else {
            let url = std::env::var("KV_INDEXER_REDIS_URL").map_err(|_| {
                "KV_INDEXER_REDIS_URL (or KV_INDEXER_REDIS_CLUSTER_NODES) is required for the redis backend"
            })?;
            Target::Single(url)
        };

        if required {
            // Fast, loud startup failure: the connect is bounded by a timeout /
            // limited retries (see conn.rs), then we PING to confirm readiness.
            let backend = match target {
                Target::Cluster(nodes) => Self::connect_cluster(nodes, ns)
                    .await
                    .map_err(|e| format!("redis connect failed: {e}"))?,
                Target::Single(url) => Self::connect_single(&url, ns)
                    .await
                    .map_err(|e| format!("redis connect failed: {e}"))?,
            };
            backend
                .ping()
                .await
                .map_err(|e| format!("redis readiness probe (PING) failed: {e}"))?;
            Ok(backend.with_worker_ttl(worker_ttl))
        } else {
            // Degraded: do not verify Redis now; start serving immediately and
            // connect lazily on first use (requests fail with Unavailable until
            // Redis is reachable, then the manager reconnects automatically).
            tracing::warn!(
                "KV_INDEXER_REDIS_REQUIRED=0: starting degraded; Redis not verified at startup"
            );
            Ok(match target {
                Target::Cluster(nodes) => Self::connect_cluster_deferred(nodes, ns),
                Target::Single(url) => Self::connect_single_deferred(&url, ns),
            }
            .with_worker_ttl(worker_ttl))
        }
    }

    async fn ping(&self) -> redis::RedisResult<()> {
        let _: redis::Value = self.conn.query(redis::cmd("PING").clone()).await?;
        Ok(())
    }

    fn worker_lock(&self, worker: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.worker_locks.lock().unwrap();
        if let Some(lock) = locks.get(worker).and_then(Weak::upgrade) {
            return lock;
        }
        if locks.len() >= 1024 {
            locks.retain(|_, lock| lock.strong_count() > 0);
        }
        let lock = Arc::new(tokio::sync::Mutex::new(()));
        locks.insert(worker.to_string(), Arc::downgrade(&lock));
        lock
    }

    // --- write path ---------------------------------------------------------

    async fn apply(
        &self,
        req: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        let worker = req.worker_id.as_str();
        // The bridge is sequential, but request timeout/retry overlap and other
        // callers can still reach one server concurrently. Serialize a worker's
        // gate/mutations/commit within this backend process.
        let worker_lock = self.worker_lock(worker);
        let _worker_guard = worker_lock.lock().await;
        let meta_key = worker_meta_key(&self.ns, worker);

        // Every accepted apply proves liveness and checks whether its incarnation
        // owes a new or previously interrupted reset.
        let touch = self.touch_meta(&meta_key, &req).await?;
        if touch.reset_needed {
            // Reset is idempotent; reset_pending clears only after full success.
            self.reset_worker(worker, &meta_key, touch.generation)
                .await?;
        }

        // An empty-actions batch is a pure heartbeat: liveness was just
        // refreshed; report the current durable seq and return.
        if req.actions.is_empty() {
            let stored = self.read_worker_seq(&meta_key).await?;
            return Ok(ApplyExternalKvBatchResponse {
                last_applied_seq: stored.max(0) as u64,
                duplicate: false,
                has_applied_seq: stored >= 0,
            });
        }

        // Durable idempotency gate: skip a batch whose seq was already applied.
        // `last == -1` means the worker has no stored seq yet (accept any start).
        let gate = self
            .conn
            .invoke(
                &SEQ_CHECK,
                vec![meta_key.clone()],
                vec![req.seq.to_string(), touch.generation.to_string()],
            )
            .await
            .map_err(to_status)?;
        let (gate_status, last) = script_pair(&gate)?;
        require_current(
            gate_status,
            "worker incarnation changed while applying batch",
        )?;
        let proceed = gate_status == 1;
        if !proceed {
            return Ok(ApplyExternalKvBatchResponse {
                last_applied_seq: last.max(0) as u64,
                duplicate: true,
                has_applied_seq: last >= 0,
            });
        }

        for action in &req.actions {
            let bit = tier_bit(action.tier);
            match ExternalKvActionType::try_from(action.r#type) {
                Ok(ExternalKvActionType::ActionReport) => {
                    for chunk in action.hashes.chunks(REDIS_FANOUT_CHUNK) {
                        try_join_all(
                            chunk
                                .iter()
                                .map(|hash| self.report_one(worker, hash, bit, touch.generation)),
                        )
                        .await?;
                    }
                }
                Ok(ExternalKvActionType::ActionRevoke) => {
                    self.revoke_many(worker, &action.hashes, bit, touch.generation)
                        .await?;
                }
                Ok(ExternalKvActionType::ActionClearAllAtTier) => {
                    self.clear_all_at_tier(worker, bit, touch.generation)
                        .await?;
                }
                _ => return Err(Status::invalid_argument("unsupported action type")),
            }
        }

        // Commit seq last so replay repairs a crash after partial mutations.
        let committed = self
            .conn
            .invoke(
                &SEQ_COMMIT,
                vec![meta_key],
                vec![req.seq.to_string(), touch.generation.to_string()],
            )
            .await
            .map_err(to_status)?;
        let (status, committed) = script_pair(&committed)?;
        require_current(status, "worker incarnation changed while applying batch")?;
        Ok(ApplyExternalKvBatchResponse {
            last_applied_seq: committed.max(0) as u64,
            duplicate: false,
            has_applied_seq: committed >= 0,
        })
    }

    /// Refreshes liveness and records address, incarnation, hash spec and DP
    /// rank. The returned generation fences mutations; `reset_needed` also
    /// survives interrupted resets.
    async fn touch_meta(
        &self,
        meta_key: &str,
        req: &ApplyExternalKvBatchRequest,
    ) -> Result<WorkerTouch, Status> {
        let ttl_ms = self.worker_ttl.map(|d| d.as_millis() as u64).unwrap_or(0);
        // Empty incarnation is retained for wire compatibility but participates
        // in fencing as one stable legacy token.
        let incarnation = if req.incarnation.is_empty() {
            "__legacy__"
        } else {
            req.incarnation.as_str()
        };
        // An unset spec leaves the stored one alone rather than clearing it, so
        // a heartbeat from a publisher that does not report one cannot silently
        // disable spec checking for a worker that does.
        let spec = req
            .hash_spec
            .as_ref()
            .filter(|spec| spec.block_size > 0)
            .map(encode_hash_spec)
            .unwrap_or_default();
        let v = self
            .conn
            .invoke(
                &TOUCH_META,
                vec![meta_key.to_string()],
                vec![
                    now_ms().to_string(),
                    ttl_ms.to_string(),
                    req.worker_address.clone(),
                    incarnation.to_string(),
                    spec,
                    req.dp_rank.to_string(),
                ],
            )
            .await
            .map_err(to_status)?;
        let (status, generation) = script_pair(&v)?;
        if status < 0 {
            return Err(Status::failed_precondition(format!(
                "worker incarnation has been retired: {incarnation}"
            )));
        }
        Ok(WorkerTouch {
            reset_needed: status == 1,
            generation,
        })
    }

    /// Removes older-generation state, then atomically opens the current
    /// generation. All cleanup steps are idempotent.
    async fn reset_worker(
        &self,
        worker: &str,
        meta_key: &str,
        generation: i64,
    ) -> Result<(), Status> {
        let mut cursor = 0;
        loop {
            let (next, page) = self.reverse_page(worker, cursor).await?;
            let stale: Vec<ReverseMember> = page
                .into_iter()
                .filter(|member| member.generation < generation)
                .collect();
            for chunk in stale.chunks(REDIS_FANOUT_CHUNK) {
                try_join_all(chunk.iter().map(|member| {
                    self.clear_old_reverse_member(worker, &member.encoded, &member.hash, generation)
                }))
                .await?;
            }
            cursor = next;
            if cursor == 0 {
                break;
            }
        }

        // Finish exactly once. A concurrent resetter arriving after another
        // request has already opened the generation must not delete its seq.
        let v = self
            .conn
            .invoke(
                &RESET_FINISH,
                vec![meta_key.to_string()],
                vec![generation.to_string()],
            )
            .await
            .map_err(to_status)?;
        require_current(
            i64::from_redis_value(&v).map_err(to_status)?,
            "worker generation changed while resetting",
        )?;
        Ok(())
    }

    async fn clear_old_reverse_member(
        &self,
        worker: &str,
        member: &str,
        hash: &str,
        generation: i64,
    ) -> Result<(), Status> {
        let v = self
            .conn
            .invoke(
                &PLACEMENT_CLEAR_WORKER,
                vec![placement_key(&self.ns, hash), hit_key(&self.ns, hash)],
                vec![worker.to_string(), generation.to_string()],
            )
            .await
            .map_err(to_status)?;
        let _: i64 = i64::from_redis_value(&v).map_err(to_status)?;
        let mut srem = redis::cmd("SREM");
        srem.arg(worker_blocks_key(&self.ns, worker)).arg(member);
        self.conn.query(srem).await.map_err(to_status)?;
        Ok(())
    }

    /// Reads one `SSCAN` page of a worker's reverse index, returning the next
    /// cursor (`0` once the iteration is complete) alongside the members.
    ///
    /// `SMEMBERS` would occupy Redis for the whole set and materialize it in a
    /// single reply, and a worker legitimately owns as many blocks as its cache
    /// holds. Scanning may return a member more than once and may miss members
    /// added or removed mid-iteration; both callers are idempotent and only act
    /// on generations that are already settled, so neither matters here.
    async fn reverse_page(
        &self,
        worker: &str,
        cursor: u64,
    ) -> Result<(u64, Vec<ReverseMember>), Status> {
        let mut cmd = redis::cmd("SSCAN");
        cmd.arg(worker_blocks_key(&self.ns, worker))
            .arg(cursor)
            .arg("COUNT")
            .arg(REVERSE_SCAN_PAGE);
        let value = self.conn.query(cmd).await.map_err(to_status)?;
        let (next, members) = <(u64, Vec<String>)>::from_redis_value(&value).map_err(to_status)?;
        Ok((
            next,
            members.into_iter().map(ReverseMember::parse).collect(),
        ))
    }

    /// Reads the worker's durable seq (`-1` when none has been stored yet).
    async fn read_worker_seq(&self, meta_key: &str) -> Result<i64, Status> {
        let mut cmd = redis::cmd("HGET");
        cmd.arg(meta_key).arg("seq");
        let v = self.conn.query(cmd).await.map_err(to_status)?;
        Ok(Option::<i64>::from_redis_value(&v)
            .map_err(to_status)?
            .unwrap_or(-1))
    }

    async fn report_one(
        &self,
        worker: &str,
        hash: &str,
        bit: i64,
        generation: i64,
    ) -> Result<(), Status> {
        let v = self
            .conn
            .invoke(
                &PLACEMENT_SET,
                vec![placement_key(&self.ns, hash)],
                vec![worker.to_string(), bit.to_string(), generation.to_string()],
            )
            .await
            .map_err(to_status)?;
        require_current(
            i64::from_redis_value(&v).map_err(to_status)?,
            "newer worker generation already owns placement",
        )?;

        // Always reassert reverse state so replay repairs a failed prior SADD.
        let mut cmd = redis::cmd("SADD");
        cmd.arg(worker_blocks_key(&self.ns, worker))
            .arg(reverse_member(generation, hash));
        self.conn.query(cmd).await.map_err(to_status)?;
        Ok(())
    }

    async fn revoke_many(
        &self,
        worker: &str,
        hashes: &[String],
        bit: i64,
        generation: i64,
    ) -> Result<(), Status> {
        for chunk in hashes.chunks(REDIS_FANOUT_CHUNK) {
            try_join_all(
                chunk
                    .iter()
                    .map(|hash| self.revoke_one(worker, hash, bit, generation)),
            )
            .await?;
        }
        Ok(())
    }

    async fn revoke_one(
        &self,
        worker: &str,
        hash: &str,
        bit: i64,
        generation: i64,
    ) -> Result<(), Status> {
        // Placement and hit keys share a slot; the script drops hits when the
        // final placement disappears.
        let v = self
            .conn
            .invoke(
                &PLACEMENT_CLEAR,
                vec![placement_key(&self.ns, hash), hit_key(&self.ns, hash)],
                vec![worker.to_string(), bit.to_string(), generation.to_string()],
            )
            .await
            .map_err(to_status)?;
        let (status, _) = script_pair(&v)?;
        require_current(status, "newer worker generation already owns placement")?;
        let worker_gone = status == 1;
        if worker_gone {
            let mut cmd = redis::cmd("SREM");
            cmd.arg(worker_blocks_key(&self.ns, worker))
                .arg(reverse_member(generation, hash));
            if generation == 0 {
                cmd.arg(hash);
            }
            self.conn.query(cmd).await.map_err(to_status)?;
        }
        Ok(())
    }

    async fn clear_all_at_tier(
        &self,
        worker: &str,
        bit: i64,
        generation: i64,
    ) -> Result<(), Status> {
        let mut cursor = 0;
        loop {
            let (next, page) = self.reverse_page(worker, cursor).await?;
            let hashes: Vec<String> = page
                .into_iter()
                .filter_map(|member| (member.generation == generation).then_some(member.hash))
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            self.revoke_many(worker, &hashes, bit, generation).await?;
            cursor = next;
            if cursor == 0 {
                return Ok(());
            }
        }
    }

    // --- read path ----------------------------------------------------------

    /// Reads one block hash's placement hash. Legacy values that carry only a
    /// bitmask are interpreted as generation zero.
    async fn read_placement(&self, hash: &str) -> Result<Vec<PlacementOwner>, Status> {
        let value = self
            .conn
            .invoke(&MATCH_HASH, vec![placement_key(&self.ns, hash)], Vec::new())
            .await
            .map_err(to_status)?;
        let flat = Vec::<String>::from_redis_value(&value).map_err(to_status)?;
        Ok(flat
            .chunks(2)
            .filter(|pair| pair.len() == 2)
            .map(|pair| {
                let (generation, mask) = parse_placement_value(&pair[1]);
                PlacementOwner {
                    worker: pair[0].clone(),
                    generation,
                    mask,
                }
            })
            .collect())
    }

    /// [`Self::read_placement`] over a slice, concurrent within a bounded chunk
    /// so one request cannot create unbounded in-flight work against Redis.
    async fn read_placements(&self, hashes: &[String]) -> Result<Vec<Vec<PlacementOwner>>, Status> {
        let mut out = Vec::with_capacity(hashes.len());
        for chunk in hashes.chunks(REDIS_FANOUT_CHUNK) {
            out.extend(try_join_all(chunk.iter().map(|hash| self.read_placement(hash))).await?);
        }
        Ok(out)
    }

    /// Reads the durable registry entry for each worker, in the order given.
    async fn read_worker_views(&self, workers: &[String]) -> Result<Vec<WorkerView>, Status> {
        let ttl_flag = if self.worker_ttl.is_some() { "1" } else { "0" };
        let now_ms = now_ms().to_string();
        let mut views = Vec::with_capacity(workers.len());
        for chunk in workers.chunks(REDIS_FANOUT_CHUNK) {
            let values = try_join_all(chunk.iter().map(|worker| {
                let now_ms = now_ms.clone();
                async move {
                    let v = self
                        .conn
                        .invoke(
                            &WORKER_VIEW,
                            vec![worker_meta_key(&self.ns, worker)],
                            vec![now_ms, ttl_flag.to_string()],
                        )
                        .await?;
                    let (address, alive, generation, spec, dp_rank): (
                        String,
                        i64,
                        i64,
                        String,
                        u32,
                    ) = <(String, i64, i64, String, u32)>::from_redis_value(&v)?;
                    Ok::<_, redis::RedisError>(WorkerView {
                        address,
                        alive: alive == 1,
                        generation,
                        spec,
                        dp_rank,
                    })
                }
            }))
            .await
            .map_err(to_status)?;
            views.extend(values);
        }
        Ok(views)
    }

    /// Prefix-aware routing query.
    ///
    /// Three phases, arranged so the cost tracks the prefix that actually
    /// exists rather than the length of the request:
    ///
    /// 1. Read the first block. Only workers holding it can hold any prefix, so
    ///    a miss here answers the whole query in one round trip.
    /// 2. Read the registry once, for exactly those workers. The candidate set
    ///    only ever shrinks from here, so no later block needs a registry read.
    /// 3. Walk forward in growing chunks. A candidate drops out at its first
    ///    gap, and the scan stops as soon as none are left.
    ///
    /// The number of block hashes read is therefore bounded by roughly twice
    /// the real prefix length plus one chunk, independent of how many blocks
    /// the caller asked about, and never exceeds
    /// [`PREFIX_SCAN_MAX_BLOCKS`].
    ///
    /// The registry read in step 2 is a snapshot. A worker that restarts
    /// *during* the scan can still contribute placement its new incarnation has
    /// not yet erased, so its prefix may be reported longer than it can serve.
    /// The reverse cannot happen: the snapshot's generation rejects the new
    /// incarnation's entries, which can only shorten a prefix. Re-reading the
    /// registry per block would remove the window at the cost of the property
    /// that makes this query cheap, and the index is advisory — a stale
    /// candidate costs one cache miss, which the worker then recomputes.
    async fn do_match_prefix(
        &self,
        req: MatchExternalKvPrefixRequest,
    ) -> Result<MatchExternalKvPrefixResponse, Status> {
        // The service layer rejects an empty query, but the trait is public and
        // its default implementation answers one, so this must too rather than
        // indexing into nothing.
        let Some(first) = req.hashes.first() else {
            return Ok(prefix_response(Vec::new(), req.top_k, 0, 0));
        };
        let hashes = &req.hashes;
        let required_spec = req
            .hash_spec
            .as_ref()
            .filter(|spec| spec.block_size > 0)
            .map(encode_hash_spec);

        let first_owners = self.read_placement(first).await?;
        if first_owners.is_empty() {
            return Ok(prefix_response(Vec::new(), req.top_k, 1, 0));
        }

        let worker_ids: Vec<String> = first_owners
            .iter()
            .map(|owner| owner.worker.clone())
            .collect();
        let views = self.read_worker_views(&worker_ids).await?;

        let mut spec_mismatched_workers = 0u32;
        let mut extending: Vec<PrefixCandidate> = Vec::with_capacity(first_owners.len());
        for (owner, view) in first_owners.into_iter().zip(views) {
            // An empty address is unroutable, so the worker is invisible to a
            // router no matter how much of the prefix it holds.
            if !view.alive || view.address.is_empty() || owner.generation != view.generation {
                continue;
            }
            // A worker that never reported a spec is unchecked rather than
            // rejected. Publishers and routers upgrade independently, and
            // dropping every silent worker the moment a router starts sending
            // specs would take out routing for the length of the rollout.
            if let Some(required) = &required_spec {
                if !view.spec.is_empty() && &view.spec != required {
                    spec_mismatched_workers += 1;
                    continue;
                }
            }
            extending.push(PrefixCandidate {
                worker: owner.worker,
                address: view.address,
                dp_rank: view.dp_rank,
                generation: view.generation,
                tiers: tiers_from_mask(owner.mask).into_iter().collect(),
                matched: 1,
            });
        }
        if extending.is_empty() {
            return Ok(prefix_response(
                Vec::new(),
                req.top_k,
                1,
                spec_mismatched_workers,
            ));
        }

        let scan_limit = hashes.len().min(PREFIX_SCAN_MAX_BLOCKS);
        let mut stalled: Vec<PrefixCandidate> = Vec::new();
        let mut queried = 1usize;
        let mut chunk = PREFIX_SCAN_CHUNK;
        while queried < scan_limit && !extending.is_empty() {
            let end = (queried + chunk).min(scan_limit);
            let rows = self.read_placements(&hashes[queried..end]).await?;
            // Count the whole chunk: it was read, even if the prefix broke part
            // way through it. Reporting less would hide the scan's real cost.
            queried = end;
            for owners in rows {
                let owned: HashMap<&str, &PlacementOwner> = owners
                    .iter()
                    .map(|owner| (owner.worker.as_str(), owner))
                    .collect();
                let mut still = Vec::with_capacity(extending.len());
                for mut candidate in extending.drain(..) {
                    match owned.get(candidate.worker.as_str()) {
                        Some(owner) if owner.generation == candidate.generation => {
                            candidate.tiers.extend(tiers_from_mask(owner.mask));
                            candidate.matched += 1;
                            still.push(candidate);
                        }
                        _ => stalled.push(candidate),
                    }
                }
                extending = still;
                if extending.is_empty() {
                    break;
                }
            }
            chunk = chunk.saturating_mul(2).min(REDIS_FANOUT_CHUNK);
        }
        stalled.append(&mut extending);

        if req.count_as_hit {
            // The answer is already computed, and this is telemetry. Failing
            // the query here would turn a counter write into a routing outage:
            // the caller reads that as an unreachable index and, after a few
            // such queries, trips its breaker and stops asking at all.
            if let Err(error) = self.bump_prefix_hits(hashes, &stalled).await {
                tracing::warn!(%error, "prefix hit counting failed; returning the match anyway");
            }
        }

        let matches = stalled
            .into_iter()
            .map(|candidate| ExternalKvPrefixMatch {
                worker_id: candidate.worker,
                address: candidate.address,
                dp_rank: candidate.dp_rank,
                matched_prefix_blocks: candidate.matched as u32,
                tiers: candidate.tiers.into_iter().collect(),
            })
            .collect();
        Ok(prefix_response(
            matches,
            req.top_k,
            queried as u32,
            spec_mismatched_workers,
        ))
    }

    /// Credits a hit to every block a returned worker can actually reuse.
    ///
    /// Unlike [`Self::do_match`], which counts any matched hash, this counts
    /// only the contiguous prefix the caller can serve from that worker — the
    /// blocks a routing decision based on this response would really reuse —
    /// and only the first [`PREFIX_HIT_MAX_BLOCKS`] of it.
    async fn bump_prefix_hits(
        &self,
        hashes: &[String],
        candidates: &[PrefixCandidate],
    ) -> Result<(), Status> {
        let now = now_ms().to_string();
        let best = candidates
            .iter()
            .map(|candidate| candidate.matched)
            .max()
            .unwrap_or(0)
            .min(PREFIX_HIT_MAX_BLOCKS);
        let bumps: Vec<(&String, Vec<&PrefixCandidate>)> = (0..best)
            .map(|index| {
                let owners = candidates
                    .iter()
                    .filter(|candidate| candidate.matched > index)
                    .collect();
                (&hashes[index], owners)
            })
            .collect();
        for chunk in bumps.chunks(REDIS_FANOUT_CHUNK) {
            try_join_all(chunk.iter().map(|(hash, owners)| {
                let mut args = Vec::with_capacity(1 + owners.len() * 2);
                args.push(now.clone());
                for owner in owners {
                    args.push(owner.worker.clone());
                    args.push(owner.generation.to_string());
                }
                self.conn.invoke(
                    &HIT_BUMP,
                    vec![placement_key(&self.ns, hash), hit_key(&self.ns, hash)],
                    args,
                )
            }))
            .await
            .map_err(to_status)?;
        }
        Ok(())
    }

    async fn do_match(
        &self,
        req: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        let hashes = dedup_preserve_order(&req.hashes);

        // Per-hash placement read, preserving hash order. Hit counting happens
        // after worker liveness and generation have been validated.
        let per_hash = self.read_placements(&hashes).await?;

        // Keep the placement generation alongside each candidate. Legacy
        // numeric values are generation zero.
        let mut worker_order: Vec<String> = Vec::new();
        let mut by_worker: HashMap<String, Vec<(String, i64, i64)>> = HashMap::new();
        for (hash, owners) in hashes.iter().zip(per_hash) {
            for owner in owners {
                let entry = by_worker.entry(owner.worker.clone()).or_insert_with(|| {
                    worker_order.push(owner.worker.clone());
                    Vec::new()
                });
                entry.push((hash.clone(), owner.generation, owner.mask));
            }
        }

        // Fetch each matched worker's `addr` (for routing) and liveness. The
        // durable meta hash stores `live_until_ms`; stale workers are dropped so
        // `match` never routes to a dead node while placement entries linger.
        let metas = self.read_worker_views(&worker_order).await?;

        let mut matches = Vec::new();
        let mut matched_hashes: HashMap<String, Vec<(String, i64)>> = HashMap::new();
        for (worker, view) in worker_order.into_iter().zip(metas) {
            if !view.alive {
                continue;
            }
            let mut by_tier: BTreeMap<i32, Vec<String>> = BTreeMap::new();
            for (hash, generation, mask) in by_worker.remove(&worker).unwrap_or_default() {
                if generation != view.generation {
                    continue;
                }
                for tier in tiers_from_mask(mask) {
                    by_tier.entry(tier).or_default().push(hash.clone());
                }
                matched_hashes
                    .entry(hash)
                    .or_default()
                    .push((worker.clone(), view.generation));
            }
            if by_tier.is_empty() {
                continue;
            }
            matches.push(ExternalKvNodeMatch {
                worker_id: worker,
                address: view.address,
                hashes_by_tier: by_tier
                    .into_iter()
                    .map(|(tier, hashes)| TierHashes { tier, hashes })
                    .collect(),
                dp_rank: view.dp_rank,
            });
        }

        if req.count_as_hit {
            let matched_hashes: Vec<(String, Vec<(String, i64)>)> =
                matched_hashes.into_iter().collect();
            let now = now_ms().to_string();
            for chunk in matched_hashes.chunks(REDIS_FANOUT_CHUNK) {
                try_join_all(chunk.iter().map(|(hash, owners)| {
                    let mut args = Vec::with_capacity(1 + owners.len() * 2);
                    args.push(now.clone());
                    for (worker, generation) in owners {
                        args.push(worker.clone());
                        args.push(generation.to_string());
                    }
                    self.conn.invoke(
                        &HIT_BUMP,
                        vec![placement_key(&self.ns, hash), hit_key(&self.ns, hash)],
                        args,
                    )
                }))
                .await
                .map_err(to_status)?;
            }
        }

        Ok(MatchExternalKvResponse { matches })
    }

    async fn do_hit_counts(
        &self,
        req: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        let hashes = dedup_preserve_order(&req.hashes);
        let mut counts = Vec::with_capacity(hashes.len());
        for chunk in hashes.chunks(REDIS_FANOUT_CHUNK) {
            let values = try_join_all(chunk.iter().map(|hash| async move {
                let mut cmd = redis::cmd("HGET");
                cmd.arg(hit_key(&self.ns, hash)).arg("c");
                let v = self.conn.query(cmd).await?;
                let count: Option<i64> = Option::<i64>::from_redis_value(&v)?;
                Ok::<_, redis::RedisError>(count)
            }))
            .await
            .map_err(to_status)?;
            counts.extend(values);
        }

        let entries = hashes
            .into_iter()
            .zip(counts)
            .filter_map(|(hash, count)| {
                count.map(|c| HitCountEntry {
                    hash,
                    hit_count_total: c.max(0) as u64,
                })
            })
            .collect();
        Ok(GetExternalKvHitCountsResponse { entries })
    }
}

#[tonic::async_trait]
impl KvIndexerBackend for RedisKvIndexerBackend {
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        self.apply(request).await
    }

    async fn match_external_kv(
        &self,
        request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        self.do_match(request).await
    }

    async fn match_external_kv_prefix(
        &self,
        request: MatchExternalKvPrefixRequest,
    ) -> Result<MatchExternalKvPrefixResponse, Status> {
        self.do_match_prefix(request).await
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        self.do_hit_counts(request).await
    }

    /// Ready only when Redis answers a PING (covers degraded/lazy-connect starts
    /// and transient store outages).
    async fn health(&self) -> bool {
        self.ping().await.is_ok()
    }
}

fn dedup_preserve_order(hashes: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    hashes
        .iter()
        .filter(|h| seen.insert(h.as_str().to_string()))
        .cloned()
        .collect()
}

fn script_pair(value: &redis::Value) -> Result<(i64, i64), Status> {
    let values = Vec::<i64>::from_redis_value(value).map_err(to_status)?;
    values
        .first()
        .zip(values.get(1))
        .map(|(&first, &second)| (first, second))
        .ok_or_else(|| Status::internal("invalid Redis script response"))
}

fn require_current(status: i64, message: &'static str) -> Result<(), Status> {
    (status >= 0)
        .then_some(())
        .ok_or_else(|| Status::failed_precondition(message))
}

/// Encodes a hash spec into the single string stored on the worker.
///
/// Only ever compared for equality, never parsed back, so the encoding just has
/// to be injective: `namespace` goes last because it is the one field that may
/// contain the separator.
fn encode_hash_spec(spec: &HashSpec) -> String {
    format!(
        "{}:{}:{}:{}",
        spec.block_size, spec.algo, spec.version, spec.namespace
    )
}

fn parse_placement_value(value: &str) -> (i64, i64) {
    match value.split_once(':') {
        Some((generation, mask)) => (generation.parse().unwrap_or(0), mask.parse().unwrap_or(0)),
        None => (0, value.parse().unwrap_or(0)),
    }
}

fn reverse_member(generation: i64, hash: &str) -> String {
    format!("{generation}:{hash}")
}

fn parse_reverse_member(member: &str) -> (i64, &str) {
    match member.split_once(':') {
        Some((generation, hash)) => match generation.parse() {
            Ok(generation) => (generation, hash),
            Err(_) => (0, member),
        },
        None => (0, member),
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn to_status(err: redis::RedisError) -> Status {
    Status::unavailable(format!("redis backend error: {err}"))
}

/// Parses `KV_INDEXER_WORKER_TTL_SECS` (default `120`; `0` disables liveness).
fn parse_worker_ttl() -> Result<Option<Duration>, BoxError> {
    const DEFAULT_WORKER_TTL_SECS: u64 = 120;
    let secs = match std::env::var("KV_INDEXER_WORKER_TTL_SECS") {
        Ok(v) => v.trim().parse::<u64>().map_err(|_| {
            format!("KV_INDEXER_WORKER_TTL_SECS must be a non-negative integer, got {v:?}")
        })?,
        Err(_) => DEFAULT_WORKER_TTL_SECS,
    };
    Ok((secs > 0).then(|| Duration::from_secs(secs)))
}

#[cfg(test)]
mod fault_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_preserves_first_seen_order() {
        let input = vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
            "c".to_string(),
            "b".to_string(),
        ];
        assert_eq!(dedup_preserve_order(&input), vec!["a", "b", "c"]);
    }

    #[test]
    fn placement_value_parser_is_backward_compatible() {
        assert_eq!(parse_placement_value("6"), (0, 6));
        assert_eq!(parse_placement_value("3:10"), (3, 10));
        assert_eq!(parse_placement_value("bad"), (0, 0));
    }

    #[test]
    fn reverse_member_parser_is_backward_compatible() {
        assert_eq!(parse_reverse_member("hash-a"), (0, "hash-a"));
        assert_eq!(parse_reverse_member("legacy:hash"), (0, "legacy:hash"));
        assert_eq!(parse_reverse_member("3:hash-a"), (3, "hash-a"));
        assert_eq!(reverse_member(3, "hash-a"), "3:hash-a");
    }

    #[test]
    fn worker_lock_is_shared_while_active_and_prunes_inactive_entries() {
        let backend = RedisKvIndexerBackend::connect_single_deferred("redis://127.0.0.1:1", "test");
        let first = backend.worker_lock("worker-a");
        let same = backend.worker_lock("worker-a");
        assert!(Arc::ptr_eq(&first, &same));
        drop(first);
        drop(same);

        for index in 0..1100 {
            drop(backend.worker_lock(&format!("worker-{index}")));
        }
        assert!(backend.worker_locks.lock().unwrap().len() < 1024);
    }
}
