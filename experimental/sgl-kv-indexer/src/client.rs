// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Routing-side client for the prefix index.
//!
//! A router consults the index to guess which worker already holds a request's
//! prefix. That guess is advisory: the workers are authoritative, and a router
//! that cannot reach the index must still route. The API is shaped around that
//! rule — [`ExternalPrefixIndex::match_prefix`] returns
//! [`PrefixIndexOutcome`], which has no error variant, so an index outage
//! cannot be accidentally propagated into a failed inference request. Every
//! failure becomes [`PrefixIndexOutcome::NoSignal`] and the caller falls back
//! to whatever it would have done without an index.
//!
//! The second rule is that the query runs on a routing hot path. The default
//! budget is single-digit milliseconds, and repeated failures trip a breaker so
//! an unreachable index costs one atomic load per request rather than a
//! connection attempt.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tonic::transport::{Channel, Endpoint};
use tonic::{Code, Status};
use tracing::warn;

use crate::pb::kv_indexer_client::KvIndexerClient;
use crate::pb::{HashAlgorithm, HashSpec, MatchExternalKvPrefixRequest, TierType};
use crate::service::MAX_HASHES_PER_REQUEST;

/// Default per-query budget. Deliberately far below the bridge's RPC timeout:
/// an apply may wait seconds for a durable write, but a routing query that
/// takes that long has already cost more than the cache hit it was looking for.
pub const DEFAULT_QUERY_TIMEOUT: Duration = Duration::from_millis(10);
/// Consecutive outages before the breaker opens.
pub const DEFAULT_BREAKER_THRESHOLD: u32 = 5;
/// How long the breaker stays open before queries resume.
///
/// Note the interaction with a cold channel: connecting takes longer than a
/// query budget, so a router's first few queries fail and open the breaker
/// before the channel is up. The connection continues establishing in the
/// background, so a router recovers after one cooldown rather than staying
/// down — but it does start out without cache-aware routing for that long.
pub const DEFAULT_BREAKER_COOLDOWN: Duration = Duration::from_secs(5);
/// How often a persistent hash-space disagreement is reported. It stays true
/// until someone changes the configuration, so it must not be logged per query.
const SPEC_MISMATCH_LOG_INTERVAL: Duration = Duration::from_secs(60);

/// Renders a block hash the way SGLang publishes it, which is also the form the
/// indexer stores. Routers compute `i64` hashes; this is the only supported
/// conversion to the wire's string form.
pub fn hash_to_wire(hash: i64) -> String {
    hash.to_string()
}

/// Inverse of [`hash_to_wire`]. `None` for a hash that did not originate from
/// SGLang's numeric encoding.
pub fn hash_from_wire(wire: &str) -> Option<i64> {
    wire.parse::<i64>().ok()
}

impl HashSpec {
    /// The spec for SGLang's block hashing at a worker's page size. `bigram` is
    /// true for EAGLE-family models, which hash over overlapping token pairs.
    pub fn sglang(block_size: u32, bigram: bool) -> Self {
        let algo = if bigram {
            HashAlgorithm::HashAlgoSha256ChainBigram
        } else {
            HashAlgorithm::HashAlgoSha256ChainUnigram
        };
        Self {
            block_size,
            algo: algo as i32,
            version: 0,
            namespace: String::new(),
        }
    }
}

/// A routing query: the request's block hashes in order, and the hash space
/// they were computed in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixQuery {
    /// Ordered block hashes; `hashes[0]` must be the request's first block.
    pub hashes: Vec<i64>,
    pub spec: HashSpec,
    /// Stop after this many blocks. 0 means "all of them".
    pub max_blocks: u32,
    /// Return at most this many workers. 0 means "all of them".
    pub top_k: u32,
    pub count_as_hit: bool,
}

impl PrefixQuery {
    pub fn new(hashes: Vec<i64>, spec: HashSpec) -> Self {
        Self {
            hashes,
            spec,
            max_blocks: 0,
            top_k: 0,
            count_as_hit: false,
        }
    }
}

/// One worker's reusable prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMatch {
    pub worker_id: String,
    /// The worker's routing identity. A router matches this against its own
    /// registered worker URL, so the two must agree byte for byte.
    pub address: String,
    pub dp_rank: u32,
    /// How many leading blocks of the query this worker holds contiguously.
    pub matched_prefix_blocks: u32,
    pub tiers: Vec<TierType>,
}

/// Why a query produced no usable routing signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoSignalReason {
    /// The index answered, but no worker holds even the first block.
    Empty,
    /// The index could not be reached, or the breaker is open.
    Unavailable,
    /// The query exceeded its budget.
    Timeout,
    /// The server does not implement this query — it predates the RPC. Like an
    /// outage this is a property of the index rather than of one request, so it
    /// opens the breaker instead of costing a round trip per request.
    Unsupported,
    /// Workers hold the prefix but recorded a different hash space. The caller
    /// and the workers disagree on block size or hashing mode, so no query will
    /// ever match until the configuration is fixed.
    SpecMismatch,
    /// The index rejected the query: a malformed request, or a server too old
    /// to implement it.
    Rejected,
}

/// The result of a routing query. There is no error variant on purpose: an
/// advisory index must not be able to fail a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixIndexOutcome {
    Matched {
        /// Longest prefix first.
        matches: Vec<PrefixMatch>,
        best_prefix_blocks: u32,
    },
    NoSignal(NoSignalReason),
}

impl PrefixIndexOutcome {
    /// The workers tied for the longest prefix — the set a load-balancing
    /// caller chooses between. Empty when there was no signal.
    pub fn best_matches(&self) -> &[PrefixMatch] {
        let Self::Matched {
            matches,
            best_prefix_blocks,
        } = self
        else {
            return &[];
        };
        // Matches are sorted longest first, so the tied set is a prefix of it.
        let tied = matches
            .iter()
            .take_while(|m| m.matched_prefix_blocks == *best_prefix_blocks)
            .count();
        &matches[..tied]
    }

    /// The longest prefix any worker holds; 0 when there was no signal.
    pub fn best_prefix_blocks(&self) -> u32 {
        match self {
            Self::Matched {
                best_prefix_blocks, ..
            } => *best_prefix_blocks,
            Self::NoSignal(_) => 0,
        }
    }
}

/// A source of prefix-match answers. Implemented by [`KvIndexerPrefixIndex`]
/// over gRPC, and by callers' own types in tests or for an in-process index.
#[tonic::async_trait]
pub trait ExternalPrefixIndex: Send + Sync {
    async fn match_prefix(&self, query: &PrefixQuery) -> PrefixIndexOutcome;
}

#[derive(Debug, Clone)]
pub struct PrefixIndexConfig {
    /// gRPC endpoint, e.g. `http://127.0.0.1:50051`.
    pub endpoint: String,
    pub query_timeout: Duration,
    pub connect_timeout: Duration,
    pub breaker_threshold: u32,
    pub breaker_cooldown: Duration,
}

impl PrefixIndexConfig {
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            query_timeout: DEFAULT_QUERY_TIMEOUT,
            connect_timeout: Duration::from_secs(1),
            breaker_threshold: DEFAULT_BREAKER_THRESHOLD,
            breaker_cooldown: DEFAULT_BREAKER_COOLDOWN,
        }
    }
}

/// Skips queries while the index is known to be failing, so an outage costs an
/// atomic load per request instead of a connection attempt and a full timeout.
///
/// Deadlines are measured against a monotonic clock. A wall clock would work
/// until the host's time stepped backwards, at which point a breaker that had
/// just opened would stay open for the size of the correction — and because
/// only a successful query closes it, and no query is issued while it is open,
/// nothing would ever reopen it short of a restart.
#[derive(Debug)]
struct Breaker {
    threshold: u32,
    cooldown: Duration,
    base: Instant,
    consecutive_failures: AtomicU32,
    /// Milliseconds since `base`; 0 means closed.
    open_until_ms: AtomicU64,
}

impl Breaker {
    fn new(threshold: u32, cooldown: Duration) -> Self {
        Self {
            threshold,
            cooldown,
            base: Instant::now(),
            consecutive_failures: AtomicU32::new(0),
            open_until_ms: AtomicU64::new(0),
        }
    }

    fn elapsed_ms(&self) -> u64 {
        self.base.elapsed().as_millis() as u64
    }

    fn is_open(&self) -> bool {
        self.open_until_ms.load(Ordering::Relaxed) > self.elapsed_ms()
    }

    fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
        self.open_until_ms.store(0, Ordering::Relaxed);
    }

    fn record_failure(&self) {
        let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        if self.threshold > 0 && failures >= self.threshold {
            let until = self
                .elapsed_ms()
                .saturating_add(self.cooldown.as_millis() as u64)
                // 0 is the closed sentinel, so never land on it.
                .max(1);
            self.open_until_ms.store(until, Ordering::Relaxed);
            self.consecutive_failures.store(0, Ordering::Relaxed);
        }
    }
}

/// Allows one message per interval. A hash-space disagreement persists until
/// someone fixes the configuration, so reporting it per query would flood the
/// log at routing QPS — while reporting it once per process would go silent on a
/// misconfiguration that recurs later in a long-lived router.
///
/// Two threads can pass the gate together and emit a duplicate line. It guards
/// nothing but log volume, so that is not worth a compare-exchange.
#[derive(Debug)]
struct LogThrottle {
    base: Instant,
    interval: Duration,
    /// Milliseconds since `base` before which no message is emitted.
    next_ms: AtomicU64,
}

impl LogThrottle {
    fn new(interval: Duration) -> Self {
        Self {
            base: Instant::now(),
            interval,
            next_ms: AtomicU64::new(0),
        }
    }

    fn allow(&self) -> bool {
        let now = self.base.elapsed().as_millis() as u64;
        if self.next_ms.load(Ordering::Relaxed) > now {
            return false;
        }
        self.next_ms.store(
            now.saturating_add(self.interval.as_millis() as u64).max(1),
            Ordering::Relaxed,
        );
        true
    }
}

/// gRPC-backed [`ExternalPrefixIndex`].
pub struct KvIndexerPrefixIndex {
    client: KvIndexerClient<Channel>,
    query_timeout: Duration,
    breaker: Breaker,
    spec_mismatch_log: LogThrottle,
}

impl std::fmt::Debug for KvIndexerPrefixIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvIndexerPrefixIndex")
            .field("query_timeout", &self.query_timeout)
            .field("breaker", &self.breaker)
            .finish()
    }
}

impl KvIndexerPrefixIndex {
    /// Builds a client without waiting for a connection. Connecting eagerly
    /// would make router startup depend on an advisory service; instead the
    /// first queries return `NoSignal` until the channel comes up.
    pub fn connect_lazy(config: PrefixIndexConfig) -> Result<Self, tonic::transport::Error> {
        let channel = Endpoint::from_shared(config.endpoint)?
            .connect_timeout(config.connect_timeout)
            .connect_lazy();
        Ok(Self {
            client: KvIndexerClient::new(channel),
            query_timeout: config.query_timeout,
            breaker: Breaker::new(config.breaker_threshold, config.breaker_cooldown),
            spec_mismatch_log: LogThrottle::new(SPEC_MISMATCH_LOG_INTERVAL),
        })
    }
}

#[tonic::async_trait]
impl ExternalPrefixIndex for KvIndexerPrefixIndex {
    async fn match_prefix(&self, query: &PrefixQuery) -> PrefixIndexOutcome {
        if query.hashes.is_empty() {
            return PrefixIndexOutcome::NoSignal(NoSignalReason::Empty);
        }
        if self.breaker.is_open() {
            return PrefixIndexOutcome::NoSignal(NoSignalReason::Unavailable);
        }

        let mut request = tonic::Request::new(build_request(query));
        // The header lets the server abandon work it can no longer deliver; the
        // wrapper below is what actually bounds the caller, since the header
        // alone depends on the server honoring it.
        request.set_timeout(self.query_timeout);

        let mut client = self.client.clone();
        let call = client.match_external_kv_prefix(request);
        let response = match tokio::time::timeout(self.query_timeout, call).await {
            Ok(Ok(response)) => response.into_inner(),
            Ok(Err(status)) => {
                let reason = classify_status(&status);
                if is_index_outage(reason) {
                    self.breaker.record_failure();
                }
                return PrefixIndexOutcome::NoSignal(reason);
            }
            Err(_) => {
                self.breaker.record_failure();
                return PrefixIndexOutcome::NoSignal(NoSignalReason::Timeout);
            }
        };
        self.breaker.record_success();

        if response.matches.is_empty() {
            // A cold cache and a hash-space disagreement look identical from
            // here — both return nothing — which is exactly why the server
            // reports the mismatch count separately.
            if response.spec_mismatched_workers > 0 {
                if self.spec_mismatch_log.allow() {
                    warn!(
                        spec_mismatched_workers = response.spec_mismatched_workers,
                        block_size = query.spec.block_size,
                        algo = query.spec.algo,
                        "kv-indexer: workers hold this prefix but recorded a different hash \
                         space; cache-aware routing will never match until block size and \
                         hashing mode agree"
                    );
                }
                return PrefixIndexOutcome::NoSignal(NoSignalReason::SpecMismatch);
            }
            return PrefixIndexOutcome::NoSignal(NoSignalReason::Empty);
        }

        PrefixIndexOutcome::Matched {
            matches: response
                .matches
                .into_iter()
                .map(|m| PrefixMatch {
                    worker_id: m.worker_id,
                    address: m.address,
                    dp_rank: m.dp_rank,
                    matched_prefix_blocks: m.matched_prefix_blocks,
                    tiers: m
                        .tiers
                        .into_iter()
                        .filter_map(|tier| TierType::try_from(tier).ok())
                        .collect(),
                })
                .collect(),
            best_prefix_blocks: response.best_prefix_blocks,
        }
    }
}

/// Builds the wire request, capping the query at the protocol limit rather than
/// letting the server reject it. A truncated query answers a shorter question —
/// which is what `max_blocks` already exists to express — while a rejected one
/// answers none.
fn build_request(query: &PrefixQuery) -> MatchExternalKvPrefixRequest {
    MatchExternalKvPrefixRequest {
        hashes: query
            .hashes
            .iter()
            .copied()
            .take(MAX_HASHES_PER_REQUEST)
            .map(hash_to_wire)
            .collect(),
        hash_spec: Some(query.spec.clone()),
        max_blocks: query.max_blocks,
        top_k: query.top_k,
        count_as_hit: query.count_as_hit,
    }
}

/// Whether a fallback reason describes the index or the request that produced
/// it. Only an index-level problem opens the breaker: counting a rejected query
/// would let one malformed request take cache-aware routing away from every
/// other request for a cooldown.
fn is_index_outage(reason: NoSignalReason) -> bool {
    matches!(
        reason,
        NoSignalReason::Unavailable | NoSignalReason::Timeout | NoSignalReason::Unsupported
    )
}

fn classify_status(status: &Status) -> NoSignalReason {
    match status.code() {
        Code::DeadlineExceeded | Code::Cancelled => NoSignalReason::Timeout,
        Code::Unavailable | Code::Internal | Code::Unknown | Code::Aborted => {
            NoSignalReason::Unavailable
        }
        Code::Unimplemented => NoSignalReason::Unsupported,
        // Something about this request, not about the index.
        _ => NoSignalReason::Rejected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> HashSpec {
        HashSpec::sglang(64, false)
    }

    #[test]
    fn hash_round_trips_through_the_wire_form() {
        for hash in [0_i64, 1, -1, i64::MAX, i64::MIN, -1905904552702706914] {
            assert_eq!(hash_from_wire(&hash_to_wire(hash)), Some(hash));
        }
    }

    #[test]
    fn non_numeric_wire_hash_is_not_an_sglang_hash() {
        assert_eq!(hash_from_wire("prefix"), None);
        assert_eq!(hash_from_wire(""), None);
        // Above i64::MAX: a publisher using unsigned encoding, which the server
        // normalizes on ingest. It is not a value a router can have produced.
        assert_eq!(hash_from_wire("18446744073709551615"), None);
    }

    #[test]
    fn sglang_spec_selects_the_hashing_mode() {
        assert_eq!(
            HashSpec::sglang(64, false).algo,
            HashAlgorithm::HashAlgoSha256ChainUnigram as i32
        );
        assert_eq!(
            HashSpec::sglang(64, true).algo,
            HashAlgorithm::HashAlgoSha256ChainBigram as i32
        );
        assert_eq!(HashSpec::sglang(64, true).block_size, 64);
    }

    #[test]
    fn best_matches_returns_only_the_tied_longest() {
        let outcome = PrefixIndexOutcome::Matched {
            matches: vec![
                PrefixMatch {
                    worker_id: "a".into(),
                    address: "http://a".into(),
                    dp_rank: 0,
                    matched_prefix_blocks: 4,
                    tiers: vec![TierType::TierHbm],
                },
                PrefixMatch {
                    worker_id: "b".into(),
                    address: "http://b".into(),
                    dp_rank: 0,
                    matched_prefix_blocks: 4,
                    tiers: vec![TierType::TierHbm],
                },
                PrefixMatch {
                    worker_id: "c".into(),
                    address: "http://c".into(),
                    dp_rank: 0,
                    matched_prefix_blocks: 1,
                    tiers: vec![TierType::TierHbm],
                },
            ],
            best_prefix_blocks: 4,
        };
        let best: Vec<&str> = outcome
            .best_matches()
            .iter()
            .map(|m| m.worker_id.as_str())
            .collect();
        assert_eq!(best, vec!["a", "b"]);
        assert_eq!(outcome.best_prefix_blocks(), 4);
    }

    #[test]
    fn no_signal_has_no_matches_and_no_prefix() {
        let outcome = PrefixIndexOutcome::NoSignal(NoSignalReason::Unavailable);
        assert!(outcome.best_matches().is_empty());
        assert_eq!(outcome.best_prefix_blocks(), 0);
    }

    #[test]
    fn status_codes_map_to_fallback_reasons() {
        assert_eq!(
            classify_status(&Status::deadline_exceeded("slow")),
            NoSignalReason::Timeout
        );
        assert_eq!(
            classify_status(&Status::unavailable("down")),
            NoSignalReason::Unavailable
        );
        // A server too old to know the RPC describes the index, not the query.
        assert_eq!(
            classify_status(&Status::unimplemented("old server")),
            NoSignalReason::Unsupported
        );
        assert_eq!(
            classify_status(&Status::invalid_argument("bad query")),
            NoSignalReason::Rejected
        );
        assert_eq!(
            classify_status(&Status::resource_exhausted("too many hashes")),
            NoSignalReason::Rejected
        );
    }

    /// The breaker exists to skip a failing index, so only index-level trouble
    /// may open it. A rejected query is about that query; counting it would let
    /// one malformed request take cache-aware routing away from every other
    /// request for a whole cooldown.
    #[test]
    fn only_index_level_trouble_opens_the_breaker() {
        assert!(is_index_outage(NoSignalReason::Unavailable));
        assert!(is_index_outage(NoSignalReason::Timeout));
        assert!(is_index_outage(NoSignalReason::Unsupported));
        assert!(!is_index_outage(NoSignalReason::Rejected));
        assert!(!is_index_outage(NoSignalReason::Empty));
        assert!(!is_index_outage(NoSignalReason::SpecMismatch));
    }

    /// The breaker's deadline is an interval, so it must be measured on a clock
    /// that cannot move backwards. On a wall clock, a backwards step larger
    /// than the cooldown would wedge the breaker open permanently: closing it
    /// takes a successful query, and no query is issued while it is open.
    #[test]
    fn breaker_deadline_is_monotonic() {
        let breaker = Breaker::new(1, Duration::from_millis(1));
        breaker.record_failure();
        assert!(breaker.is_open());
        std::thread::sleep(Duration::from_millis(5));
        assert!(
            !breaker.is_open(),
            "the breaker must reopen for business once the cooldown elapses"
        );
    }

    /// An oversized query is capped instead of sent and rejected: a truncated
    /// query answers a shorter question, a rejected one answers none.
    #[test]
    fn oversized_query_is_capped_rather_than_rejected() {
        let hashes: Vec<i64> = (0..(MAX_HASHES_PER_REQUEST as i64 + 10)).collect();
        let request = build_request(&PrefixQuery::new(hashes, spec()));
        assert_eq!(request.hashes.len(), MAX_HASHES_PER_REQUEST);
        assert_eq!(
            request.hashes.first().map(String::as_str),
            Some("0"),
            "the cap must drop the tail, not the prefix the answer depends on"
        );
    }

    #[test]
    fn request_carries_the_query_knobs() {
        let request = build_request(&PrefixQuery {
            max_blocks: 64,
            top_k: 4,
            count_as_hit: true,
            ..PrefixQuery::new(vec![1, -2], spec())
        });
        assert_eq!(request.hashes, vec!["1".to_string(), "-2".to_string()]);
        assert_eq!(request.max_blocks, 64);
        assert_eq!(request.top_k, 4);
        assert!(request.count_as_hit);
        assert_eq!(request.hash_spec, Some(spec()));
    }

    #[test]
    fn breaker_opens_after_consecutive_failures_and_recovers_on_success() {
        let breaker = Breaker::new(3, Duration::from_secs(60));
        assert!(!breaker.is_open());
        breaker.record_failure();
        breaker.record_failure();
        assert!(
            !breaker.is_open(),
            "must tolerate a blip below the threshold"
        );
        breaker.record_failure();
        assert!(breaker.is_open());
        breaker.record_success();
        assert!(!breaker.is_open(), "one good answer closes the breaker");
    }

    #[test]
    fn log_throttle_admits_one_message_per_interval() {
        let throttle = LogThrottle::new(Duration::from_millis(50));
        assert!(throttle.allow(), "the first report must always get through");
        assert!(!throttle.allow());
        assert!(!throttle.allow());
        std::thread::sleep(Duration::from_millis(60));
        assert!(
            throttle.allow(),
            "a condition still true after the interval must report again"
        );
    }

    #[test]
    fn breaker_forgets_isolated_failures() {
        let breaker = Breaker::new(3, Duration::from_secs(60));
        for _ in 0..10 {
            breaker.record_failure();
            breaker.record_failure();
            breaker.record_success();
        }
        assert!(
            !breaker.is_open(),
            "failures separated by successes must not accumulate into an open breaker"
        );
    }

    /// An empty query never reaches the wire: there is nothing to match, and a
    /// server would reject it as invalid.
    #[tokio::test]
    async fn empty_query_short_circuits_without_a_server() {
        let index = KvIndexerPrefixIndex::connect_lazy(PrefixIndexConfig::new(
            "http://127.0.0.1:1".to_string(),
        ))
        .expect("lazy connect never dials");
        let outcome = index
            .match_prefix(&PrefixQuery::new(Vec::new(), spec()))
            .await;
        assert_eq!(outcome, PrefixIndexOutcome::NoSignal(NoSignalReason::Empty));
    }

    /// The contract that makes the index safe to depend on: an unreachable
    /// server yields a fallback signal, not an error the caller has to handle.
    #[tokio::test]
    async fn unreachable_index_yields_no_signal() {
        let index = KvIndexerPrefixIndex::connect_lazy(PrefixIndexConfig::new(
            "http://127.0.0.1:1".to_string(),
        ))
        .expect("lazy connect never dials");
        let outcome = index
            .match_prefix(&PrefixQuery::new(vec![1, 2, 3], spec()))
            .await;
        assert!(
            matches!(outcome, PrefixIndexOutcome::NoSignal(_)),
            "an unreachable index must degrade, not fail: got {outcome:?}"
        );
        assert!(outcome.best_matches().is_empty());
    }

    /// Repeated outages open the breaker, and an open breaker answers without
    /// dialing — which is what keeps an index outage off the routing budget.
    #[tokio::test]
    async fn repeated_failures_open_the_breaker() {
        let mut config = PrefixIndexConfig::new("http://127.0.0.1:1".to_string());
        config.breaker_threshold = 2;
        config.query_timeout = Duration::from_millis(50);
        let index = KvIndexerPrefixIndex::connect_lazy(config).expect("lazy connect never dials");
        let query = PrefixQuery::new(vec![1], spec());
        for _ in 0..2 {
            let _ = index.match_prefix(&query).await;
        }
        assert!(
            index.breaker.is_open(),
            "two failed queries at threshold 2 must open the breaker"
        );
    }

    /// The short-circuit is what makes an open breaker worth having. Measured
    /// against a query budget large enough that reaching the wire would be
    /// unmistakable, rather than against a wall-clock figure that a loaded
    /// machine could trip on its own.
    #[tokio::test]
    async fn an_open_breaker_answers_without_reaching_the_wire() {
        let mut config = PrefixIndexConfig::new("http://127.0.0.1:1".to_string());
        config.query_timeout = Duration::from_secs(30);
        let index = KvIndexerPrefixIndex::connect_lazy(config).expect("lazy connect never dials");
        index.breaker.record_failure();
        index.breaker.record_failure();
        index.breaker.record_failure();
        index.breaker.record_failure();
        index.breaker.record_failure();
        assert!(index.breaker.is_open());

        let started = std::time::Instant::now();
        let outcome = index.match_prefix(&PrefixQuery::new(vec![1], spec())).await;
        assert_eq!(
            outcome,
            PrefixIndexOutcome::NoSignal(NoSignalReason::Unavailable)
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "an open breaker must not pay the query budget, took {:?}",
            started.elapsed()
        );
    }
}
