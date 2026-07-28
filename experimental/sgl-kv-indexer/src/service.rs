// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashMap};

use tonic::{Request, Response, Status};

use crate::pb::kv_indexer_server::KvIndexer;
use crate::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, ExternalKvAction,
    ExternalKvActionType, ExternalKvPrefixMatch, GetExternalKvHitCountsRequest,
    GetExternalKvHitCountsResponse, HashAlgorithm, HashSpec, MatchExternalKvPrefixRequest,
    MatchExternalKvPrefixResponse, MatchExternalKvRequest, MatchExternalKvResponse,
};

/// Protocol-level resource bounds. The Redis backend additionally chunks its
/// fan-out, but rejecting oversized requests here prevents any backend from
/// allocating or scheduling work proportional to an unbounded repeated field.
///
/// Public so a client can keep a query inside the limit rather than discover it
/// as a rejection.
pub const MAX_HASHES_PER_REQUEST: usize = 16_384;
const MAX_ACTIONS_PER_BATCH: usize = 256;
/// Highest valid [`crate::pb::TierType`] discriminant. Kept next to the
/// validator so adding a tier to the proto is a single-line change here and in
/// the storage bitmask decoder.
pub(crate) const MAX_TIER: i32 = 3;

/// Storage backend for the indexer. Deliberately narrow: every mutation flows
/// through `apply_external_kv_batch`, preserving one ordered write path.
///
/// Async because real backends (e.g. Redis) do network IO; the trait is made
/// dyn-safe via `#[tonic::async_trait]` so the server can select a backend at
/// runtime and hold it as `Arc<dyn KvIndexerBackend>`.
#[tonic::async_trait]
pub trait KvIndexerBackend: Send + Sync + 'static {
    /// Applies a whole SGLang KVEventBatch. The actions are pre-validated and
    /// must be applied in order. `seq` is a per-worker monotonic idempotency
    /// key: a durable backend stores the last applied seq per worker, skips a
    /// batch whose seq was already applied (a duplicate), and reports its
    /// durable position back in [`ApplyExternalKvBatchResponse::last_applied_seq`].
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status>;

    async fn match_external_kv(
        &self,
        request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status>;

    /// Prefix-aware routing query. The default implementation derives the
    /// answer from [`Self::match_external_kv`], which makes it the executable
    /// definition of the prefix semantics: a backend that overrides this for
    /// efficiency must agree with it hash for hash.
    ///
    /// Two things the default cannot do, because the placement-set API does not
    /// expose them: it never checks `hash_spec` (it reports zero mismatches),
    /// and it ignores `count_as_hit`. A backend that supports either overrides
    /// this method.
    ///
    /// `max_blocks` is already applied: [`KvIndexerService`] truncates the
    /// hashes before dispatch so every backend measures `queried_blocks`
    /// against the same input. Implementations read `request.hashes` as given.
    async fn match_external_kv_prefix(
        &self,
        request: MatchExternalKvPrefixRequest,
    ) -> Result<MatchExternalKvPrefixResponse, Status> {
        let queried_blocks = request.hashes.len() as u32;
        let placements = self
            .match_external_kv(MatchExternalKvRequest {
                hashes: request.hashes.clone(),
                count_as_hit: false,
            })
            .await?;
        let matches = fold_prefix_matches(&request.hashes, &placements);
        Ok(prefix_response(matches, request.top_k, queried_blocks, 0))
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status>;

    /// Readiness probe: `true` when the backend can serve requests. Stateless
    /// backends are always ready; a durable backend (Redis) overrides this to
    /// reflect store connectivity so the gRPC health service can report it.
    async fn health(&self) -> bool {
        true
    }
}

/// Blanket impl so the server can hold the selected backend as
/// `Arc<dyn KvIndexerBackend>` and still satisfy `KvIndexerService<B>`.
#[tonic::async_trait]
impl KvIndexerBackend for std::sync::Arc<dyn KvIndexerBackend> {
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        (**self).apply_external_kv_batch(request).await
    }

    async fn match_external_kv(
        &self,
        request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        (**self).match_external_kv(request).await
    }

    async fn match_external_kv_prefix(
        &self,
        request: MatchExternalKvPrefixRequest,
    ) -> Result<MatchExternalKvPrefixResponse, Status> {
        (**self).match_external_kv_prefix(request).await
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        (**self).get_external_kv_hit_counts(request).await
    }

    async fn health(&self) -> bool {
        (**self).health().await
    }
}

#[derive(Debug)]
pub struct KvIndexerService<B> {
    backend: B,
}

impl<B> KvIndexerService<B>
where
    B: KvIndexerBackend,
{
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

#[tonic::async_trait]
impl<B> KvIndexer for KvIndexerService<B>
where
    B: KvIndexerBackend,
{
    async fn match_external_kv(
        &self,
        request: Request<MatchExternalKvRequest>,
    ) -> Result<Response<MatchExternalKvResponse>, Status> {
        let mut request = request.into_inner();
        validate_hashes(&request.hashes)?;
        normalize_hashes(&mut request.hashes);
        let response = self.backend.match_external_kv(request).await?;
        Ok(Response::new(response))
    }

    async fn match_external_kv_prefix(
        &self,
        request: Request<MatchExternalKvPrefixRequest>,
    ) -> Result<Response<MatchExternalKvPrefixResponse>, Status> {
        let mut request = request.into_inner();
        validate_hashes(&request.hashes)?;
        validate_hash_spec(request.hash_spec.as_ref())?;
        normalize_hashes(&mut request.hashes);
        // Applying `max_blocks` here rather than in each backend keeps
        // `queried_blocks` measured against the same input everywhere.
        if request.max_blocks > 0 {
            request.hashes.truncate(request.max_blocks as usize);
        }
        let response = self.backend.match_external_kv_prefix(request).await?;
        Ok(Response::new(response))
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: Request<GetExternalKvHitCountsRequest>,
    ) -> Result<Response<GetExternalKvHitCountsResponse>, Status> {
        let mut request = request.into_inner();
        validate_hashes(&request.hashes)?;
        // Counts are keyed by the same normalized hash the write path stored,
        // so skipping this here would make a block's counter unreadable by the
        // encoding it was reported under.
        normalize_hashes(&mut request.hashes);
        let response = self.backend.get_external_kv_hit_counts(request).await?;
        Ok(Response::new(response))
    }

    async fn apply_external_kv_batch(
        &self,
        request: Request<ApplyExternalKvBatchRequest>,
    ) -> Result<Response<ApplyExternalKvBatchResponse>, Status> {
        let mut request = request.into_inner();
        validate_worker_id(&request.worker_id)?;
        validate_actions(&request.actions)?;
        validate_hash_spec(request.hash_spec.as_ref())?;
        for action in &mut request.actions {
            normalize_hashes(&mut action.hashes);
        }
        let response = self.backend.apply_external_kv_batch(request).await?;
        Ok(Response::new(response))
    }
}

/// Rewrites integer-valued hashes into the canonical signed 64-bit decimal form
/// SGLang publishes, so a publisher that encodes the same block as an unsigned
/// value above `i64::MAX` lands on the same storage key as one that encodes it
/// as the equivalent negative number.
///
/// Non-integer hashes pass through untouched: the wire type is an opaque string
/// and nothing in the protocol requires a publisher to use SGLang's numeric
/// encoding.
fn normalize_hash(hash: &str) -> Option<String> {
    // Re-emit rather than accept any form that merely parses: `007` and `+7`
    // are valid `i64` inputs, so a check for parseability alone would leave two
    // spellings of one block on two storage keys.
    if let Ok(value) = hash.parse::<i64>() {
        let canonical = value.to_string();
        return (canonical != hash).then_some(canonical);
    }
    hash.parse::<u64>().ok().map(|v| (v as i64).to_string())
}

fn normalize_hashes(hashes: &mut [String]) {
    for hash in hashes {
        if let Some(canonical) = normalize_hash(hash) {
            *hash = canonical;
        }
    }
}

/// Folds a placement-set match response into per-worker contiguous prefix
/// lengths, longest first.
///
/// A worker's prefix stops at the first requested hash it does not hold, which
/// is stricter than "holds the deepest matched block": it can only understate
/// what a worker is able to reuse, never overstate it. Workers with an empty
/// address are dropped because a router cannot forward to them.
pub(crate) fn fold_prefix_matches(
    hashes: &[String],
    placements: &MatchExternalKvResponse,
) -> Vec<ExternalKvPrefixMatch> {
    let mut matches = Vec::with_capacity(placements.matches.len());
    for node in &placements.matches {
        if node.address.is_empty() {
            continue;
        }
        let mut tiers_by_hash: HashMap<&str, Vec<i32>> = HashMap::new();
        for held in &node.hashes_by_tier {
            for hash in &held.hashes {
                tiers_by_hash
                    .entry(hash.as_str())
                    .or_default()
                    .push(held.tier);
            }
        }

        let mut tiers = BTreeSet::new();
        let mut matched = 0usize;
        for hash in hashes {
            let Some(held_tiers) = tiers_by_hash.get(hash.as_str()) else {
                break;
            };
            tiers.extend(held_tiers.iter().copied());
            matched += 1;
        }
        if matched == 0 {
            continue;
        }
        matches.push(ExternalKvPrefixMatch {
            worker_id: node.worker_id.clone(),
            address: node.address.clone(),
            dp_rank: node.dp_rank,
            matched_prefix_blocks: matched as u32,
            tiers: tiers.into_iter().collect(),
        });
    }
    matches
}

/// Orders matches longest-prefix first and assembles the response. Ties break
/// on `worker_id` so the ordering is stable across backends and runs, which is
/// what lets the contract tests compare two implementations element by element.
pub(crate) fn prefix_response(
    mut matches: Vec<ExternalKvPrefixMatch>,
    top_k: u32,
    queried_blocks: u32,
    spec_mismatched_workers: u32,
) -> MatchExternalKvPrefixResponse {
    matches.sort_by(|a, b| {
        b.matched_prefix_blocks
            .cmp(&a.matched_prefix_blocks)
            .then_with(|| a.worker_id.cmp(&b.worker_id))
    });
    let best_prefix_blocks = matches
        .first()
        .map(|m| m.matched_prefix_blocks)
        .unwrap_or(0);
    if top_k > 0 {
        matches.truncate(top_k as usize);
    }
    MatchExternalKvPrefixResponse {
        matches,
        best_prefix_blocks,
        queried_blocks,
        spec_mismatched_workers,
    }
}

fn validate_worker_id(worker_id: &str) -> Result<(), Status> {
    if worker_id.is_empty() {
        return Err(Status::invalid_argument("worker_id must not be empty"));
    }
    Ok(())
}

/// A spec is either absent (no checking) or fully specified. A `block_size` of
/// zero with a declared algorithm is a caller bug that would otherwise disable
/// checking silently, which is the exact failure the spec exists to prevent.
fn validate_hash_spec(spec: Option<&HashSpec>) -> Result<(), Status> {
    let Some(spec) = spec else {
        return Ok(());
    };
    if HashAlgorithm::try_from(spec.algo).is_err() {
        return Err(Status::invalid_argument("hash_spec.algo is not supported"));
    }
    let declared =
        spec.algo != HashAlgorithm::HashAlgoUnspecified as i32 || !spec.namespace.is_empty();
    if spec.block_size == 0 && declared {
        return Err(Status::invalid_argument(
            "hash_spec.block_size must be set when a hash algorithm or namespace is declared",
        ));
    }
    Ok(())
}

fn validate_hashes(hashes: &[String]) -> Result<(), Status> {
    if hashes.is_empty() {
        return Err(Status::invalid_argument("hashes must not be empty"));
    }
    if hashes.len() > MAX_HASHES_PER_REQUEST {
        return Err(Status::resource_exhausted(format!(
            "request contains {} hashes; maximum is {MAX_HASHES_PER_REQUEST}",
            hashes.len()
        )));
    }
    if hashes.iter().any(|hash| hash.is_empty()) {
        return Err(Status::invalid_argument(
            "hashes must not contain empty values",
        ));
    }
    Ok(())
}

fn validate_tier(tier: i32) -> Result<(), Status> {
    match tier {
        1..=MAX_TIER => Ok(()),
        0 => Err(Status::invalid_argument("tier must not be TIER_UNKNOWN")),
        _ => Err(Status::invalid_argument("tier is not supported")),
    }
}

fn validate_actions(actions: &[ExternalKvAction]) -> Result<(), Status> {
    // An empty actions list is a valid heartbeat: it carries no mutation and
    // simply refreshes the worker's liveness on the server. Non-empty batches
    // still have every action validated below.
    if actions.len() > MAX_ACTIONS_PER_BATCH {
        return Err(Status::resource_exhausted(format!(
            "batch contains {} actions; maximum is {MAX_ACTIONS_PER_BATCH}",
            actions.len()
        )));
    }
    let total_hashes: usize = actions.iter().map(|action| action.hashes.len()).sum();
    if total_hashes > MAX_HASHES_PER_REQUEST {
        return Err(Status::resource_exhausted(format!(
            "batch contains {total_hashes} hashes; maximum is {MAX_HASHES_PER_REQUEST}"
        )));
    }
    for action in actions {
        validate_tier(action.tier)?;
        match ExternalKvActionType::try_from(action.r#type) {
            Ok(ExternalKvActionType::ActionReport) | Ok(ExternalKvActionType::ActionRevoke) => {
                validate_hashes(&action.hashes)?;
            }
            // CLEAR_ALL_AT_TIER carries only a tier; hashes are ignored.
            Ok(ExternalKvActionType::ActionClearAllAtTier) => {}
            Ok(ExternalKvActionType::ActionUnknown) | Err(_) => {
                return Err(Status::invalid_argument("action type is not supported"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hbm() -> i32 {
        crate::pb::TierType::TierHbm as i32
    }

    fn action(r#type: ExternalKvActionType, tier: i32, hashes: &[&str]) -> ExternalKvAction {
        ExternalKvAction {
            r#type: r#type as i32,
            tier,
            hashes: hashes.iter().map(|h| h.to_string()).collect(),
        }
    }

    #[test]
    fn validate_actions_allows_empty_as_heartbeat() {
        // An empty batch is a liveness heartbeat, not an error.
        assert!(validate_actions(&[]).is_ok());
    }

    #[test]
    fn validate_actions_rejects_unknown_type() {
        let actions = [action(ExternalKvActionType::ActionUnknown, hbm(), &["1"])];
        assert!(validate_actions(&actions).is_err());
    }

    #[test]
    fn validate_actions_rejects_bad_tier() {
        let actions = [action(ExternalKvActionType::ActionReport, 0, &["1"])];
        assert!(validate_actions(&actions).is_err());
    }

    #[test]
    fn validate_actions_requires_hashes_for_report_and_revoke() {
        assert!(
            validate_actions(&[action(ExternalKvActionType::ActionReport, hbm(), &[])]).is_err()
        );
        assert!(
            validate_actions(&[action(ExternalKvActionType::ActionRevoke, hbm(), &[])]).is_err()
        );
    }

    #[test]
    fn validate_actions_allows_empty_hashes_for_clear_all_at_tier() {
        let actions = [action(
            ExternalKvActionType::ActionClearAllAtTier,
            hbm(),
            &[],
        )];
        assert!(validate_actions(&actions).is_ok());
    }

    #[test]
    fn validate_hashes_rejects_oversized_query() {
        let hashes = vec!["1".to_string(); MAX_HASHES_PER_REQUEST + 1];
        let error = validate_hashes(&hashes).unwrap_err();
        assert_eq!(error.code(), tonic::Code::ResourceExhausted);
    }

    #[test]
    fn validate_actions_rejects_oversized_batch() {
        let hashes = vec!["1"; MAX_HASHES_PER_REQUEST / 2 + 1];
        let actions = [
            action(ExternalKvActionType::ActionReport, hbm(), &hashes),
            action(ExternalKvActionType::ActionReport, hbm(), &hashes),
        ];
        let error = validate_actions(&actions).unwrap_err();
        assert_eq!(error.code(), tonic::Code::ResourceExhausted);
    }

    #[test]
    fn validate_actions_rejects_too_many_actions() {
        let clear = action(ExternalKvActionType::ActionClearAllAtTier, hbm(), &[]);
        let actions = vec![clear; MAX_ACTIONS_PER_BATCH + 1];
        let error = validate_actions(&actions).unwrap_err();
        assert_eq!(error.code(), tonic::Code::ResourceExhausted);
    }

    #[test]
    fn validate_worker_id_rejects_empty_value() {
        assert!(validate_worker_id("").is_err());
        assert!(validate_worker_id("worker-1").is_ok());
    }

    /// Every spelling of one 64-bit value must land on one storage key. The
    /// unsigned form is what another publisher might emit; the padded and signed
    /// forms merely parse, which is not the same as being canonical.
    #[test]
    fn normalize_hash_collapses_every_spelling_of_a_value() {
        for spelling in ["-1", "18446744073709551615"] {
            let canonical = normalize_hash(spelling).unwrap_or_else(|| spelling.to_string());
            assert_eq!(canonical, "-1", "{spelling} must canonicalize to -1");
        }
        for spelling in ["7", "007", "+7"] {
            let canonical = normalize_hash(spelling).unwrap_or_else(|| spelling.to_string());
            assert_eq!(canonical, "7", "{spelling} must canonicalize to 7");
        }
    }

    /// A hash that is already canonical is left alone, so normalization never
    /// rewrites the common case.
    #[test]
    fn normalize_hash_leaves_canonical_and_opaque_hashes_alone() {
        assert_eq!(normalize_hash("0"), None);
        assert_eq!(normalize_hash("-1905904552702706914"), None);
        assert_eq!(normalize_hash(&i64::MIN.to_string()), None);
        // Not SGLang's numeric encoding at all; the wire type is opaque.
        assert_eq!(normalize_hash("prefix"), None);
        assert_eq!(normalize_hash(""), None);
    }
}
