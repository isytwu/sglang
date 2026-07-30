// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use tonic::{Request, Response, Status};

use crate::pb::kv_indexer_server::KvIndexer;
use crate::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, ExternalKvAction,
    ExternalKvActionType, ExternalKvPrefixMatch, GetExternalKvHitCountsRequest,
    GetExternalKvHitCountsResponse, MatchExternalKvPrefixRequest, MatchExternalKvPrefixResponse,
    MatchExternalKvRequest, MatchExternalKvResponse, TierType, WorkerCacheSpec,
};

/// Protocol-level resource bounds. The Redis backend additionally chunks its
/// fan-out, but rejecting oversized requests here prevents any backend from
/// allocating or scheduling work proportional to an unbounded repeated field.
const MAX_HASHES_PER_REQUEST: usize = 16_384;
const MAX_ACTIONS_PER_BATCH: usize = 256;

/// Storage backend for the indexer. Deliberately narrow: every mutation flows
/// through `apply_external_kv_batch`, preserving one ordered write path.
///
/// Async because real backends (e.g. Redis) do network IO; the trait is made
/// dyn-safe via `#[tonic::async_trait]` so the server can select a backend at
/// runtime and hold it as `Arc<dyn KvIndexerBackend>`.
#[tonic::async_trait]
pub trait KvIndexerBackend: Send + Sync + 'static {
    /// Applies a whole SGLang KVEventBatch. The actions are pre-validated and
    /// must be applied in order. Applies are unconditional: the request `seq` is
    /// informational only and a redelivered batch is applied again.
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status>;

    async fn match_external_kv(
        &self,
        request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status>;

    /// Collects the per-worker, per-block component placement needed to compute a
    /// prefix, aligned with `hashes`.
    ///
    /// The default implementation is component-blind: it composes
    /// `match_external_kv` and treats every held block as a legacy whole-block
    /// placement (no components, no size, no spec). Component-aware backends
    /// override it to attach each worker's `WorkerCacheSpec` and the resident
    /// component set per `(hash, tier)`.
    async fn collect_worker_prefix_inputs(
        &self,
        hashes: &[String],
    ) -> Result<Vec<WorkerPrefixInput>, Status> {
        let matched = self
            .match_external_kv(MatchExternalKvRequest {
                hashes: hashes.to_vec(),
                count_as_hit: false,
            })
            .await?;
        Ok(legacy_inputs_from_match(hashes, &matched))
    }

    /// Answers, per worker, the longest reusable request prefix it holds.
    ///
    /// This default implementation *is* the written definition of the prefix
    /// semantics: it collects each worker's component placement (see
    /// [`KvIndexerBackend::collect_worker_prefix_inputs`]) and runs the shared
    /// rule engine ([`compute_prefix_response`]), so any backend that overrides
    /// it for performance must stay field-for-field identical (except
    /// `blocks_read`, which is observability, not semantics).
    ///
    /// The result is a safe lower bound of what the worker can actually reuse: a
    /// component-aware match applies each required component's rule (contiguous /
    /// trailing-window / exact-boundary), so the indexer can only under-report,
    /// never over-report, whenever its index state is accurate.
    async fn match_external_kv_prefix(
        &self,
        request: MatchExternalKvPrefixRequest,
    ) -> Result<MatchExternalKvPrefixResponse, Status> {
        let limit = prefix_limit(request.hashes.len(), request.max_blocks);
        let hashes: Vec<String> = request.hashes.into_iter().take(limit).collect();
        if hashes.is_empty() {
            return Ok(MatchExternalKvPrefixResponse::default());
        }
        // The default path reads placement for every considered block.
        let inputs = self.collect_worker_prefix_inputs(&hashes).await?;
        Ok(compute_prefix_response(&inputs, hashes.len() as u32))
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status>;
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

    async fn collect_worker_prefix_inputs(
        &self,
        hashes: &[String],
    ) -> Result<Vec<WorkerPrefixInput>, Status> {
        (**self).collect_worker_prefix_inputs(hashes).await
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
        let request = request.into_inner();
        validate_hashes(&request.hashes)?;
        let response = self.backend.match_external_kv(request).await?;
        Ok(Response::new(response))
    }

    async fn match_external_kv_prefix(
        &self,
        request: Request<MatchExternalKvPrefixRequest>,
    ) -> Result<Response<MatchExternalKvPrefixResponse>, Status> {
        let request = request.into_inner();
        validate_hashes(&request.hashes)?;
        let response = self.backend.match_external_kv_prefix(request).await?;
        Ok(Response::new(response))
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: Request<GetExternalKvHitCountsRequest>,
    ) -> Result<Response<GetExternalKvHitCountsResponse>, Status> {
        let request = request.into_inner();
        validate_hashes(&request.hashes)?;
        let response = self.backend.get_external_kv_hit_counts(request).await?;
        Ok(Response::new(response))
    }

    async fn apply_external_kv_batch(
        &self,
        request: Request<ApplyExternalKvBatchRequest>,
    ) -> Result<Response<ApplyExternalKvBatchResponse>, Status> {
        let request = request.into_inner();
        validate_worker_id(&request.worker_id)?;
        validate_actions(&request.actions)?;
        let response = self.backend.apply_external_kv_batch(request).await?;
        Ok(Response::new(response))
    }
}

fn validate_worker_id(worker_id: &str) -> Result<(), Status> {
    if worker_id.is_empty() {
        return Err(Status::invalid_argument("worker_id must not be empty"));
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
        1..=3 => Ok(()),
        0 => Err(Status::invalid_argument("tier must not be TIER_UNKNOWN")),
        _ => Err(Status::invalid_argument("tier is not supported")),
    }
}

fn validate_actions(actions: &[ExternalKvAction]) -> Result<(), Status> {
    // An empty actions list is accepted and applied as a no-op; it only refreshes
    // the worker's recorded address. Non-empty batches still have every action
    // validated below.
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

/// Number of leading blocks to consider for a prefix query: bounded by the
/// request length and, when the caller set one, by `max_blocks` (0 disables the
/// caller ceiling). Backends may impose their own additional scan cap.
pub(crate) fn prefix_limit(len: usize, max_blocks: u32) -> usize {
    if max_blocks == 0 {
        len
    } else {
        len.min(max_blocks as usize)
    }
}

/// KV component bits. The set and their match rules are fixed: a component's
/// rule is a property of its type (FULL is a path component, SWA a trailing
/// window, MAMBA a boundary checkpoint), so the indexer applies fixed semantics
/// rather than a per-worker rule binding.
pub const COMPONENT_FULL: u32 = 1 << 0;
pub const COMPONENT_SWA: u32 = 1 << 1;
pub const COMPONENT_MAMBA: u32 = 1 << 2;

/// Maps an on-wire component label to its bit; `None` for a label this build
/// does not model (ignored, so an unknown future component never counts).
pub fn component_bit(name: &str) -> Option<u32> {
    match name {
        "full" => Some(COMPONENT_FULL),
        "swa" => Some(COMPONENT_SWA),
        "mamba" => Some(COMPONENT_MAMBA),
        _ => None,
    }
}

/// Tiers the indexer treats as servable when deciding a component is reusable,
/// as a bitmask of `1 << TierType`. V1: HBM (device) and DRAM (load-backable
/// host); SSD is not counted.
const SERVABLE_TIER_MASK: u32 =
    (1 << (TierType::TierHbm as u32)) | (1 << (TierType::TierDram as u32));

/// Highest `WorkerCacheSpec.version` this build knows how to interpret. A spec
/// from the future (higher version) is treated as unusable and fails closed,
/// rather than being misread against the current rule set. Version 0 is the
/// proto default (unversioned) and is accepted as the current version.
const SUPPORTED_SPEC_VERSION: u32 = 1;

/// Whether `tier` is set in a `1 << TierType` bitmask.
fn tier_in_mask(mask: u32, tier: i32) -> bool {
    tier >= 0 && mask & (1u32 << tier) != 0
}

/// The resident KV components for one block at one worker, per tier, plus the
/// block's token count (used to accumulate SWA trailing windows).
#[derive(Debug, Clone)]
pub struct BlockComponents {
    pub token_count: u32,
    /// `(tier, component bitmask)` for every tier at which the worker holds the
    /// block. A legacy whole-block placement carries mask `0` (held, no detail).
    pub tier_masks: Vec<(i32, u32)>,
}

/// Everything the prefix rule engine needs about one candidate worker: its
/// routing identity, its (optional) component spec, and, aligned with the query
/// hashes, the block placement (`None` for a block the worker does not hold).
#[derive(Debug, Clone)]
pub struct WorkerPrefixInput {
    pub worker_id: String,
    pub address: String,
    pub spec: Option<WorkerCacheSpec>,
    pub blocks: Vec<Option<BlockComponents>>,
}

/// Builds component-blind (legacy) prefix inputs from a `MatchExternalKv` result.
/// Each block the worker holds becomes a whole-block placement (mask `0`) with no
/// size and no spec — reproducing the pre-component behaviour.
pub(crate) fn legacy_inputs_from_match(
    hashes: &[String],
    matched: &MatchExternalKvResponse,
) -> Vec<WorkerPrefixInput> {
    matched
        .matches
        .iter()
        .map(|node| {
            let mut tiers_by_hash: HashMap<&str, Vec<i32>> = HashMap::new();
            for tier in &node.hashes_by_tier {
                for hash in &tier.hashes {
                    tiers_by_hash
                        .entry(hash.as_str())
                        .or_default()
                        .push(tier.tier);
                }
            }
            let blocks = hashes
                .iter()
                .map(|hash| {
                    tiers_by_hash
                        .get(hash.as_str())
                        .map(|tiers| BlockComponents {
                            token_count: 0,
                            tier_masks: tiers.iter().map(|tier| (*tier, 0u32)).collect(),
                        })
                })
                .collect();
            WorkerPrefixInput {
                worker_id: node.worker_id.clone(),
                address: node.address.clone(),
                spec: None,
                blocks,
            }
        })
        .collect()
}

/// Runs the component-aware rule engine over each worker and assembles the
/// response. This is the single definition of the prefix semantics; every
/// backend feeds the same engine so fast paths cannot drift from it.
pub(crate) fn compute_prefix_response(
    inputs: &[WorkerPrefixInput],
    blocks_read: u32,
) -> MatchExternalKvPrefixResponse {
    let entries = inputs
        .iter()
        .filter_map(|worker| {
            // An empty address is unroutable for the router (see the proto's
            // worker_address contract); drop it rather than report a match it
            // can never intersect.
            if worker.address.is_empty() {
                return None;
            }
            let prefix = compute_worker_prefix(worker.spec.as_ref(), &worker.blocks);
            (prefix > 0).then(|| (worker.worker_id.clone(), worker.address.clone(), prefix))
        })
        .collect();
    assemble_prefix_response(entries, blocks_read)
}

/// The reusable prefix length for one worker: a safe lower bound on what it can
/// serve. Returns 0 (the worker is excluded) when a component-aware store lacks a
/// spec or the spec carries an unusable rule.
pub(crate) fn compute_worker_prefix(
    spec: Option<&WorkerCacheSpec>,
    blocks: &[Option<BlockComponents>],
) -> u32 {
    match spec {
        // No spec: a worker that reports components but declares none cannot be
        // interpreted safely, so it is excluded. A purely legacy worker keeps the
        // whole-block contiguous prefix (unchanged behaviour).
        None => {
            if blocks_carry_components(blocks) {
                0
            } else {
                legacy_contiguous_prefix(blocks)
            }
        }
        // A declared spec with no components, or from an unsupported (future)
        // version, cannot be interpreted safely → fail closed (worker excluded)
        // rather than misread. A full-only worker declares no spec (None arm).
        Some(spec) if spec.components == 0 || spec.version > SUPPORTED_SPEC_VERSION => 0,
        Some(spec) => component_aware_prefix(spec, blocks),
    }
}

/// The count of leading blocks the worker holds (the legacy whole-block prefix).
fn legacy_contiguous_prefix(blocks: &[Option<BlockComponents>]) -> u32 {
    blocks.iter().take_while(|block| block.is_some()).count() as u32
}

/// Whether any held block carries a non-zero component mask — the signal that a
/// worker is reporting component-aware placement.
fn blocks_carry_components(blocks: &[Option<BlockComponents>]) -> bool {
    blocks
        .iter()
        .flatten()
        .any(|block| block.tier_masks.iter().any(|(_, mask)| *mask != 0))
}

/// The largest boundary `N` such that every required component's fixed rule
/// holds, in a single forward scan:
///   * FULL (always required)  — contiguous: present on every block `0..N`.
///   * SWA (if present)         — trailing window: an unbroken run of SWA ending
///     at `N-1` covering `swa_window_tokens` tokens, or reaching the head.
///   * MAMBA (if present)       — exact boundary: present on block `N-1`.
fn component_aware_prefix(spec: &WorkerCacheSpec, blocks: &[Option<BlockComponents>]) -> u32 {
    let swa_required = spec.components & COMPONENT_SWA != 0;
    let mamba_required = spec.components & COMPONENT_MAMBA != 0;
    let window = spec.swa_window_tokens as u64;
    // SWA without a positive window is an unusable spec → fail closed.
    if swa_required && window == 0 {
        return 0;
    }

    let mut best = 0u32;
    let mut swa_run = 0u64; // contiguous SWA tokens ending at the current block
    let mut swa_head_broken = false; // a SWA gap has been seen before this block
    for (index, block) in blocks.iter().enumerate() {
        // FULL gates contiguity: the prefix cannot extend past a block missing it.
        if !component_available(block, COMPONENT_FULL, spec.full_tier_mask) {
            break;
        }
        let mut boundary_ok = true;
        if swa_required {
            if component_available(block, COMPONENT_SWA, spec.swa_tier_mask) {
                swa_run += block.as_ref().map(|b| b.token_count as u64).unwrap_or(0);
                // Valid if the run reaches the head (never broken) or fills a
                // window; the head case matches the unified cache's accumulator
                // seeded at infinity.
                boundary_ok &= !swa_head_broken || swa_run >= window;
            } else {
                swa_run = 0;
                swa_head_broken = true;
                boundary_ok = false; // boundary block itself must carry SWA
            }
        }
        if mamba_required {
            boundary_ok &= component_available(block, COMPONENT_MAMBA, spec.mamba_tier_mask);
        }
        if boundary_ok {
            best = (index + 1) as u32;
        }
    }
    best
}

/// Whether `component` (a single bit) is resident on `block` at some tier that is
/// both declared servable for that component (`spec_tier_mask`) and servable by
/// the indexer (`SERVABLE_TIER_MASK`).
fn component_available(
    block: &Option<BlockComponents>,
    component: u32,
    spec_tier_mask: u32,
) -> bool {
    let Some(block) = block else {
        return false;
    };
    block.tier_masks.iter().any(|(tier, mask)| {
        mask & component != 0
            && tier_in_mask(SERVABLE_TIER_MASK, *tier)
            && tier_in_mask(spec_tier_mask, *tier)
    })
}

/// Sorts `(worker_id, address, prefix)` entries by prefix descending and builds
/// the response. Shared so the Redis fast path, which computes prefixes during
/// its scan, produces byte-identical shape to the default implementation.
pub(crate) fn assemble_prefix_response(
    mut entries: Vec<(String, String, u32)>,
    blocks_read: u32,
) -> MatchExternalKvPrefixResponse {
    entries.sort_by_key(|entry| std::cmp::Reverse(entry.2));
    let best_prefix_blocks = entries.first().map(|entry| entry.2).unwrap_or(0);
    let matches = entries
        .into_iter()
        .map(
            |(worker_id, worker_address, matched_prefix_blocks)| ExternalKvPrefixMatch {
                worker_address,
                matched_prefix_blocks,
                worker_id,
            },
        )
        .collect();
    MatchExternalKvPrefixResponse {
        matches,
        best_prefix_blocks,
        blocks_read,
    }
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
            component_masks: Vec::new(),
            block_sizes: Vec::new(),
        }
    }

    #[test]
    fn validate_actions_allows_empty_batch() {
        // An empty batch carries no mutation but is not an error.
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

    // --- component-aware prefix rule engine ---

    fn dram() -> i32 {
        crate::pb::TierType::TierDram as i32
    }
    fn ssd() -> i32 {
        crate::pb::TierType::TierSsd as i32
    }

    /// OR the tiers into a `1 << TierType` bitmask.
    fn tmask(tiers: &[i32]) -> u32 {
        tiers.iter().fold(0, |m, t| m | (1u32 << t))
    }

    /// A held block with `(tier, component bitmask)` placements and a token count.
    fn blk(tiers: &[(i32, u32)], token_count: u32) -> Option<BlockComponents> {
        Some(BlockComponents {
            token_count,
            tier_masks: tiers.to_vec(),
        })
    }

    /// A legacy whole-block placement (mask 0) at HBM.
    fn legacy_blk() -> Option<BlockComponents> {
        blk(&[(hbm(), 0)], 0)
    }

    fn spec(
        components: u32,
        swa_window_tokens: u32,
        full_tiers: &[i32],
        swa_tiers: &[i32],
        mamba_tiers: &[i32],
    ) -> WorkerCacheSpec {
        WorkerCacheSpec {
            version: 1,
            components,
            swa_window_tokens,
            full_tier_mask: tmask(full_tiers),
            swa_tier_mask: tmask(swa_tiers),
            mamba_tier_mask: tmask(mamba_tiers),
        }
    }

    #[test]
    fn legacy_no_spec_is_contiguous() {
        let blocks = vec![legacy_blk(), legacy_blk(), legacy_blk(), None, legacy_blk()];
        assert_eq!(compute_worker_prefix(None, &blocks), 3);
    }

    #[test]
    fn component_report_without_spec_is_excluded() {
        // A worker that reports components but declared no spec cannot be
        // interpreted safely, so it contributes nothing (NoSignal-safe).
        let blocks = vec![
            blk(&[(hbm(), COMPONENT_FULL)], 16),
            blk(&[(hbm(), COMPONENT_FULL)], 16),
        ];
        assert_eq!(compute_worker_prefix(None, &blocks), 0);
    }

    #[test]
    fn contiguous_full_stops_at_first_gap() {
        let s = spec(COMPONENT_FULL, 0, &[hbm(), dram()], &[], &[]);
        let blocks = vec![
            blk(&[(hbm(), COMPONENT_FULL)], 16),
            blk(&[(dram(), COMPONENT_FULL)], 16), // full may live on a different servable tier
            blk(&[(hbm(), COMPONENT_SWA)], 16),   // no full here -> prefix stops
            blk(&[(hbm(), COMPONENT_FULL)], 16),
        ];
        assert_eq!(compute_worker_prefix(Some(&s), &blocks), 2);
    }

    #[test]
    fn ssd_only_is_not_servable_in_v1() {
        let s = spec(COMPONENT_FULL, 0, &[hbm(), dram()], &[], &[]);
        let blocks = vec![blk(&[(ssd(), COMPONENT_FULL)], 16)];
        assert_eq!(compute_worker_prefix(Some(&s), &blocks), 0);
    }

    #[test]
    fn trailing_window_requires_unbroken_window_before_boundary() {
        // window = 100 tokens, 50 tokens per block: two contiguous swa blocks
        // cover a window. full is present on every block.
        let s = spec(COMPONENT_FULL | COMPONENT_SWA, 100, &[hbm()], &[hbm()], &[]);
        let with_swa = || blk(&[(hbm(), COMPONENT_FULL | COMPONENT_SWA)], 50);
        let no_swa = || blk(&[(hbm(), COMPONENT_FULL)], 50);
        // swa present everywhere -> full length reusable.
        let blocks = vec![with_swa(), with_swa(), with_swa(), with_swa(), with_swa()];
        assert_eq!(compute_worker_prefix(Some(&s), &blocks), 5);
        // swa tombstoned at block index 3: the largest boundary whose trailing
        // 100-token window is unbroken is N=3 (blocks 1..2 cover 100 tokens).
        let holed = vec![with_swa(), with_swa(), with_swa(), no_swa(), with_swa()];
        assert_eq!(compute_worker_prefix(Some(&s), &holed), 3);
    }

    #[test]
    fn trailing_window_head_is_always_valid() {
        // Fewer tokens than a window, but an unbroken run from the head is valid
        // (matches the unified cache's window accumulator seeded at infinity).
        let s = spec(
            COMPONENT_FULL | COMPONENT_SWA,
            1000,
            &[hbm()],
            &[hbm()],
            &[],
        );
        let blocks = vec![blk(&[(hbm(), COMPONENT_FULL | COMPONENT_SWA)], 16); 2];
        assert_eq!(compute_worker_prefix(Some(&s), &blocks), 2);
    }

    #[test]
    fn swa_without_window_excludes_the_worker() {
        // SWA present but no window configured is an unusable spec -> fail closed.
        let s = spec(COMPONENT_FULL | COMPONENT_SWA, 0, &[hbm()], &[hbm()], &[]);
        let blocks = vec![blk(&[(hbm(), COMPONENT_FULL | COMPONENT_SWA)], 16)];
        assert_eq!(compute_worker_prefix(Some(&s), &blocks), 0);
    }

    #[test]
    fn exact_boundary_only_matches_at_a_checkpoint() {
        // mamba lives only on the 4th block (a leaf checkpoint). full is on all.
        let s = spec(
            COMPONENT_FULL | COMPONENT_MAMBA,
            0,
            &[hbm(), dram()],
            &[],
            &[hbm(), dram()],
        );
        let blocks = vec![
            blk(&[(hbm(), COMPONENT_FULL)], 16),
            blk(&[(hbm(), COMPONENT_FULL)], 16),
            blk(&[(hbm(), COMPONENT_FULL)], 16),
            blk(&[(hbm(), COMPONENT_FULL | COMPONENT_MAMBA)], 16),
        ];
        assert_eq!(compute_worker_prefix(Some(&s), &blocks), 4);
        // A shorter request that never reaches the checkpoint cannot reuse it.
        assert_eq!(compute_worker_prefix(Some(&s), &blocks[..2]), 0);
    }

    #[test]
    fn unsupported_spec_version_excludes_the_worker() {
        let mut s = spec(COMPONENT_FULL, 0, &[hbm()], &[], &[]);
        s.version = SUPPORTED_SPEC_VERSION + 1; // a future wire we cannot read
        let blocks = vec![blk(&[(hbm(), COMPONENT_FULL)], 16)];
        assert_eq!(compute_worker_prefix(Some(&s), &blocks), 0);
    }

    #[test]
    fn empty_declared_spec_excludes_the_worker() {
        // A full-only worker declares no spec (None); a declared-but-empty spec
        // (no components) is a misconfiguration and fails closed.
        let s = spec(0, 0, &[hbm()], &[], &[]);
        let blocks = vec![blk(&[(hbm(), COMPONENT_FULL)], 16)];
        assert_eq!(compute_worker_prefix(Some(&s), &blocks), 0);
    }

    #[test]
    fn missing_component_data_under_spec_excludes() {
        // Spec requires full+swa, but the worker reported legacy whole-block
        // placement (mask 0, e.g. the component flag was off): full itself cannot
        // be confirmed, so it is excluded rather than over-reported.
        let s = spec(COMPONENT_FULL | COMPONENT_SWA, 100, &[hbm()], &[hbm()], &[]);
        let blocks = vec![legacy_blk(), legacy_blk()];
        assert_eq!(compute_worker_prefix(Some(&s), &blocks), 0);
    }
}
