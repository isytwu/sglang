// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! The contract a KV-aware router relies on.
//!
//! sgl-router routes on a process-local hash tree fed by ZMQ. Pointing it at
//! this indexer instead only works if the indexer answers the same question the
//! tree does, so this file pins the relationship between them.
//!
//! The tree's answer is reproduced by [`ReferenceTree`] below — a port of
//! `sgl-router`'s `HashTree`, small enough to state in full and independent
//! enough that a regression in the indexer cannot also move the oracle. The two
//! crates live in separate Cargo workspaces, so sharing the real one is not an
//! option.
//!
//! Two properties are asserted against it:
//!
//! * **Equivalence.** With gap-free placement — what SGLang produces, since its
//!   radix cache evicts leaves before their ancestors — the two agree exactly,
//!   on the prefix length and on the winning worker set.
//! * **Safety.** Where they can differ, the indexer is always the conservative
//!   one. It requires a worker to hold every block of the prefix; the tree only
//!   requires the deepest. So the indexer can under-report a cache hit, but can
//!   never send a request to a worker that cannot serve it.
//!
//! The hashes are SGLang's real block hashes, taken from the cross-language
//! goldens that `sgl-router`'s `kv_events::hash` module pins against the engine.
#![cfg(feature = "redis-backend")]

#[path = "common/require.rs"]
mod require;
#[path = "common/id.rs"]
mod test_id;
#[path = "common/kv.rs"]
mod test_kv;

use std::collections::{BTreeSet, HashMap};

use sgl_kv_indexer::client::{hash_from_wire, hash_to_wire};
use sgl_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ExternalKvActionType, ExternalKvPrefixMatch,
    MatchExternalKvPrefixRequest, MatchExternalKvPrefixResponse, MatchExternalKvRequest,
};
use sgl_kv_indexer::{KvIndexerBackend, RedisKvIndexerBackend};
use test_id::nanos;
use test_kv::{action, apply_request, hbm};

/// SGLang block hashes for a four-block chain, from the engine-pinned
/// cross-language goldens (`chain([10,20,30,40,50,60,70,80], 2)`). Using real
/// values keeps the wire encoding under test rather than assuming it.
const CHAIN: [i64; 4] = [
    978178666101069530,
    -895308556211281782,
    -8033692805846017938,
    835415944263129316,
];

// ---------------------------------------------------------------------------
// Reference oracle: sgl-router's HashTree, reduced to what match_prefix needs.
// ---------------------------------------------------------------------------

/// The subset of `sgl-router`'s `MatchResult` this contract covers.
#[derive(Debug, PartialEq, Eq)]
struct TreeMatch {
    matched_blocks: usize,
    /// Workers at the deepest matched node, by routing URL.
    workers: BTreeSet<String>,
}

/// Port of `sgl-router`'s `HashTree`.
///
/// A trie over block hashes where each node records the workers that reported
/// that block. The detail that matters here is that node existence and worker
/// membership are independent: removing a worker from a node leaves the node in
/// place as long as it still has children, so a walk passes straight through it.
/// That is what lets the tree reach a depth no single worker can actually
/// serve.
#[derive(Default)]
struct ReferenceTree {
    /// Keyed by the full chain from the root, because a node's identity in the
    /// real tree is its path, not its block hash alone.
    nodes: HashMap<Vec<i64>, BTreeSet<String>>,
}

impl ReferenceTree {
    /// `HashTree::insert` for a chain rooted at the request's first block.
    fn insert(&mut self, worker: &str, chain: &[i64]) {
        for depth in 0..chain.len() {
            self.nodes
                .entry(chain[..=depth].to_vec())
                .or_default()
                .insert(worker.to_string());
        }
    }

    /// `HashTree::remove` for one block. The node survives: the real tree only
    /// prunes once a node has neither workers nor children, and every node we
    /// remove from here still has a child.
    fn remove(&mut self, worker: &str, path: &[i64]) {
        if let Some(workers) = self.nodes.get_mut(path) {
            workers.remove(worker);
        }
    }

    /// `HashTree::match_prefix(None, query)`.
    fn match_prefix(&self, query: &[i64]) -> TreeMatch {
        let mut matched = 0usize;
        let mut deepest = BTreeSet::new();
        for depth in 0..query.len() {
            let Some(workers) = self.nodes.get(&query[..=depth]) else {
                break;
            };
            matched = depth + 1;
            deepest = workers.clone();
        }
        TreeMatch {
            matched_blocks: matched,
            workers: deepest,
        }
    }
}

/// The tree the given placements would have built, for placements with no gaps.
fn tree_of(placements: &[(&str, Vec<i64>)]) -> ReferenceTree {
    let mut tree = ReferenceTree::default();
    for (worker, held) in placements {
        tree.insert(worker, held);
    }
    tree
}

// ---------------------------------------------------------------------------
// Indexer side
// ---------------------------------------------------------------------------

async fn backend(test: &str) -> Option<RedisKvIndexerBackend> {
    let ns = format!("contract:{test}:{}", nanos());
    if let Ok(url) = std::env::var("KV_INDEXER_REDIS_URL") {
        Some(
            RedisKvIndexerBackend::connect_single(&url, ns)
                .await
                .expect("connect single"),
        )
    } else {
        require::skip(test, "KV_INDEXER_REDIS_URL is not set");
        None
    }
}

fn wire(hashes: &[i64]) -> Vec<String> {
    hashes.iter().copied().map(hash_to_wire).collect()
}

/// One batch for `worker`, whose id and address are both its routing URL —
/// mirroring the identity contract a router depends on.
fn batch(
    worker: &str,
    seq: u64,
    kind: ExternalKvActionType,
    hashes: &[i64],
) -> ApplyExternalKvBatchRequest {
    let wire = wire(hashes);
    let refs: Vec<&str> = wire.iter().map(String::as_str).collect();
    apply_request(worker, worker, seq, vec![action(kind, hbm(), &refs)])
}

async fn apply(b: &RedisKvIndexerBackend, request: ApplyExternalKvBatchRequest) {
    b.apply_external_kv_batch(request)
        .await
        .expect("apply batch");
}

async fn load(b: &RedisKvIndexerBackend, placements: &[(&str, Vec<i64>)]) {
    for (worker, held) in placements {
        apply(
            b,
            batch(worker, 1, ExternalKvActionType::ActionReport, held),
        )
        .await;
    }
}

async fn query(b: &RedisKvIndexerBackend, hashes: &[i64]) -> MatchExternalKvPrefixResponse {
    b.match_external_kv_prefix(MatchExternalKvPrefixRequest {
        hashes: wire(hashes),
        ..Default::default()
    })
    .await
    .expect("prefix match")
}

/// The indexer's answer in the oracle's shape: longest prefix, and the workers
/// tied at it.
fn as_tree_match(resp: &MatchExternalKvPrefixResponse) -> TreeMatch {
    TreeMatch {
        matched_blocks: resp.best_prefix_blocks as usize,
        workers: resp
            .matches
            .iter()
            .filter(|m| m.matched_prefix_blocks == resp.best_prefix_blocks)
            .map(|m| m.address.clone())
            .collect(),
    }
}

fn prefix_pairs(resp: &MatchExternalKvPrefixResponse) -> Vec<(String, u32)> {
    resp.matches
        .iter()
        .map(|m| (m.worker_id.clone(), m.matched_prefix_blocks))
        .collect()
}

macro_rules! contract {
    ($name:ident, $b:ident, $body:block) => {
        #[tokio::test]
        async fn $name() {
            let Some($b) = backend(stringify!($name)).await else {
                return;
            };
            $body
        }
    };
}

// --- equivalence on gap-free placement -----------------------------------

contract!(agrees_with_the_tree_on_a_full_prefix_hit, b, {
    let placements = vec![("http://w1:30000", CHAIN.to_vec())];
    load(&b, &placements).await;
    let resp = query(&b, &CHAIN).await;
    assert_eq!(
        as_tree_match(&resp),
        tree_of(&placements).match_prefix(&CHAIN)
    );
    assert_eq!(resp.best_prefix_blocks, 4);
});

contract!(agrees_with_the_tree_on_a_partial_prefix_hit, b, {
    let placements = vec![("http://w1:30000", CHAIN[..2].to_vec())];
    load(&b, &placements).await;
    let resp = query(&b, &CHAIN).await;
    assert_eq!(
        as_tree_match(&resp),
        tree_of(&placements).match_prefix(&CHAIN)
    );
    assert_eq!(resp.best_prefix_blocks, 2);
});

contract!(agrees_with_the_tree_on_a_complete_miss, b, {
    // Holding only a later block is worth nothing: the request cannot skip the
    // blocks before it.
    let placements = vec![("http://w1:30000", vec![CHAIN[3]])];
    load(&b, &placements).await;
    let resp = query(&b, &CHAIN).await;
    assert_eq!(
        as_tree_match(&resp),
        tree_of(&placements).match_prefix(&CHAIN)
    );
    assert_eq!(resp.best_prefix_blocks, 0);
});

// The tie-break the router depends on: it picks by load among the workers
// holding the deepest prefix, so that set has to be exactly right.
contract!(agrees_with_the_tree_on_the_winning_worker_set, b, {
    let placements = vec![
        ("http://w1:30000", CHAIN.to_vec()),
        ("http://w2:30000", CHAIN.to_vec()),
        ("http://w3:30000", CHAIN[..1].to_vec()),
    ];
    load(&b, &placements).await;
    let resp = query(&b, &CHAIN).await;
    assert_eq!(
        as_tree_match(&resp),
        tree_of(&placements).match_prefix(&CHAIN)
    );
    assert_eq!(
        as_tree_match(&resp).workers,
        BTreeSet::from(["http://w1:30000".to_string(), "http://w2:30000".to_string()]),
        "only the workers holding the deepest prefix are candidates"
    );
});

// --- the one place they diverge, and its direction ------------------------

// w1 reported the whole chain, then lost the second block to eviction. The
// tree's walk passes through that node — it still has a child — and reports a
// four-block hit on w1, which w1 cannot serve. The indexer stops at the hole.
//
// This is the entire safety argument for swapping the tree for the indexer:
// where they disagree, the indexer under-reports.
contract!(never_reports_a_longer_prefix_than_the_tree, b, {
    let worker = "http://w1:30000";
    apply(
        &b,
        batch(worker, 1, ExternalKvActionType::ActionReport, &CHAIN),
    )
    .await;
    apply(
        &b,
        batch(worker, 2, ExternalKvActionType::ActionRevoke, &[CHAIN[1]]),
    )
    .await;

    let mut tree = ReferenceTree::default();
    tree.insert(worker, &CHAIN);
    tree.remove(worker, &CHAIN[..2]);
    let expected = tree.match_prefix(&CHAIN);
    assert_eq!(
        expected.matched_blocks, 4,
        "the tree walks past a node it emptied, so this fixture must expose the divergence"
    );
    assert_eq!(expected.workers, BTreeSet::from([worker.to_string()]));

    let resp = query(&b, &CHAIN).await;
    assert_eq!(
        resp.best_prefix_blocks, 1,
        "the indexer must stop at the evicted block"
    );
    assert!(
        (resp.best_prefix_blocks as usize) < expected.matched_blocks,
        "the indexer must never claim more reusable blocks than the tree"
    );
});

// --- identity -------------------------------------------------------------

// A router intersects the returned address with its own registered worker URLs,
// so anything but a byte-exact match routes nowhere.
contract!(returns_the_worker_url_the_router_registered, b, {
    let url = "http://sglang-worker-0.default.svc:30000";
    load(&b, &[(url, CHAIN.to_vec())]).await;
    let resp = query(&b, &CHAIN).await;
    assert_eq!(resp.matches.len(), 1);
    assert_eq!(resp.matches[0].address, url);
});

// --- fast path against the written definition -----------------------------

// The trait's default `match_external_kv_prefix` is the definition of the
// prefix semantics; the Redis override exists only to answer it cheaply. This
// pins the two together so the fast path cannot drift from the spec.
contract!(redis_scan_agrees_with_the_reference_implementation, b, {
    let placements = vec![
        ("http://w1:30000", CHAIN.to_vec()),
        ("http://w2:30000", CHAIN[..2].to_vec()),
        ("http://w3:30000", CHAIN[..1].to_vec()),
    ];
    load(&b, &placements).await;

    let fast = query(&b, &CHAIN).await;
    let reference = reference_prefix(&b, &CHAIN).await;

    assert_eq!(fast.best_prefix_blocks, reference.best_prefix_blocks);
    assert_eq!(
        prefix_pairs(&fast),
        prefix_pairs(&reference),
        "the Redis scan and the reference implementation must agree per worker"
    );
    assert!(
        fast.queried_blocks <= reference.queried_blocks,
        "the fast path must not read more than the reference"
    );
});

/// Folds a plain placement-set match into prefix lengths, which is what the
/// trait's default implementation does internally.
async fn reference_prefix(
    b: &RedisKvIndexerBackend,
    hashes: &[i64],
) -> MatchExternalKvPrefixResponse {
    let placements = b
        .match_external_kv(MatchExternalKvRequest {
            hashes: wire(hashes),
            count_as_hit: false,
        })
        .await
        .expect("placement match");

    let query = wire(hashes);
    let mut matches: Vec<ExternalKvPrefixMatch> = placements
        .matches
        .iter()
        .filter(|node| !node.address.is_empty())
        .filter_map(|node| {
            let mut held: HashMap<&str, BTreeSet<i32>> = HashMap::new();
            for tier in &node.hashes_by_tier {
                for hash in &tier.hashes {
                    held.entry(hash.as_str()).or_default().insert(tier.tier);
                }
            }
            let mut tiers = BTreeSet::new();
            let mut matched = 0u32;
            for hash in &query {
                let Some(at) = held.get(hash.as_str()) else {
                    break;
                };
                tiers.extend(at.iter().copied());
                matched += 1;
            }
            (matched > 0).then(|| ExternalKvPrefixMatch {
                worker_id: node.worker_id.clone(),
                address: node.address.clone(),
                dp_rank: node.dp_rank,
                matched_prefix_blocks: matched,
                tiers: tiers.into_iter().collect(),
            })
        })
        .collect();
    matches.sort_by(|a, b| {
        b.matched_prefix_blocks
            .cmp(&a.matched_prefix_blocks)
            .then_with(|| a.worker_id.cmp(&b.worker_id))
    });
    let best_prefix_blocks = matches
        .first()
        .map(|m| m.matched_prefix_blocks)
        .unwrap_or(0);
    MatchExternalKvPrefixResponse {
        matches,
        best_prefix_blocks,
        queried_blocks: hashes.len() as u32,
        spec_mismatched_workers: 0,
    }
}

// ---------------------------------------------------------------------------
// Oracle self-checks. These need no store, and keep the comparisons above from
// passing vacuously.
// ---------------------------------------------------------------------------

#[test]
fn oracle_walks_to_the_deepest_reachable_node() {
    let tree = tree_of(&[("http://w1:30000", CHAIN.to_vec())]);
    assert_eq!(
        tree.match_prefix(&CHAIN),
        TreeMatch {
            matched_blocks: 4,
            workers: BTreeSet::from(["http://w1:30000".to_string()]),
        }
    );
}

#[test]
fn oracle_returns_only_the_deepest_nodes_workers() {
    let tree = tree_of(&[
        ("http://w1:30000", CHAIN.to_vec()),
        ("http://w2:30000", CHAIN[..1].to_vec()),
    ]);
    let matched = tree.match_prefix(&CHAIN);
    assert_eq!(matched.matched_blocks, 4);
    assert_eq!(
        matched.workers,
        BTreeSet::from(["http://w1:30000".to_string()]),
        "a worker holding only the first block is not a candidate for a 4-block prefix"
    );
}

#[test]
fn oracle_stops_where_no_node_exists() {
    let tree = tree_of(&[("http://w1:30000", CHAIN[..2].to_vec())]);
    assert_eq!(tree.match_prefix(&CHAIN).matched_blocks, 2);
}

/// The behavior the indexer deliberately does not copy: an emptied node is
/// still walked through, so the tree reports a depth its own worker set cannot
/// back up.
#[test]
fn oracle_walks_through_a_node_it_emptied() {
    let mut tree = ReferenceTree::default();
    tree.insert("http://w1:30000", &CHAIN);
    tree.remove("http://w1:30000", &CHAIN[..2]);
    let matched = tree.match_prefix(&CHAIN);
    assert_eq!(matched.matched_blocks, 4);
    assert_eq!(
        matched.workers,
        BTreeSet::from(["http://w1:30000".to_string()])
    );
}

/// The goldens must be the engine's real hashes, not placeholders: the wire
/// encoding is part of the contract, and negative values are the case a naive
/// unsigned encoding gets wrong.
#[test]
fn goldens_round_trip_through_the_wire_encoding() {
    assert!(
        CHAIN.iter().any(|hash| *hash < 0),
        "the fixture must exercise negative hashes, which SGLang routinely emits"
    );
    for hash in CHAIN {
        assert_eq!(hash_from_wire(&hash_to_wire(hash)), Some(hash));
    }
}
