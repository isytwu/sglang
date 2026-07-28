// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SGLang KV Indexer: a gRPC service that tracks externally-managed KV cache
//! block placements (as reported by inference engines such as SGLang HiCache)
//! and answers placement-match queries for KV-aware routing.

/// The SGLang event follower. Behind a feature so a routing client can depend
/// on this crate for [`client`] and [`pb`] alone, without pulling in ZMQ.
#[cfg(feature = "bridge")]
pub mod bridge;

pub mod client;

pub mod pb {
    tonic::include_proto!("kv_indexer.v1");
}

mod service;
mod shutdown;

#[cfg(feature = "redis-backend")]
pub mod redis_backend;

pub use service::{KvIndexerBackend, KvIndexerService, MAX_HASHES_PER_REQUEST};
pub use shutdown::shutdown_signal;

pub use client::{
    ExternalPrefixIndex, KvIndexerPrefixIndex, NoSignalReason, PrefixIndexConfig,
    PrefixIndexOutcome, PrefixMatch, PrefixQuery,
};

#[cfg(feature = "redis-backend")]
pub use redis_backend::RedisKvIndexerBackend;
