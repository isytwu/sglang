"""Unit tests for external-linker device pool assembly."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
    _build_deepseek_v4_device_pool_group,
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDevicePoolEntry(CustomTestCase):
    def test_sparse_multi_component_layer_ranges(self):
        k0 = torch.zeros((8, 3), dtype=torch.uint8)
        k2 = torch.zeros((8, 5), dtype=torch.uint8)
        v0 = torch.zeros((8, 7), dtype=torch.uint8)
        v2 = torch.zeros((8, 11), dtype=torch.uint8)
        pool = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[k0, k2], [v0, v2]],
            layer_mapping={0: 0, 2: 1},
            page_size=2,
            rows_are_pages=False,
            packed=False,
        )

        indices = torch.tensor([0, 1, 4, 5])
        locations = pool.prepare_locations(indices)
        self.assertEqual(locations, [0, 4])
        pointers, sizes = pool.get_page_buffer_meta(indices)
        self.assertEqual(
            pointers,
            [
                buffer[row].data_ptr()
                for row in locations
                for buffer in (k0, k2, v0, v2)
            ],
        )
        self.assertEqual(sizes, [6, 10, 14, 22] * 2)
        self.assertIsNone(pool.get_prepared_layer_range_meta(locations, 1))

        pointers, sizes, offsets = pool.get_prepared_layer_range_meta(locations, 2)
        self.assertEqual(
            pointers,
            [
                [k2[0].data_ptr()],
                [v2[0].data_ptr()],
                [k2[4].data_ptr()],
                [v2[4].data_ptr()],
            ],
        )
        self.assertEqual(sizes, [[10], [22], [10], [22]])
        self.assertEqual(offsets, [[6], [14], [6], [14]])

    def test_rejects_invalid_pages_and_empty_buffers(self):
        with self.assertRaisesRegex(ValueError, "has no storage buffers"):
            DevicePoolEntry(
                name=PoolName.KV,
                indices_from_pool=PoolName.KV,
                device_pool=None,
                components=[],
                layer_mapping={},
                page_size=2,
                rows_are_pages=False,
            )

        pool = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[torch.zeros((8, 3), dtype=torch.uint8)]],
            layer_mapping={0: 0},
            page_size=2,
            rows_are_pages=False,
        )
        for indices, error in (
            (torch.tensor([0]), "multiple of page_size"),
            (torch.tensor([1, 2]), "aligned contiguous pages"),
            (torch.tensor([0, 2]), "aligned contiguous pages"),
            (torch.tensor([8, 9]), "exceeds buffer shapes"),
        ):
            with self.subTest(indices=indices.tolist()):
                with self.assertRaisesRegex(ValueError, error):
                    pool.prepare_locations(indices)

    def test_dcp_widened_locs_collapse_to_rank_rows(self):
        def pool(dcp_size):
            return DevicePoolEntry(
                name=PoolName.KV,
                indices_from_pool=PoolName.KV,
                device_pool=None,
                components=[[torch.zeros((10, 4), dtype=torch.uint8)]],
                layer_mapping={0: 0},
                page_size=8,
                rows_are_pages=False,
                dcp_size=dcp_size,
            )

        # A widened page of 8 locs is 2 rows on each of the 4 ranks.
        self.assertEqual(pool(4).prepare_locations(torch.arange(8, 16)), [2])
        _, sizes = pool(4).get_page_buffer_meta(torch.arange(8, 16))
        self.assertEqual(sizes, [4 * 2])
        with self.assertRaisesRegex(ValueError, "must be a multiple of"):
            pool(3)


class TestDevicePoolGroup(CustomTestCase):
    def test_resolve_transfers_expands_physical_pools(self):
        entries = [
            SimpleNamespace(
                name=PoolName.KV,
                indices_from_pool=PoolName.KV,
                translate_indices=lambda indices: indices,
            ),
            SimpleNamespace(
                name=PoolName.INDEXER,
                indices_from_pool=PoolName.KV,
                translate_indices=lambda indices: indices + 100,
            ),
        ]
        group = DevicePoolGroup(entries, num_layers=2, page_size=2)
        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=["a", "b"],
            device_indices=torch.tensor([0, 1, 4, 5]),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        resolved = group.resolve_transfers([transfer])

        self.assertEqual(
            [item.name for item in resolved], [PoolName.KV, PoolName.INDEXER]
        )
        self.assertEqual(resolved[0].host_indices.tolist(), [0, 1, 4, 5])
        self.assertEqual(resolved[1].host_indices.tolist(), [100, 101, 104, 105])
        self.assertTrue(
            all(item.hit_policy == PoolHitPolicy.ALL_PAGES for item in resolved)
        )

    def test_partial_side_pool_requires_explicit_opt_in(self):
        entry = SimpleNamespace(
            name=PoolName.SWA,
            indices_from_pool=PoolName.SWA,
            translate_indices=lambda indices: indices + 100,
        )
        group = DevicePoolGroup([entry], num_layers=1, page_size=2)
        transfer = PoolTransfer(
            name=PoolName.SWA,
            keys=["b", "d"],
            device_indices=torch.tensor([20, 21, 24, 25]),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        self.assertEqual(group.resolve_transfers([transfer]), [])
        resolved = group.resolve_transfers(
            [transfer], allow_partial=True, allow_missing_kv=True
        )

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].name, PoolName.SWA)
        self.assertEqual(resolved[0].keys, ["b", "d"])
        self.assertEqual(resolved[0].host_indices.tolist(), [120, 121, 124, 125])
        self.assertEqual(resolved[0].hit_policy, PoolHitPolicy.TRAILING_PAGES)


class TestHybridDevicePoolAssembler(CustomTestCase):
    def test_deepseek_v4_maps_sparse_sidecars(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4LayerItem,
            DeepSeekV4TokenToKVPool,
        )

        def state_pool():
            return SimpleNamespace(
                ring_size=2,
                kv_score_buffer=SimpleNamespace(kv_score=torch.zeros((8, 3))),
            )

        kvcache = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        kvcache._unified_kv = False
        kvcache.start_layer = 1
        kvcache.end_layer = 4
        kvcache.swa_page_size = 2
        kvcache.swa_kv_pool = SimpleNamespace(
            page_size=2,
            kv_buffer=[torch.zeros((8, 3), dtype=torch.uint8) for _ in range(3)],
        )
        kvcache.c4_kv_pool = SimpleNamespace(
            kv_buffer=[torch.zeros((8, 5), dtype=torch.uint8) for _ in range(2)],
            bytes_per_page_padded=5,
        )
        kvcache.c4_indexer_kv_pool = SimpleNamespace(
            index_k_with_scale_buffer=[
                torch.zeros((8, 7), dtype=torch.uint8) for _ in range(2)
            ]
        )
        kvcache.c128_kv_pool = SimpleNamespace(
            kv_buffer=[torch.zeros((8, 11), dtype=torch.uint8)],
            bytes_per_page_padded=11,
        )
        kvcache.layer_mapping = [
            DeepSeekV4LayerItem(0, -1),
            DeepSeekV4LayerItem(4, 0),
            DeepSeekV4LayerItem(128, 0),
            DeepSeekV4LayerItem(4, 1),
        ]
        kvcache.compress_state_pools = [None, state_pool(), None, state_pool()]
        kvcache.indexer_compress_state_pools = [
            None,
            state_pool(),
            None,
            state_pool(),
        ]
        draft_swa_buffers = [
            torch.zeros((8, 13), dtype=torch.uint8),
            torch.zeros((8, 17), dtype=torch.uint8),
        ]

        group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=2,
            params=SimpleNamespace(
                mtp_draft_device_pools=(
                    SimpleNamespace(
                        swa_kv_pool=SimpleNamespace(kv_buffer=draft_swa_buffers)
                    ),
                )
            ),
            components={ComponentType.FULL, ComponentType.SWA},
        )

        self.assertEqual(group.num_layers, 3)
        self.assertTrue(group.rank_replicated)
        self.assertEqual(
            set(group.entry_map),
            {
                PoolName.SWA,
                PoolName.DEEPSEEK_V4_C4,
                PoolName.DEEPSEEK_V4_C4_INDEXER,
                PoolName.DEEPSEEK_V4_C128,
                PoolName.DEEPSEEK_V4_C4_STATE,
                PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            },
        )
        self.assertEqual(group.sources[PoolName.DEEPSEEK_V4_C4], PoolName.KV)
        self.assertEqual(group.sources[PoolName.DEEPSEEK_V4_C4_STATE], PoolName.SWA)

        c4_pool = group.entry_map[PoolName.DEEPSEEK_V4_C4]
        pointers, sizes = c4_pool.get_page_buffer_meta(torch.tensor([0, 1]))
        self.assertEqual(len(pointers), 2)
        self.assertEqual(sizes, [5, 5])
        _, sizes, offsets = c4_pool.get_prepared_layer_range_meta([0], 2)
        self.assertEqual(sizes, [[5]])
        self.assertEqual(offsets, [[5]])
        self.assertIsNone(c4_pool.get_prepared_layer_range_meta([0], 1))

        swa_pool = group.entry_map[PoolName.SWA]
        _, sizes, offsets = swa_pool.get_prepared_layer_range_meta([0], 0)
        self.assertEqual(sizes, [[3, 13]])
        self.assertEqual(offsets, [[0, 9]])
        _, sizes, offsets = swa_pool.get_prepared_layer_range_meta([0], 1)
        self.assertEqual(sizes, [[3, 17]])
        self.assertEqual(offsets, [[3, 22]])

    def test_unified_deepseek_v4_uses_only_compressed_pools(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4LayerItem

        for split_indexer in (False, True):
            with self.subTest(split_indexer=split_indexer):
                c4 = [torch.zeros((4, width), dtype=torch.uint8) for width in (5, 7)]
                c128 = [torch.zeros((4, 11), dtype=torch.uint8)]
                expected = {
                    PoolName.DEEPSEEK_V4_C4: c4,
                    PoolName.DEEPSEEK_V4_C128: c128,
                }
                if split_indexer:
                    payload = [
                        torch.zeros((4, 1, 4, 1, 16), dtype=torch.uint8)
                        for _ in range(2)
                    ]
                    scale = [
                        torch.zeros((4, 1, 4, 1), dtype=torch.uint8) for _ in range(2)
                    ]
                    indexer = SimpleNamespace(
                        index_k_with_scale_buffer=None,
                        index_k_payload_buffer=payload,
                        index_k_scale_buffer=scale,
                    )
                    expected[PoolName.DEEPSEEK_V4_C4_INDEXER] = [
                        b.flatten(1) for b in payload
                    ]
                    expected[PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE] = [
                        b.flatten(1) for b in scale
                    ]
                else:
                    buffers = [
                        torch.zeros((4, width), dtype=torch.uint8) for width in (13, 17)
                    ]
                    indexer = SimpleNamespace(index_k_with_scale_buffer=buffers)
                    expected[PoolName.DEEPSEEK_V4_C4_INDEXER] = buffers
                regions = {4: (c4, 7), 128: (c128, 11)}
                kvcache = SimpleNamespace(
                    _unified_kv=True,
                    start_layer=0,
                    end_layer=3,
                    layer_mapping=[
                        DeepSeekV4LayerItem(4, 1),
                        DeepSeekV4LayerItem(128, 0),
                        DeepSeekV4LayerItem(4, 0),
                    ],
                    # Unified KV has neither paged KV nor an index-addressed SWA pool.
                    swa_kv_pool=None,
                    c4_kv_pool=None,
                    c128_kv_pool=None,
                    swa_page_size=3,
                    c4_indexer_kv_pool=indexer,
                    unified_region_buffers=Mock(side_effect=regions.__getitem__),
                    # bf16 unified rows are a single pool: no rope sibling.
                    unified_rope_region_buffers=Mock(return_value=None),
                )
                group = _build_deepseek_v4_device_pool_group(kvcache, page_size=2)

                # expected carries no *_ROPE entry, so this also pins that the
                # bf16 layout ships exactly what it did before.
                self.assertEqual(set(group.entry_map), set(expected))
                self.assertEqual(set(group.sources.values()), {PoolName.KV})
                self.assertTrue(group.rank_replicated)
                for name, buffers in expected.items():
                    entry = group.entry_map[name]
                    actual = entry.components[0]
                    self.assertEqual(len(actual), len(buffers))
                    for got, want in zip(actual, buffers):
                        self.assertEqual(got.data_ptr(), want.data_ptr())
                        self.assertEqual(got.shape, want.shape)
                self.assertEqual(
                    kvcache.unified_region_buffers.call_args_list, [call(4), call(128)]
                )
                resolved = group.resolve_transfers(
                    [
                        PoolTransfer(
                            name=PoolName.KV,
                            keys=["page-0"],
                            device_indices=torch.tensor([0, 1]),
                        )
                    ]
                )
                self.assertEqual({t.name for t in resolved}, set(expected))

    def test_unified_deepseek_v4_fp8_ships_the_rope_half(self):
        """fp8 rows are two pools; shipping nope alone leaves stale rope."""
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4LayerItem

        c4 = [torch.zeros((4, width), dtype=torch.uint8) for width in (5, 7)]
        c128 = [torch.zeros((4, 11), dtype=torch.uint8)]
        c4_rope = [torch.zeros((4, 3), dtype=torch.uint8) for _ in range(2)]
        c128_rope = [torch.zeros((4, 3), dtype=torch.uint8)]
        indexer = [torch.zeros((4, width), dtype=torch.uint8) for width in (13, 17)]
        regions = {4: (c4, 7), 128: (c128, 11)}
        rope_regions = {4: (c4_rope, 3), 128: (c128_rope, 3)}
        kvcache = SimpleNamespace(
            _unified_kv=True,
            start_layer=0,
            end_layer=3,
            layer_mapping=[
                DeepSeekV4LayerItem(4, 1),
                DeepSeekV4LayerItem(128, 0),
                DeepSeekV4LayerItem(4, 0),
            ],
            swa_kv_pool=None,
            c4_kv_pool=None,
            c128_kv_pool=None,
            swa_page_size=3,
            c4_indexer_kv_pool=SimpleNamespace(index_k_with_scale_buffer=indexer),
            unified_region_buffers=Mock(side_effect=regions.__getitem__),
            unified_rope_region_buffers=Mock(side_effect=rope_regions.__getitem__),
        )
        group = _build_deepseek_v4_device_pool_group(kvcache, page_size=2)

        expected_rope = {
            PoolName.DEEPSEEK_V4_C4_ROPE: c4_rope,
            PoolName.DEEPSEEK_V4_C128_ROPE: c128_rope,
        }
        self.assertTrue(set(expected_rope).issubset(group.entry_map))
        for name, buffers in expected_rope.items():
            actual = group.entry_map[name].components[0]
            self.assertEqual(len(actual), len(buffers))
            for got, want in zip(actual, buffers):
                self.assertEqual(got.data_ptr(), want.data_ptr())
                self.assertEqual(got.shape, want.shape)
        self.assertEqual(
            kvcache.unified_rope_region_buffers.call_args_list, [call(4), call(128)]
        )

        # One row index addresses both halves: same layer mapping, same source.
        for rope_name, nope_name in (
            (PoolName.DEEPSEEK_V4_C4_ROPE, PoolName.DEEPSEEK_V4_C4),
            (PoolName.DEEPSEEK_V4_C128_ROPE, PoolName.DEEPSEEK_V4_C128),
        ):
            self.assertEqual(
                group.entry_map[rope_name].layer_mapping,
                group.entry_map[nope_name].layer_mapping,
            )
            self.assertEqual(group.sources[rope_name], group.sources[nope_name])

        # One KV transfer must expand to both halves.
        resolved = group.resolve_transfers(
            [
                PoolTransfer(
                    name=PoolName.KV,
                    keys=["page-0"],
                    device_indices=torch.tensor([0, 1]),
                )
            ]
        )
        self.assertTrue(set(expected_rope).issubset({t.name for t in resolved}))

    def test_deepseek_v4_still_rejects_hisparse(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool

        kvcache = SimpleNamespace(
            c4_kv_pool=HiSparseC4DevicePool.__new__(HiSparseC4DevicePool)
        )
        with self.assertRaisesRegex(ValueError, "does not support HiSparse"):
            _build_deepseek_v4_device_pool_group(kvcache, 2)

    def test_dsa_uses_hybrid_assembler_strategy(self):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        def dsa_pool(kv_width, index_width):
            pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
            pool.page_size = 2
            pool.layer_num = 1
            pool.kv_buffer = [torch.zeros((8, kv_width), dtype=torch.uint8)]
            pool.index_key_cache = SimpleNamespace(
                buffer=[torch.zeros((4, index_width), dtype=torch.uint8)]
            )
            return pool

        kvcache = dsa_pool(3, 7)
        kvcache.layer_num = 2
        kvcache.kv_buffer.append(torch.zeros((8, 5), dtype=torch.uint8))
        kvcache.index_key_cache.buffer.append(torch.zeros((4, 11), dtype=torch.uint8))
        draft_pools = (dsa_pool(13, 17), dsa_pool(19, 23))

        with get_parallel().override(attn_dcp_size=1):
            group = resolve_hybrid_device_pool_group(
                kvcache=kvcache,
                page_size=2,
                params=SimpleNamespace(mtp_draft_device_pools=draft_pools),
                components={ComponentType.FULL},
            )

        self.assertEqual(group.num_layers, 2)
        self.assertTrue(group.rank_replicated)
        self.assertEqual(set(group.entry_map), {PoolName.KV, PoolName.INDEXER})
        self.assertEqual(
            group.sources,
            {
                PoolName.KV: PoolName.KV,
                PoolName.INDEXER: PoolName.KV,
            },
        )
        _, sizes, offsets = group.entry_map[PoolName.KV].get_prepared_layer_range_meta(
            [0], 0
        )
        self.assertEqual(sizes, [[6, 26]])
        self.assertEqual(offsets, [[0, 16]])
        _, sizes, offsets = group.entry_map[
            PoolName.INDEXER
        ].get_prepared_layer_range_meta([0], 0)
        self.assertEqual(sizes, [[7, 17]])
        self.assertEqual(offsets, [[0, 18]])
        _, sizes, offsets = group.entry_map[
            PoolName.INDEXER
        ].get_prepared_layer_range_meta([0], 1)
        self.assertEqual(sizes, [[11, 23]])
        self.assertEqual(offsets, [[7, 35]])

    def _mamba_group(
        self,
        *,
        dcp_size=1,
        page_size=2,
        kv_scale=None,
        unified_memory=False,
        draft=None,
    ):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        # Kimi K3 shape: one MLA layer (1) between two KDA layers (0, 2).
        kvcache = HybridLinearKVPool.__new__(HybridLinearKVPool)
        kvcache.start_layer = 0
        kvcache.full_attention_layer_id_mapping = {1: 0}
        kvcache.full_kv_pool = SimpleNamespace(
            kv_buffer=[torch.zeros((18, 3), dtype=torch.uint8)],
            kv_scale_buffer=kv_scale,
        )
        mamba_cache = SimpleNamespace(
            conv=[torch.zeros((2, 6, 7), dtype=torch.uint8)],
            temporal=torch.zeros((2, 6, 5), dtype=torch.uint8),
        )
        params = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                mamba_ckpt_pool=None,
                short_conv_pool=SimpleNamespace(enabled=False),
                ngram_pool=SimpleNamespace(enabled=False),
                mamba_map={0: 0, 2: 1},
                mamba_pool=SimpleNamespace(num_mamba_layers=2, mamba_cache=mamba_cache),
            ),
            mtp_draft_device_pools=(draft,) if draft else (),
        )
        with (
            get_parallel().override(attn_dcp_size=dcp_size),
            patch(
                "sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler.get_memory",
                return_value=SimpleNamespace(enable_unified_memory=unified_memory),
            ),
        ):
            return resolve_hybrid_device_pool_group(
                kvcache=kvcache,
                page_size=page_size,
                params=params,
                components={ComponentType.FULL, ComponentType.MAMBA},
            )

    def test_mamba_packs_mla_kv_and_slot_addressed_state(self):
        group = self._mamba_group()
        self.assertEqual(group.num_layers, 3)
        self.assertFalse(group.rank_replicated)
        self.assertTrue(group.kv_rank_replicated)
        kv = group.entry_map[PoolName.KV]
        self.assertEqual(kv.prepare_locations(torch.arange(4, 6)), [4])
        self.assertIsNone(kv.get_prepared_layer_range_meta([4], 0))
        # One object per slot: every layer's temporal state, then its conv state.
        mamba = group.entry_map[PoolName.MAMBA]
        self.assertEqual(mamba.prepare_locations(torch.tensor([3])), [3])
        _, sizes, offsets = mamba.get_prepared_layer_range_meta([3], 2)
        self.assertEqual((sizes, offsets), ([[5, 7]], [[5, 17]]))

    def test_mamba_collapses_only_the_kv_under_dcp(self):
        group = self._mamba_group(dcp_size=2, page_size=4)
        self.assertFalse(group.kv_rank_replicated)
        kv, mamba = group.entry_map[PoolName.KV], group.entry_map[PoolName.MAMBA]
        self.assertEqual(kv.prepare_locations(torch.arange(8, 12)), [4])
        self.assertEqual(mamba.prepare_locations(torch.tensor([5])), [5])

    def test_mamba_carries_a_dspark_draft(self):
        for dcp_size in (1, 2):
            draft = SimpleNamespace(
                page_size=2 * dcp_size,
                k_buffer=[torch.zeros((18, 4), dtype=torch.uint8)],
                v_buffer=[torch.zeros((18, 4), dtype=torch.uint8)],
                k_scale_buffer=None,
                native_k_scale_buffer=None,
            )
            with self.subTest(dcp_size=dcp_size):
                group = self._mamba_group(
                    dcp_size=dcp_size, page_size=2 * dcp_size, draft=draft
                )
                self.assertEqual(group.sources[PoolName.DRAFT], PoolName.KV)
                entry = group.entry_map[PoolName.DRAFT]
                # Draft layer 0 is loaded with the first full-attention layer.
                self.assertEqual(entry.layer_mapping, {1: 0})
                # The draft is not sharded: its rows are the widened locs.
                locs = torch.arange(8, 8 + 2 * dcp_size)
                self.assertEqual(entry.prepare_locations(locs), [8])

    def test_linker_packs_a_dspark_draft_for_hybrid_linear_targets(self):
        from sglang.srt.mem_cache.memory_pool import (
            HybridLinearKVPool,
            MHATokenToKVPool,
        )
        from sglang.srt.speculative import base_spec_worker as spec
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        draft = SimpleNamespace(
            token_to_kv_pool=MHATokenToKVPool.__new__(MHATokenToKVPool),
            model_config=SimpleNamespace(
                num_nextn_predict_layers=0,
                hf_config=SimpleNamespace(architectures=["DSparkDraftModel"]),
            ),
        )
        target = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            token_to_kv_pool=HybridLinearKVPool.__new__(HybridLinearKVPool),
        )
        worker = SimpleNamespace(
            target_worker=SimpleNamespace(model_runner=target),
            _draft_model_runners=lambda: (draft,),
        )
        with patch.object(
            spec,
            "get_memory",
            return_value=SimpleNamespace(
                enable_hierarchical_cache=False,
                enable_unified_cache_external_linker=True,
            ),
        ):
            plan = spec.BaseSpecWorker._build_hicache_draft_plan(worker)
        self.assertEqual(plan.mode, spec.HiCacheDraftMode.PACKED)
        self.assertEqual(target.mtp_draft_device_pools, (draft.token_to_kv_pool,))

    def test_mamba_rejects_unsupported_configs(self):
        with self.assertRaisesRegex(ValueError, "--enable-unified-memory"):
            self._mamba_group(unified_memory=True)
        with self.assertRaisesRegex(ValueError, "scaled KV caches"):
            self._mamba_group(kv_scale=[torch.zeros((18, 1))])

    def test_dsa_rejects_dcp(self):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        kvcache = DSATokenToKVPool.__new__(DSATokenToKVPool)
        kvcache.page_size = 2
        with get_parallel().override(attn_dcp_size=8):
            with self.assertRaisesRegex(ValueError, "does not support --dcp-size"):
                resolve_hybrid_device_pool_group(
                    kvcache=kvcache,
                    page_size=2,
                    params=SimpleNamespace(mtp_draft_device_pools=()),
                    components={ComponentType.FULL},
                )

    def test_linker_requires_packed_draft(self):
        """Do not accept draft state that the linker would omit from storage."""
        from sglang.srt.speculative import base_spec_worker as spec
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        draft = SimpleNamespace(
            token_to_kv_pool=object(),
            req_to_token_pool=None,
            model_config=SimpleNamespace(
                num_nextn_predict_layers=0,
                hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            ),
        )
        target = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.EAGLE, req_to_token_pool=None
        )
        worker = SimpleNamespace(
            target_worker=SimpleNamespace(model_runner=target),
            _draft_model_runners=lambda: (draft,),
        )
        for linker_enabled, nextn_layers in (
            (False, 0),
            (True, 0),
            (False, 1),
            (True, 1),
        ):
            draft.model_config.num_nextn_predict_layers = nextn_layers
            with (
                self.subTest(linker=linker_enabled, nextn=nextn_layers),
                patch.object(
                    spec,
                    "get_memory",
                    return_value=SimpleNamespace(
                        enable_hierarchical_cache=not linker_enabled,
                        enable_unified_cache_external_linker=linker_enabled,
                    ),
                ),
            ):
                if linker_enabled and not nextn_layers:
                    with self.assertRaisesRegex(
                        NotImplementedError, "only supports packed"
                    ):
                        spec.BaseSpecWorker._build_hicache_draft_plan(worker)
                    self.assertEqual(target.mtp_draft_device_pools, ())
                else:
                    plan = spec.BaseSpecWorker._build_hicache_draft_plan(worker)
                    self.assertEqual(
                        plan.mode,
                        spec.HiCacheDraftMode.PACKED
                        if nextn_layers
                        else spec.HiCacheDraftMode.SIDECAR,
                    )
                    self.assertEqual(plan.device_pools, (draft.token_to_kv_pool,))
                    self.assertEqual(
                        target.mtp_draft_device_pools,
                        plan.device_pools if nextn_layers else (),
                    )

    def test_unsupported_strategy_fails_with_context(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        kvcache = SWAKVPool.__new__(SWAKVPool)
        with self.assertRaisesRegex(
            ValueError,
            "does not support the direct external linker: _SwaStrategy",
        ):
            resolve_hybrid_device_pool_group(
                kvcache=kvcache,
                page_size=2,
                params=SimpleNamespace(),
                components={ComponentType.FULL, ComponentType.SWA},
            )


if __name__ == "__main__":
    unittest.main()
