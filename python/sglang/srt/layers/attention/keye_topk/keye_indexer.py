"""KeyeIndexer for KeyeTopKMask Sparse Attention

Computes sparse attention indices using a lightweight FP8 indexer network:
    I_{t,s} = sum_{j=1}^{H^I} w^I_{t,j} * ReLU(q^I_{t,j} * k^I_s)

Current implementation:
  KeyeIndexer — unified FP8 path:
      projections/norms/RoPE/Hadamard → act_quant → FP8 paged cache → deep_gemm kernels
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.layers.attention.nsa.nsa_indexer import BaseIndexerMetadata
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import add_prefix, ceil_align, is_cuda, is_hip

if is_cuda():
    try:
        import deep_gemm
    except ImportError as e:
        deep_gemm = e

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KeyeTokenToKVPool

_is_cuda = is_cuda()
_is_hip = is_hip()

logger = logging.getLogger(__name__)

DUAL_STREAM_TOKEN_THRESHOLD = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Hadamard transform."""
    # Upstream moved hadamard_transform out of sgl_kernel; mirror nsa_indexer.rotate_activation:
    # HIP -> fast_hadamard_transform (AMD), CUDA -> sglang.jit_kernel.hadamard.
    if _is_hip:
        from fast_hadamard_transform import hadamard_transform
    else:
        from sglang.jit_kernel.hadamard import hadamard_transform

    hidden_size = x.size(-1)
    assert (hidden_size & (hidden_size - 1)) == 0, (
        f"Hidden size ({hidden_size}) must be a power of 2 for Hadamard transform."
    )
    assert x.dtype == torch.bfloat16
    return hadamard_transform(x, scale=hidden_size**-0.5)


# ---------------------------------------------------------------------------
# KeyeIndexer — unified FP8 indexer implementation
# ---------------------------------------------------------------------------


class KeyeIndexer(MultiPlatformOp):
    """FP8-accelerated Keye Top-K sparse attention indexer (MQA).

    Responsibilities:
      - Build lightweight indexer Q/K/gate activations.
      - Apply norm + MRoPE + Hadamard transform.
      - Quantise Q/K to FP8 and write K + scale to the index cache.
      - Compute top-k indices for prefill/decode via deep_gemm kernels.
    """

    _fp8_logged: bool = False

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        topk: int,
        mrope_section: List[int],
        rope_theta: float = 8000000.0,
        main_head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        scale_fmt: Optional[str] = "ue8m0",
        block_size: int = 128,
        layer_id: int = 0,
        alt_stream: Optional[torch.cuda.Stream] = None,
        quant_config: Optional[object] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.topk = topk
        self.layer_id = layer_id
        self.total_q_dim = num_heads * head_dim
        self.softmax_scale = head_dim**-0.5
        self.alt_stream = alt_stream

        assert main_head_dim is not None, (
            "main_head_dim is required for mrope_section scaling"
        )
        scaling_factor = self.head_dim / main_head_dim
        self.indexer_mrope_section = [int(x * scaling_factor) for x in mrope_section]
        assert sum(self.indexer_mrope_section) * 2 == head_dim, (
            f"Indexer mrope section {self.indexer_mrope_section} sum*2 != head_dim {head_dim}"
        )

        # Use MRotaryEmbedding (via get_rope_wrapper) to align with SGLang framework style.
        # Provides a pre-computed cos_sin_cache (O(N) lookup vs recompute each forward)
        # and dispatches to the Triton MRoPE kernel on CUDA.
        self.rotary_emb = get_rope_wrapper(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            is_neox_style=True,
            rope_scaling={
                "rope_type": "default",
                "mrope_section": self.indexer_mrope_section,
            },
            dtype=torch.bfloat16,
            device=get_global_server_args().device,
        )

        self.scale_fmt = scale_fmt
        self.block_size = block_size

        self.q_proj = ReplicatedLinear(
            hidden_size,
            self.total_q_dim + num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = ReplicatedLinear(
            hidden_size,
            head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("k_proj", prefix),
        )

        from sglang.srt.layers.layernorm import LayerNorm, RMSNorm

        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6, dtype=torch.float32)

        if not _is_cuda or isinstance(deep_gemm, Exception):
            raise RuntimeError(
                "KeyeIndexer requires CUDA and deep_gemm. "
                "Make sure deep_gemm is installed."
            )

        self.sm_count = deep_gemm.get_num_sms()
        self.half_device_sm_count = ceil_align(self.sm_count // 2, 8)

        assert self.head_dim in (32, 64, 128), (
            f"KeyeIndexer head_dim={self.head_dim} must be 32, 64, or 128"
        )
        self.block_size = min(self.head_dim, self.block_size)

        if not KeyeIndexer._fp8_logged:
            KeyeIndexer._fp8_logged = True
            logger.info(
                "KeyeIndexer: using FP8 path "
                f"(head_dim={self.head_dim}, num_heads={self.num_heads}, topk={self.topk}, "
                f"block_size={self.block_size}, scale_fmt={self.scale_fmt!r}, "
                f"sm_count={self.sm_count})"
            )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute only bf16 K (skip Q and gate-w). Used by the skip_topk fast path."""
        key, _ = self.k_proj(x)
        key = self.k_norm(key)

        # MRotaryEmbedding.forward expects positions [3, N] and tensors [N, D].
        # Use key as a dummy query; only the returned key is used.
        if positions.dim() == 1:
            positions_2d = positions.unsqueeze(0).expand(3, -1)  # [3, N]
        else:
            positions_2d = positions  # already [3, N]

        if key.dtype != torch.bfloat16:
            key = key.to(torch.bfloat16)
        _, key = self.rotary_emb(positions_2d, key, key)
        return _rotate_activation(key)

    def _get_q_k_w_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute bf16 Q [N, H, D], K [N, D] (single MQA head), and gate-w [N, H].

        Steps: linear projection → reshape + norm → MRoPE → Hadamard transform.
        """

        total_tokens = x.shape[0]
        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q_and_w, _ = self.q_proj(x)
            query = q_and_w[..., : self.total_q_dim]
            w_raw = q_and_w[..., self.total_q_dim :]
            query = query.view(total_tokens, self.num_heads, self.head_dim)
            query = self.q_norm(query.reshape(-1, self.head_dim)).view(
                total_tokens, self.num_heads, self.head_dim
            )
            with torch.cuda.stream(self.alt_stream):
                key, _ = self.k_proj(x)
                key = self.k_norm(key)
            current_stream.wait_stream(self.alt_stream)
        else:
            q_and_w, _ = self.q_proj(x)
            key, _ = self.k_proj(x)
            query = q_and_w[..., : self.total_q_dim]
            w_raw = q_and_w[..., self.total_q_dim :]
            query = query.view(total_tokens, self.num_heads, self.head_dim)
            query = self.q_norm(query.reshape(-1, self.head_dim)).view(
                total_tokens, self.num_heads, self.head_dim
            )
            key = self.k_norm(key)

        # MRotaryEmbedding.forward expects:
        #   positions: [3, N]  (2D MRoPE)
        #   query:     [N, num_heads * head_dim]  (flattened)
        #   key:       [N, head_dim]
        # and returns the already-rotated (query, key) in the same shapes.
        if positions.dim() == 1:
            positions_2d = positions.unsqueeze(0).expand(3, -1)  # [3, N]
        else:
            positions_2d = positions  # already [3, N]

        if query.dtype != torch.bfloat16:
            query = query.to(torch.bfloat16)
        if key.dtype != torch.bfloat16:
            key = key.to(torch.bfloat16)

        q_2d = query.view(total_tokens, self.total_q_dim)  # [N, H*D]
        q_2d, key = self.rotary_emb(positions_2d, q_2d, key)
        query = q_2d.view(total_tokens, self.num_heads, self.head_dim)  # [N, H, D]

        # Hadamard transform (applied after RoPE, same as before)
        q_flat = query.reshape(total_tokens * self.num_heads, self.head_dim)

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query = _rotate_activation(q_flat).view(total_tokens, self.num_heads, self.head_dim)
            with torch.cuda.stream(self.alt_stream):
                key = _rotate_activation(key)
            current_stream.wait_stream(self.alt_stream)
        else:
            query = _rotate_activation(q_flat).view(total_tokens, self.num_heads, self.head_dim)
            key = _rotate_activation(key)

        return query, key, w_raw

    def _should_chunk_mqa_logits(
        self, num_q: int, num_k: int, device: torch.device
    ) -> Tuple[bool, int]:
        """Return (need_chunk, free_mem_bytes). Chunks large logits to avoid OOM."""
        if num_q * num_k < 8_000_000:
            return False, 0
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        logits_bytes = num_q * num_k * 4  # float32
        need_chunk = (logits_bytes * 2 > free_mem) or (logits_bytes > total_mem * 0.3)
        return need_chunk, free_mem

    @torch.compile(dynamic=True)
    def _get_logits_head_gate(self, w_raw: torch.Tensor, q_scale: torch.Tensor):
        """Fold per-head FP8 scale into gate weights: weights[t,h] = w[t,h] * q_scale[t,h] * softmax_scale."""
        w = w_raw.float()
        weights = w.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights.squeeze(-1)

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward_cuda(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        """Run the unified FP8 indexer and return top-k indices."""
        if not return_indices:
            return None
        if not hasattr(forward_batch.token_to_kv_pool, "index_head_dim"):
            return None
        if x.shape[0] == 0:
            return torch.full((0, self.topk), -1, dtype=torch.int32, device=x.device)

        metadata: Optional[
            BaseIndexerMetadata
        ] = forward_batch.attn_backend.get_indexer_metadata(layer_id, forward_batch)
        if metadata is None:
            return None

        total_tokens = x.shape[0]
        pool = forward_batch.token_to_kv_pool
        from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

        loc = forward_batch.out_cache_loc
        if not loc.is_contiguous():
            loc = loc.contiguous()
        if loc.dtype != torch.int64:
            loc = loc.to(torch.int64)

        # Fast path: when max_kv_len <= topk every position is selected; skip logits.
        if (
            forward_batch.forward_mode.is_extend_without_speculative()
            and forward_batch.seq_lens_cpu is not None
            and len(forward_batch.seq_lens_cpu) > 0
        ):
            max_kv_len = int(forward_batch.seq_lens_cpu.max().item())
            if max_kv_len <= self.topk:
                key = self._get_k_bf16(x, positions)
                k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)
                pool.set_index_k_scale_buffer(
                    layer_id=layer_id,
                    loc=loc,
                    index_k=k_fp8,
                    index_k_scale=k_scale,
                )

                device = x.device
                seq_lens_cpu = forward_batch.seq_lens_cpu
                extend_lens_cpu = forward_batch.extend_seq_lens_cpu
                result = torch.full(
                    (total_tokens, self.topk), -1, dtype=torch.int32, device=device
                )
                q_offset = 0
                for i in range(forward_batch.batch_size):
                    seq_len = int(seq_lens_cpu[i].item())
                    extend_len = int(extend_lens_cpu[i])
                    history = seq_len - extend_len
                    if extend_len > 0:
                        col_idx = torch.arange(max_kv_len, dtype=torch.int32, device=device)
                        valid_counts = torch.arange(
                            history + 1,
                            history + extend_len + 1,
                            dtype=torch.int32,
                            device=device,
                        )
                        mask = col_idx.unsqueeze(0) < valid_counts.unsqueeze(1)
                        result[q_offset : q_offset + extend_len, :max_kv_len] = torch.where(
                            mask,
                            col_idx.unsqueeze(0).expand(extend_len, -1),
                            torch.full_like(
                                col_idx.unsqueeze(0).expand(extend_len, -1), -1
                            ),
                        )
                    q_offset += extend_len
                return result

        enable_dual_stream = (
            self.alt_stream is not None
            and get_is_capture_mode()
            and 0 < x.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
        )

        if enable_dual_stream and forward_batch.forward_mode.is_decode_or_idle():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query, key, w_raw = self._get_q_k_w_bf16(x, positions, enable_dual_stream)
            q_fp8_2d, q_scale_2d = act_quant(
                query.reshape(-1, self.head_dim), self.block_size, self.scale_fmt
            )
            with torch.cuda.stream(self.alt_stream):
                k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)
                pool.set_index_k_scale_buffer(
                    layer_id=layer_id,
                    loc=loc,
                    index_k=k_fp8,
                    index_k_scale=k_scale,
                )
            current_stream.wait_stream(self.alt_stream)
        else:
            query, key, w_raw = self._get_q_k_w_bf16(x, positions, enable_dual_stream)
            q_fp8_2d, q_scale_2d = act_quant(
                query.reshape(-1, self.head_dim), self.block_size, self.scale_fmt
            )
            k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)
            pool.set_index_k_scale_buffer(
                layer_id=layer_id,
                loc=loc,
                index_k=k_fp8,
                index_k_scale=k_scale,
            )

        q_fp8 = q_fp8_2d.view(total_tokens, self.num_heads, self.head_dim)
        q_scale = q_scale_2d.view(total_tokens, self.num_heads, 1)
        weights = self._get_logits_head_gate(w_raw, q_scale)

        if forward_batch.forward_mode.is_extend_without_speculative():
            return self._get_topk_ragged(forward_batch, layer_id, q_fp8, weights, metadata)
        return self._get_topk_paged(forward_batch, layer_id, q_fp8, weights, metadata)

    def _get_topk_ragged(
        self,
        forward_batch,
        layer_id,
        q_fp8,
        weights,
        metadata: Optional[BaseIndexerMetadata] = None,
    ):
        """Prefill top-k via deep_gemm.fp8_mqa_logits (ragged batch)."""
        assert forward_batch.forward_mode.is_extend_without_speculative()
        assert forward_batch.seq_lens_cpu is not None
        assert forward_batch.extend_seq_lens_cpu is not None

        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, KeyeTokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page_size=64"

        block_tables_raw = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :
        ]
        strided = torch.arange(0, block_tables_raw.shape[-1], page_size, device="cuda")
        block_tables = block_tables_raw[:, strided] // page_size

        k_fp8_list, k_scale_list, ks_list, ke_list = [], [], [], []
        q_offset = 0
        k_offset = 0

        for i in range(forward_batch.batch_size):
            seq_len = int(forward_batch.seq_lens_cpu[i].item())
            extend_len = int(forward_batch.extend_seq_lens_cpu[i])
            history_len = seq_len - extend_len

            k_fp8_i, k_scale_i = (
                forward_batch.token_to_kv_pool.get_index_k_scale_buffer(
                    layer_id, seq_len, block_tables[i]
                )
            )
            k_fp8_list.append(k_fp8_i)
            k_scale_list.append(k_scale_i)

            ks_list.append(
                torch.full((extend_len,), k_offset, dtype=torch.int32, device="cuda")
            )
            ke_list.append(
                k_offset
                + history_len
                + torch.arange(1, extend_len + 1, dtype=torch.int32, device="cuda")
            )
            q_offset += extend_len
            k_offset += seq_len

        k_fp8_cat = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
        k_scale_cat = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
        kv_fp8 = (k_fp8_cat, k_scale_cat)
        ks = torch.cat(ks_list, dim=0)
        ke = torch.cat(ke_list, dim=0)

        total_tokens = q_fp8.shape[0]
        device = q_fp8.device

        block_q = 128 // self.num_heads
        padded_q_offset = (q_offset + block_q - 1) // block_q * block_q
        need_pad = padded_q_offset > q_offset

        def _pad_to(t: torch.Tensor, target_len: int) -> torch.Tensor:
            if t.shape[0] >= target_len:
                return t
            pad_shape = (target_len - t.shape[0],) + t.shape[1:]
            return torch.cat(
                [
                    t,
                    torch.zeros(pad_shape, dtype=t.dtype, device=t.device),
                ],
                dim=0,
            )

        need_chunk, free_mem = self._should_chunk_mqa_logits(
            padded_q_offset, k_offset, device
        )

        if not need_chunk:
            q_in = _pad_to(q_fp8[:q_offset], padded_q_offset) if need_pad else q_fp8[:q_offset]
            w_in = _pad_to(weights[:q_offset], padded_q_offset) if need_pad else weights[:q_offset]
            ks_in = _pad_to(ks, padded_q_offset) if need_pad else ks
            ke_in = _pad_to(ke, padded_q_offset) if need_pad else ke
            logits = deep_gemm.fp8_mqa_logits(
                q_in, kv_fp8, w_in, ks_in, ke_in, clean_logits=False
            )
            raw_result = metadata.topk_transform(
                logits[:q_offset], self.topk, forward_batch, ks=ks, context_length=k_offset,
            )
            return raw_result

        # Chunk to avoid OOM on large sequences
        bytes_per_row = k_offset * 4
        max_rows = max(block_q, int((free_mem * 0.25) // max(bytes_per_row, 1)))
        max_rows = (min(max_rows, padded_q_offset) // block_q) * block_q
        max_rows = max(max_rows, block_q)

        topk_result = torch.full(
            (total_tokens, self.topk), -1, device=device, dtype=torch.int32
        )
        start = 0
        while start < q_offset:
            end_real = min(start + max_rows, q_offset)
            end_pad = (end_real - start + block_q - 1) // block_q * block_q + start
            chunk_need_pad = end_pad > end_real

            q_chunk = (
                _pad_to(q_fp8[start:end_real], end_pad - start)
                if chunk_need_pad
                else q_fp8[start:end_real]
            )
            w_chunk = (
                _pad_to(weights[start:end_real], end_pad - start)
                if chunk_need_pad
                else weights[start:end_real]
            )
            ks_chunk = (
                _pad_to(ks[start:end_real], end_pad - start)
                if chunk_need_pad
                else ks[start:end_real]
            )
            ke_chunk = (
                _pad_to(ke[start:end_real], end_pad - start)
                if chunk_need_pad
                else ke[start:end_real]
            )

            logits_chunk = deep_gemm.fp8_mqa_logits(
                q_chunk,
                kv_fp8,
                w_chunk,
                ks_chunk,
                ke_chunk,
                clean_logits=False,
            )
            real_len = end_real - start
            logits_real = logits_chunk[:real_len]
            topk_result[start:end_real] = metadata.topk_transform(
                logits_real, self.topk,
                forward_batch,
                ks=ks[start:end_real],
                ke_offset=ke[start:end_real] - ks[start:end_real],
                context_length=k_offset,
            )
            start = end_real

        return topk_result

    def _get_topk_paged(
        self,
        forward_batch,
        layer_id,
        q_fp8,
        weights,
        metadata: Optional[BaseIndexerMetadata],
    ):
        """Decode top-k via deep_gemm.fp8_paged_mqa_logits."""
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, KeyeTokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page_size=64"

        seqlens_32 = metadata.get_seqlens_int32()
        block_tables = metadata.get_page_table_64()

        max_seq_len = block_tables.shape[1] * page_size
        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )

        schedule_metadata = getattr(metadata, "paged_mqa_schedule_metadata", None)
        if schedule_metadata is None:
            schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                seqlens_32, page_size, self.sm_count
            )

        head_dim_with_sf = self.head_dim + self.head_dim // self.block_size * 4
        q_4d = q_fp8.unsqueeze(1)
        kv_4d = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], page_size, 1, head_dim_with_sf
        )

        logits = deep_gemm.fp8_paged_mqa_logits(
            q_4d,
            kv_4d,
            weights,
            seqlens_32,
            block_tables,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )

        return metadata.topk_transform(logits, self.topk, forward_batch, ks=None, context_length=max_seq_len)
