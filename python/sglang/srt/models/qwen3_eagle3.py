"""SGLang-native DeepSpec Qwen3 EAGLE3 draft model."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.llama import LlamaMLP
from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.utils import add_prefix
from sglang.srt.utils.hf_transformers.common import get_rope_config


class Qwen3Eagle3Attention(nn.Module):
    """Qwen3 attention over concatenated token and draft hidden features."""

    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        input_size = 2 * hidden_size
        tp_size = int(get_tensor_model_parallel_world_size())
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                "Qwen3 EAGLE3 attention heads must be divisible by TP size: "
                f"num_heads={self.total_num_heads}, tp_size={tp_size}."
            )
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size != 0:
                raise ValueError(
                    "Qwen3 EAGLE3 KV heads must be divisible by TP size when "
                    f"num_kv_heads >= tp_size: num_kv_heads={self.total_num_kv_heads}, "
                    f"tp_size={tp_size}."
                )
        elif tp_size % self.total_num_kv_heads != 0:
            raise ValueError(
                "Qwen3 EAGLE3 TP size must be divisible by replicated KV heads: "
                f"num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )

        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(
            getattr(config, "head_dim", hidden_size // self.total_num_heads)
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        attention_bias = bool(getattr(config, "attention_bias", False))
        self.qkv_proj = QKVParallelLinear(
            input_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        # Keep the row-parallel reduction enabled. Unlike the regular Qwen3
        # decoder, this layer is not wrapped by LayerCommunicator.
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            reduce_results=True,
            prefix=add_prefix("o_proj", prefix),
        )

        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        rope_theta, rope_scaling = get_rope_config(config)
        self.rope_theta = float(rope_theta)
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        rope_is_neox_style = bool(getattr(config, "rope_is_neox_style", True))
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3Eagle3DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.self_attn = Qwen3Eagle3Attention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=int(config.intermediate_size),
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            reduce_results=True,
        )
        self.hidden_norm = RMSNorm(self.hidden_size, eps=eps)
        self.input_layernorm = RMSNorm(self.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del residual
        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Eagle3Backbone(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = int(config.hidden_size)
        self.target_hidden_size = int(
            getattr(config, "target_hidden_size", self.hidden_size)
        )
        self.target_layer_ids = [int(x) for x in config.target_layer_ids]
        if not self.target_layer_ids:
            raise ValueError("Qwen3Eagle3Model requires non-empty target_layer_ids.")

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.fc = nn.Linear(
            self.target_hidden_size * len(self.target_layer_ids),
            self.hidden_size,
            bias=False,
        )
        self.layers = nn.ModuleList(
            [
                Qwen3Eagle3DecoderLayer(
                    config=config,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(int(config.num_hidden_layers))
            ]
        )
        self.norm = RMSNorm(
            self.hidden_size, eps=float(getattr(config, "rms_norm_eps", 1e-6))
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        del pp_proxy_tensors
        if input_embeds is None:
            embeds = forward_batch.mm_input_embeds
            if (
                forward_batch.forward_mode.is_extend()
                and forward_batch.contains_mm_inputs()
                and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
            ):
                if embeds is None:
                    raise RuntimeError(
                        "Qwen3 EAGLE3 multimodal extend expected mm_input_embeds."
                    )
                embeds = torch.cat(
                    [embeds[:-1], self.embed_tokens(input_ids[-1].unsqueeze(0))]
                )
            if embeds is None:
                embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        hidden_states = forward_batch.spec_info.hidden_states
        if int(hidden_states.shape[-1]) == int(self.fc.in_features):
            hidden_states = self.fc(hidden_states)
        elif int(hidden_states.shape[-1]) != self.hidden_size:
            raise ValueError(
                "Qwen3 EAGLE3 auxiliary hidden width mismatch: expected either "
                f"{self.fc.in_features} captured features or {self.hidden_size} "
                f"projected features, got shape={tuple(hidden_states.shape)}."
            )

        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                embeds,
                hidden_states,
                forward_batch,
                residual,
            )
        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )
        return hidden_states_to_logits, [hidden_states_to_aux]


class Qwen3Eagle3Model(LlamaForCausalLMEagle3):
    """Entry class for unmodified DeepSpec ``Qwen3Eagle3Model`` configs."""

    eagle3_backbone_cls = Qwen3Eagle3Backbone
    strict_weight_coverage = True


EntryClass = [Qwen3Eagle3Model]
