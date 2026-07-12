"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sglang.srt.utils import add_prefix

# Adapted from
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
"""Inference-only LLaMA-EAGLE model compatible with HuggingFace weights."""

import copy
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP


def expected_qwen3_eagle3_weight_names(num_hidden_layers: int) -> set[str]:
    expected = {
        "embed_tokens.weight",
        "fc.weight",
        "norm.weight",
        "lm_head.weight",
    }
    for layer_id in range(int(num_hidden_layers)):
        prefix = f"layers.{layer_id}"
        expected.update(
            {
                f"{prefix}.hidden_norm.weight",
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            }
        )
    return expected


class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix)

        # override qkv
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        if config.model_type == "llama4_text":
            inter_size = config.intermediate_size_mlp
        else:
            inter_size = config.intermediate_size

        self.mlp = LlamaMLP(
            config.hidden_size, inter_size, config.hidden_act, quant_config, prefix
        )

        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_scaling = rope_parameters
        else:
            rope_scaling = getattr(config, "rope_scaling", None)
        self.is_mrope_enabled = (
            rope_scaling is not None and "mrope_section" in rope_scaling
        )
        # fix rope_scaling for qwen2.5-vl
        if self.is_mrope_enabled:
            rope_scaling["rope_type"] = "default"

        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self.fc = torch.nn.Linear(
            self.hidden_size_in * 3,
            config.hidden_size,
            bias=getattr(config, "bias", False),
        )

        self.midlayer = LlamaDecoderLayer(config, 0, quant_config, prefix)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            embeds = forward_batch.mm_input_embeds
            if (
                forward_batch.forward_mode.is_extend()
                and forward_batch.contains_mm_inputs()
                and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
            ):
                assert embeds is not None
                embeds = torch.cat(
                    [embeds[:-1], self.embed_tokens(input_ids[-1].unsqueeze(0))]
                )
            if embeds is None:
                embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        hidden_states = forward_batch.spec_info.hidden_states
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        # idle batch
        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        residual = None
        hidden_states, residual = self.midlayer(
            positions,
            embeds,
            hidden_states,
            forward_batch,
            residual,
        )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        # For draft decode, we capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]


class LlamaForCausalLMEagle3(LlamaForCausalLM):
    eagle3_backbone_cls = LlamaModel
    strict_weight_coverage = False

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        if self.config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")

        self.model = self.eagle3_backbone_cls(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        self.load_lm_head_from_target = False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            draft_vocab_size = getattr(config, "draft_vocab_size", None)
            if draft_vocab_size is None:
                self.load_lm_head_from_target = True
                config.draft_vocab_size = config.vocab_size
                draft_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        config_ = copy.deepcopy(config)
        # The draft logits processor has its own vocab size. Older checkpoints
        # may omit draft_vocab_size when it is identical to the target vocab.
        draft_logits_vocab_size = getattr(config_, "draft_vocab_size", None)
        config_.vocab_size = (
            int(draft_logits_vocab_size)
            if draft_logits_vocab_size is not None
            else int(config_.vocab_size)
        )
        self.logits_processor = LogitsProcessor(config_)

        self.capture_aux_hidden_states = True
        self.hot_token_id = None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        consumed_checkpoint_weights = set()
        unexpected_checkpoint_weights = set()

        def canonical_checkpoint_name(name: str) -> str:
            return name[len("model.") :] if name.startswith("model.") else name

        # Define the parameter mapping for stacked parameters
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            checkpoint_name = name
            if "d2t" in name:
                # d2t stores diffs between draft id and target id
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue

            if "t2d" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param_name = f"model.{name}" if name not in params_dict else name
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(param_name)
                    consumed_checkpoint_weights.add(
                        canonical_checkpoint_name(checkpoint_name)
                    )
                else:
                    unexpected_checkpoint_weights.add(
                        canonical_checkpoint_name(checkpoint_name)
                    )
                break
            else:
                # Handle regular parameters
                param_name = name if name in params_dict else f"model.{name}"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(param_name)
                    consumed_checkpoint_weights.add(
                        canonical_checkpoint_name(checkpoint_name)
                    )
                else:
                    unexpected_checkpoint_weights.add(
                        canonical_checkpoint_name(checkpoint_name)
                    )

        if self.strict_weight_coverage:
            expected_checkpoint_weights = expected_qwen3_eagle3_weight_names(
                len(self.model.layers)
            )
            missing = expected_checkpoint_weights - consumed_checkpoint_weights
            if missing or unexpected_checkpoint_weights:
                raise ValueError(
                    "Qwen3Eagle3Model checkpoint weight coverage failed: "
                    f"missing_core={sorted(missing)}, "
                    f"unexpected={sorted(unexpected_checkpoint_weights)}."
                )
        return loaded_params

    def get_hot_token_id(self):
        return self.hot_token_id


EntryClass = [LlamaForCausalLMEagle3]
