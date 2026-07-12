import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.models.dflash import DFlashDraftModel, Qwen3DSparkModel
from sglang.srt.models.llama_eagle3 import (
    LlamaForCausalLMEagle3,
    expected_qwen3_eagle3_weight_names,
)
from sglang.srt.models.qwen3_eagle3 import Qwen3Eagle3Model
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_utils import (
    DFLASH_NEXT_TOKEN,
    DFLASH_SAME_POSITION,
    expected_dflash_core_weight_names,
    parse_dflash_draft_config,
)
from sglang.srt.speculative.eagle_config import (
    resolve_eagle3_aux_hidden_size,
    resolve_eagle3_aux_layer_ids,
)
from sglang.srt.utils.hf_transformers.common import get_rope_config


WORKSPACE = Path(os.getenv("DEEPSPEC_COMPAT_WORKSPACE", "/mmu_mllm_hdd_3/renjunchi"))


class TestDFlashLayouts(unittest.TestCase):
    def test_synthetic_layout_alignment_for_all_required_block_sizes(self):
        for block_size in (2, 4, 7, 8, 16):
            proposal_count = block_size - 1
            for layout, expected_draft_input_size in (
                (DFLASH_SAME_POSITION, block_size),
                (DFLASH_NEXT_TOKEN, proposal_count),
            ):
                hidden = torch.arange(
                    2 * expected_draft_input_size * 3, dtype=torch.float32
                ).view(2, expected_draft_input_size, 3)
                proposals = layout.select_proposal_hidden(
                    hidden, external_block_size=block_size
                )
                self.assertEqual(
                    layout.draft_input_size(block_size), expected_draft_input_size
                )
                self.assertEqual(tuple(proposals.shape), (2, proposal_count, 3))
                expected = hidden if layout.next_token_aligned else hidden[:, 1:]
                torch.testing.assert_close(proposals, expected)

                prefix = torch.tensor([11, 29], dtype=torch.int64)
                draft_positions = prefix[:, None] + torch.arange(
                    expected_draft_input_size
                )
                target_positions = prefix[:, None] + torch.arange(block_size)
                self.assertEqual(
                    tuple(draft_positions.shape), (2, expected_draft_input_size)
                )
                self.assertEqual(tuple(target_positions.shape), (2, block_size))
                self.assertEqual(2 * expected_draft_input_size, draft_positions.numel())
                self.assertEqual(2 * block_size, target_positions.numel())

    def test_deepspec_examples(self):
        self.assertEqual(DFLASH_NEXT_TOKEN.draft_input_size(8), 7)
        self.assertEqual(DFLASH_NEXT_TOKEN.proposal_count(8), 7)
        self.assertEqual(DFLASH_NEXT_TOKEN.draft_input_size(4), 3)
        self.assertEqual(DFLASH_NEXT_TOKEN.proposal_count(4), 3)
        with self.assertRaisesRegex(ValueError, "B >= 2"):
            DFLASH_NEXT_TOKEN.draft_input_size(1)

    def test_default_block_size_and_top_level_mask_fallback(self):
        deep_config = {
            "architectures": ["Qwen3DSparkModel"],
            "num_hidden_layers": 5,
            "num_target_layers": 40,
            "target_layer_ids": [1, 10, 19, 28, 37],
            "block_size": 7,
            "mask_token_id": 151669,
        }
        parsed = parse_dflash_draft_config(draft_hf_config=deep_config)
        self.assertEqual(parsed.resolve_external_block_size(), 8)
        self.assertEqual(parsed.resolve_draft_input_size(8), 7)
        self.assertEqual(parsed.resolve_draft_input_size(4), 3)
        self.assertEqual(parsed.mask_token_id, 151669)

        nested_null = dict(deep_config)
        nested_null["dflash_config"] = {"mask_token_id": None}
        self.assertEqual(
            parse_dflash_draft_config(
                draft_hf_config=nested_null
            ).mask_token_id,
            151669,
        )

        native_config = dict(deep_config)
        native_config["architectures"] = ["DFlashDraftModel"]
        parsed = parse_dflash_draft_config(draft_hf_config=native_config)
        self.assertEqual(parsed.resolve_external_block_size(), 7)
        self.assertEqual(parsed.resolve_draft_input_size(7), 7)

    def test_rope_v4_and_v5_resolution(self):
        v4 = SimpleNamespace(rope_theta=123456, rope_scaling={"type": "linear"})
        v5 = SimpleNamespace(
            rope_theta=10000,
            rope_scaling=None,
            rope_parameters={"rope_theta": 1_000_000, "rope_type": "default"},
        )
        self.assertEqual(get_rope_config(v4), (123456, {"type": "linear"}))
        self.assertEqual(get_rope_config(v5), (1_000_000, v5.rope_parameters))

    def test_raw_model_forward_preserves_every_position(self):
        model = DFlashDraftModel.__new__(DFlashDraftModel)
        nn.Module.__init__(model)
        model.layers = nn.ModuleList()
        model.norm = nn.Identity()
        hidden = torch.randn(9, 13)
        output = DFlashDraftModel.forward(
            model,
            input_ids=torch.zeros(9, dtype=torch.long),
            positions=torch.arange(9),
            forward_batch=None,
            input_embeds=hidden,
        )
        self.assertEqual(tuple(output.hidden_states.shape), (9, 13))
        torch.testing.assert_close(output.hidden_states, hidden)


class TestEagle3Config(unittest.TestCase):
    def test_legacy_three_layer_and_deepspec_five_layer_width(self):
        legacy = {"eagle_config": {"use_aux_hidden_state": True}}
        legacy_without_aux = {"eagle_config": {"use_aux_hidden_state": False}}
        deep = {"target_layer_ids": [1, 10, 19, 28, 37]}
        nested = {"text_config": {"target_layer_ids": [2, 7, 12, 17]}}
        self.assertIsNone(resolve_eagle3_aux_layer_ids(legacy))
        self.assertEqual(
            resolve_eagle3_aux_hidden_size(legacy, target_hidden_size=5120), 15360
        )
        self.assertEqual(resolve_eagle3_aux_layer_ids(deep), [1, 10, 19, 28, 37])
        self.assertEqual(
            resolve_eagle3_aux_hidden_size(deep, target_hidden_size=5120), 25600
        )
        self.assertEqual(
            resolve_eagle3_aux_hidden_size(
                legacy_without_aux, target_hidden_size=5120
            ),
            5120,
        )
        self.assertEqual(resolve_eagle3_aux_layer_ids(nested), [2, 7, 12, 17])
        self.assertEqual(
            resolve_eagle3_aux_hidden_size(nested, target_hidden_size=5120),
            20480,
        )

    def test_registry_resolves_original_architectures(self):
        expected = {
            "DFlashDraftModel": DFlashDraftModel,
            "Qwen3DSparkModel": Qwen3DSparkModel,
            "LlamaForCausalLMEagle3": LlamaForCausalLMEagle3,
            "Qwen3Eagle3Model": Qwen3Eagle3Model,
        }
        for architecture, expected_cls in expected.items():
            model_cls, resolved_arch = ModelRegistry.resolve_model_cls([architecture])
            self.assertIs(model_cls, expected_cls)
            self.assertEqual(resolved_arch, architecture)


class TestOriginalDeepSpecCheckpoints(unittest.TestCase):
    dflash_paths = [
        WORKSPACE / "models/dflash_qwen3_4b_block7",
        WORKSPACE / "models/dflash_qwen3_8b_block7",
        WORKSPACE / "models/dflash_qwen3_14b_block7",
    ]
    eagle_paths = [
        WORKSPACE / "models/eagle3_qwen3_4b_ttt7",
        WORKSPACE / "models/eagle3_qwen3_8b_ttt7",
        WORKSPACE / "models/eagle3_qwen3_14b_ttt7",
    ]

    @classmethod
    def setUpClass(cls):
        required = cls.dflash_paths + cls.eagle_paths
        if not all((path / "config.json").is_file() for path in required):
            raise unittest.SkipTest(
                "DeepSpec compatibility checkpoints are not available"
            )

    def test_all_original_configs(self):
        for path in self.dflash_paths:
            config = AutoConfig.from_pretrained(path, local_files_only=True)
            parsed = parse_dflash_draft_config(draft_hf_config=config)
            self.assertEqual(config.architectures, ["Qwen3DSparkModel"])
            self.assertEqual(parsed.resolve_external_block_size(), 8)
            self.assertEqual(parsed.resolve_draft_input_size(8), 7)
            self.assertEqual(parsed.mask_token_id, 151669)
            rope_theta, _ = get_rope_config(config)
            self.assertEqual(rope_theta, 1_000_000)
            self.assertEqual(config.rope_parameters["rope_theta"], 1_000_000)

        expected_hidden_widths = (2560, 4096, 5120)
        expected_aux_widths = (12800, 20480, 25600)
        for path, expected_hidden_width, expected_aux_width in zip(
            self.eagle_paths, expected_hidden_widths, expected_aux_widths
        ):
            config = ModelConfig(
                model_path=str(path), trust_remote_code=False, is_draft_model=True
            )
            self.assertEqual(config.hf_config.architectures, ["Qwen3Eagle3Model"])
            self.assertEqual(
                config.hf_config.target_layer_ids,
                [
                    [1, 9, 17, 25, 33],
                    [1, 9, 17, 25, 33],
                    [1, 10, 19, 28, 37],
                ][self.eagle_paths.index(path)],
            )
            self.assertEqual(config.spec_hidden_size, expected_hidden_width)
            self.assertEqual(config.eagle_aux_hidden_size, expected_aux_width)

    def test_runtime_default_override_and_aux_widths(self):
        dflash_args = ServerArgs(
            model_path=str(WORKSPACE / "models/Qwen3-4B"),
            speculative_algorithm="DFLASH",
            speculative_draft_model_path=str(self.dflash_paths[0]),
            disable_cuda_graph=True,
            device="cuda",
        )
        self.assertEqual(dflash_args.speculative_num_draft_tokens, 8)
        explicit_args = ServerArgs(
            model_path=str(WORKSPACE / "models/Qwen3-4B"),
            speculative_algorithm="DFLASH",
            speculative_draft_model_path=str(self.dflash_paths[0]),
            speculative_dflash_block_size=4,
            disable_cuda_graph=True,
            device="cuda",
        )
        self.assertEqual(explicit_args.speculative_num_draft_tokens, 4)
        with self.assertRaisesRegex(ValueError, "B >= 2"):
            ServerArgs(
                model_path=str(WORKSPACE / "models/Qwen3-4B"),
                speculative_algorithm="DFLASH",
                speculative_draft_model_path=str(self.dflash_paths[0]),
                speculative_dflash_block_size=1,
                disable_cuda_graph=True,
                device="cuda",
            )

        eagle_args = ServerArgs(
            model_path=str(WORKSPACE / "models/Qwen3-4B"),
            speculative_algorithm="EAGLE3",
            speculative_draft_model_path=str(self.eagle_paths[0]),
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            disable_overlap_schedule=True,
            disable_cuda_graph=True,
            device="cuda",
        )
        target_config = ModelConfig.from_server_args(eagle_args)
        draft_config = ModelConfig.from_server_args(
            eagle_args,
            model_path=str(self.eagle_paths[0]),
            is_draft_model=True,
        )
        self.assertEqual(target_config.spec_hidden_size, 12800)
        self.assertEqual(target_config.eagle_aux_hidden_size, 12800)
        self.assertEqual(draft_config.spec_hidden_size, 2560)
        self.assertEqual(draft_config.eagle_aux_hidden_size, 12800)

    def test_checkpoint_weight_coverage_and_key_shapes(self):
        dflash_path = self.dflash_paths[-1] / "model.safetensors"
        with safe_open(dflash_path, framework="pt", device="cpu") as checkpoint:
            names = set(checkpoint.keys())
            expected = expected_dflash_core_weight_names(5)
            self.assertEqual(len(names), 60)
            self.assertEqual(
                names - {"embed_tokens.weight", "lm_head.weight"}, expected
            )
            self.assertEqual(
                tuple(checkpoint.get_slice("fc.weight").get_shape()), (5120, 25600)
            )
            self.assertEqual(
                tuple(
                    checkpoint.get_slice("layers.0.self_attn.q_proj.weight").get_shape()
                ),
                (5120, 5120),
            )

        eagle_path = self.eagle_paths[-1] / "model.safetensors"
        with safe_open(eagle_path, framework="pt", device="cpu") as checkpoint:
            names = set(checkpoint.keys())
            self.assertEqual(names, expected_qwen3_eagle3_weight_names(1))
            self.assertEqual(
                tuple(checkpoint.get_slice("fc.weight").get_shape()), (5120, 25600)
            )
            for projection, rows in (
                ("q_proj", 5120),
                ("k_proj", 1024),
                ("v_proj", 1024),
            ):
                self.assertEqual(
                    tuple(
                        checkpoint.get_slice(
                            f"layers.0.self_attn.{projection}.weight"
                        ).get_shape()
                    ),
                    (rows, 10240),
                )
            self.assertEqual(
                tuple(
                    checkpoint.get_slice("layers.0.self_attn.q_norm.weight").get_shape()
                ),
                (128,),
            )
            self.assertEqual(
                tuple(
                    checkpoint.get_slice("layers.0.self_attn.k_norm.weight").get_shape()
                ),
                (128,),
            )


if __name__ == "__main__":
    unittest.main()
