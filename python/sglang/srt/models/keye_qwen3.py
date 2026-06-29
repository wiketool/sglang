import os
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.keye_vit import KeyeSiglipVisionModel, Projector
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.utils import add_prefix, print_info_once
from sglang.srt.utils.hf_transformers_utils import get_processor

cached_get_processor = lru_cache(get_processor)


class KeyeVL1_5ForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3Model,
    ):
        super().__init__()

        self.config = config
        self.visual = KeyeSiglipVisionModel(
            config.vision_config,
            quant_config=None,
            prefix=add_prefix("visual", prefix),
        )

        self.mlp_AR = Projector(
            config,
            config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("mlp_AR", prefix),
        )

        self.model = language_model_cls(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.capture_aux_hidden_states = False

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        device = pixel_values.device
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        # assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        image_grid_thw = image_grid_thw.to(device)
        assert torch.all(image_grid_thw[:, 0] == 1)

        total_patches = image_grid_thw.prod(dim=1)
        width = torch.repeat_interleave(image_grid_thw[:, 2], total_patches)

        cu_seqlens = total_patches.cumsum(0)

        arange = torch.arange(cu_seqlens[-1], dtype=torch.long, device=device)
        image_position_ids = arange - torch.repeat_interleave(
            cu_seqlens.to(device) - total_patches, total_patches
        )

        width_position_ids = torch.remainder(image_position_ids, width)
        height_position_ids = torch.div(
            image_position_ids, width, rounding_mode="floor"
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(
            dtype=torch.int32, device=device
        )
        width_position_ids = width_position_ids.to(device)
        height_position_ids = height_position_ids.to(device)

        image_embeds = self.visual(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=image_position_ids,
            vision_return_embed_list=False,
            interpolate_pos_encoding=True,
            sample_indices=None,
            height_position_ids=height_position_ids,
            width_position_ids=width_position_ids,
            cu_seqlens=cu_seqlens,
            return_pooler_output=False,
            use_rope=True,
            window_size=-1,
        )

        image_embeds = torch.cat(self.mlp_AR(image_embeds, image_grid_thw), dim=0)

        return image_embeds

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:

        def split_thw(grid_thw: torch.Tensor):
            if grid_thw.dim() == 1:
                grid_thw = grid_thw.unsqueeze(0)

            clone = grid_thw.clone()
            clone[:, 0] = 1
            return torch.repeat_interleave(clone, grid_thw[:, 0], dim=0)

        video_pixel_features = [getattr(item, "feature") for item in items]
        device = video_pixel_features[0].device if len(video_pixel_features) > 0 else self.visual.device

        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        video_grid_thw = split_thw(video_grid_thw)
        video_grid_thw = video_grid_thw.to(device)
        assert torch.all(video_grid_thw[:, 0] == 1)

        # Support chunked processing to avoid OOM with long videos
        patch_chunk_size = int(os.environ.get("KEYE_VIDEO_PATCH_CHUNK_SIZE", "8192"))
        num_frames = video_grid_thw.shape[0]
        total_patches = video_grid_thw.prod(dim=1).sum().item()
        if patch_chunk_size > 0:
            chunk_size = int(max(patch_chunk_size / (total_patches / num_frames), 1))
        else:
            chunk_size = patch_chunk_size

        # If chunk_size is disabled or total frames <= chunk_size, process all at once
        if chunk_size <= 0 or num_frames <= chunk_size:
            pixel_values_videos = torch.cat(video_pixel_features, dim=0).type(self.visual.dtype)
            total_patches = video_grid_thw.prod(dim=1)
            width = torch.repeat_interleave(video_grid_thw[:, 2], total_patches)
            cu_seqlens = total_patches.cumsum(0)
            arange = torch.arange(cu_seqlens[-1], dtype=torch.long, device=device)
            video_position_ids = arange - torch.repeat_interleave(
                cu_seqlens.to(device) - total_patches, total_patches
            )

            width_position_ids = torch.remainder(video_position_ids, width)
            height_position_ids = torch.div(
                video_position_ids, width, rounding_mode="floor"
            )
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(
                dtype=torch.int32, device=device
            )
            width_position_ids = width_position_ids.to(device)
            height_position_ids = height_position_ids.to(device)

            video_embeds = self.visual(
                pixel_values=pixel_values_videos,
                image_grid_thw=video_grid_thw,
                position_ids=video_position_ids,
                height_position_ids=height_position_ids,
                width_position_ids=width_position_ids,
                vision_return_embed_list=False,
                interpolate_pos_encoding=True,
                sample_indices=None,
                cu_seqlens=cu_seqlens,
                return_pooler_output=False,
                use_rope=True,
                window_size=-1,
            )
            video_embeds = torch.cat(self.mlp_AR(video_embeds, video_grid_thw), dim=0)
        else:
            # Process in chunks
            video_embeds_list = []
            pixel_start = 0
            video_idx = 0

            for chunk_start in range(0, num_frames, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_frames)

                # Slice grid information for this chunk
                video_grid_thw_chunk = video_grid_thw[chunk_start:chunk_end]

                # Calculate total number of patches in this chunk
                # Each row in video_grid_thw_chunk is [t, h, w], total patches = sum(t*h*w)
                num_patches_in_chunk = video_grid_thw_chunk.prod(dim=1).sum().item()

                # Slice pixel values based on actual patch count
                # Build a pixel slice that may span multiple video items.
                # When a chunk's patches cross a video boundary, we
                # concatenate slices from consecutive items.
                remaining_patches = num_patches_in_chunk
                pixel_slices = []
                while remaining_patches > 0:
                    current_feature = video_pixel_features[video_idx]
                    current_len = current_feature.shape[0]
                    available = current_len - pixel_start
                    take = min(remaining_patches, available)
                    if take > 0:
                        pixel_slices.append(
                            current_feature[pixel_start : pixel_start + take].type(
                                self.visual.dtype
                            )
                        )
                    remaining_patches -= take
                    pixel_start += take
                    # Move to next video when current one is exhausted
                    if pixel_start >= current_len and remaining_patches > 0:
                        video_idx += 1
                        pixel_start = 0

                pixel_values_chunk = torch.cat(pixel_slices, dim=0)

                # Calculate position IDs for this chunk
                total_patches_chunk = video_grid_thw_chunk.prod(dim=1)
                width_chunk = torch.repeat_interleave(
                    video_grid_thw_chunk[:, 2], total_patches_chunk
                )
                cu_seqlens_chunk = total_patches_chunk.cumsum(0)
                arange_chunk = torch.arange(
                    cu_seqlens_chunk[-1], dtype=torch.long, device=device
                )
                video_position_ids_chunk = arange_chunk - torch.repeat_interleave(
                    cu_seqlens_chunk.to(device) - total_patches_chunk, total_patches_chunk
                )

                width_position_ids_chunk = torch.remainder(
                    video_position_ids_chunk, width_chunk
                )
                height_position_ids_chunk = torch.div(
                    video_position_ids_chunk, width_chunk, rounding_mode="floor"
                )
                cu_seqlens_chunk = F.pad(cu_seqlens_chunk, (1, 0), value=0).to(
                    dtype=torch.int32, device=device
                )
                width_position_ids_chunk = width_position_ids_chunk.to(device)
                height_position_ids_chunk = height_position_ids_chunk.to(device)

                video_embeds_chunk = self.visual(
                    pixel_values=pixel_values_chunk,
                    image_grid_thw=video_grid_thw_chunk,
                    position_ids=video_position_ids_chunk,
                    height_position_ids=height_position_ids_chunk,
                    width_position_ids=width_position_ids_chunk,
                    vision_return_embed_list=False,
                    interpolate_pos_encoding=True,
                    sample_indices=None,
                    cu_seqlens=cu_seqlens_chunk,
                    return_pooler_output=False,
                    use_rope=True,
                    window_size=-1,
                )

                # Apply mlp_AR for this chunk
                video_embeds_chunk = torch.cat(
                    self.mlp_AR(video_embeds_chunk, video_grid_thw_chunk), dim=0
                )
                video_embeds_list.append(video_embeds_chunk)

            # Concatenate all chunks
            video_embeds = torch.cat(video_embeds_list, dim=0)

        return video_embeds

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        """Run forward pass for Keye-8b-preview.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
        """
        if get_tensor_model_parallel_rank() == 0:
            print_info_once("Use Keye-Qwen3 model to forward!")

        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        if not (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_mm_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}"
                )

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return self.pooler(hidden_states, forward_batch)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        self.capture_aux_hidden_states = True
        if hasattr(self.model, "capture_aux_hidden_states"):
            self.model.capture_aux_hidden_states = True

        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            layers_to_capture = [val + 1 for val in layer_ids]

        if hasattr(self.model, "set_eagle3_layers_to_capture"):
            self.model.set_eagle3_layers_to_capture(layers_to_capture)
        else:
            self.model.layers_to_capture = layers_to_capture

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "visual" in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                vision_stacked_params_mapping = [
                    ("qkv_proj", "q_proj", "q"),
                    ("qkv_proj", "k_proj", "k"),
                    ("qkv_proj", "v_proj", "v"),
                ]
                ignore_names = [
                    "head.attention",
                    "head.mlp",
                    "head.layernorm",
                    "head.probe",
                ]
                for (
                    param_name,
                    weight_name,
                    shard_id,
                ) in vision_stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    try:
                        param = params_dict[name]
                    except KeyError:
                        print(params_dict.keys())
                        raise
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    if any(ignore_name in name for ignore_name in ignore_names):
                        continue
                    try:
                        param = params_dict[name]
                    except KeyError:
                        print(params_dict.keys())
                        raise
                    weight_loader = getattr(
                        param,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(param, loaded_weight)

EntryClass = [KeyeVL1_5ForConditionalGeneration]
