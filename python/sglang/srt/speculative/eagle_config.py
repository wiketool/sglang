"""Lightweight EAGLE3 checkpoint configuration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def resolve_eagle3_aux_layer_ids(draft_hf_config: Any) -> Optional[list[int]]:
    """Resolve capture ids with legacy ``eagle_config`` taking precedence."""
    eagle_config = _config_get(draft_hf_config, "eagle_config", None)
    layer_ids = None
    if eagle_config is not None:
        layer_ids = _config_get(eagle_config, "eagle_aux_hidden_state_layer_ids", None)
    if layer_ids is None:
        layer_ids = _config_get(draft_hf_config, "target_layer_ids", None)
    if layer_ids is None:
        text_config = _config_get(draft_hf_config, "text_config", None)
        if text_config is not None:
            layer_ids = _config_get(text_config, "target_layer_ids", None)
    if layer_ids is None:
        return None
    resolved = [int(x) for x in layer_ids]
    if not resolved:
        raise ValueError("EAGLE3 target layer capture ids must be non-empty.")
    return resolved


def resolve_eagle3_use_aux_hidden_state(draft_hf_config: Any) -> bool:
    eagle_config = _config_get(draft_hf_config, "eagle_config", None)
    if eagle_config is None:
        return True
    return bool(_config_get(eagle_config, "use_aux_hidden_state", True))


def resolve_eagle3_aux_hidden_size(
    draft_hf_config: Any,
    *,
    target_hidden_size: int,
    legacy_default_num_layers: int = 3,
) -> int:
    if not resolve_eagle3_use_aux_hidden_state(draft_hf_config):
        return int(target_hidden_size)
    layer_ids = resolve_eagle3_aux_layer_ids(draft_hf_config)
    num_layers = (
        len(layer_ids) if layer_ids is not None else int(legacy_default_num_layers)
    )
    if num_layers <= 0:
        raise ValueError(
            f"EAGLE3 auxiliary hidden layer count must be positive, got {num_layers}."
        )
    return int(target_hidden_size) * num_layers
