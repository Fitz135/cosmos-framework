# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Lightweight helpers shared by model conversion commands."""

import json
from pathlib import Path
from typing import Any


def _is_native_edge_snapshot(hf_path: Path) -> bool:
    """Return whether a local snapshot carries native Cosmos3-Edge metadata."""
    config_path = hf_path / "config.json"
    if config_path.is_file():
        try:
            if json.loads(config_path.read_text(encoding="utf-8")).get("model_type") == "cosmos3_edge":
                return True
        except (OSError, ValueError):
            pass
    processor_config_path = hf_path / "processor_config.json"
    if processor_config_path.is_file():
        try:
            processor_config = json.loads(processor_config_path.read_text(encoding="utf-8"))
            if processor_config.get("processor_class") == "Cosmos3EdgeProcessor":
                return True
        except (OSError, ValueError):
            pass
    return False


def redirect_edge_processor_to_local(model_dict: dict[str, Any], hf_path: Path) -> bool:
    """Point an Edge lazy processor node at files bundled in ``hf_path``.

    A local ``--checkpoint-path`` may be paired with a registered config whose
    processor still names ``nvidia/Cosmos3-Edge``. Rewriting only native Edge
    snapshots keeps HF-to-DCP conversion offline without changing other model
    families or custom tokenizer nodes.
    """
    if not _is_native_edge_snapshot(hf_path):
        return False
    tokenizer = model_dict.get("config", {}).get("vlm_config", {}).get("tokenizer")
    if tokenizer is None:
        return False
    target = str(tokenizer.get("_target_", "")).rsplit(".", 1)[-1]
    if target != "build_processor_lazy" or not (tokenizer.get("repository") or tokenizer.get("tokenizer_type")):
        return False
    tokenizer.pop("repository", None)
    tokenizer.pop("revision", None)
    tokenizer.pop("subdir", None)
    tokenizer["tokenizer_type"] = str(hf_path)
    return True


def redirect_video_vae_to_local(model_dict: dict[str, Any], vae_path: Path) -> bool:
    """Point the model's video tokenizer at an already-downloaded VAE file."""
    tokenizer = model_dict.get("config", {}).get("tokenizer")
    if tokenizer is None or "vae_path" not in tokenizer:
        return False
    tokenizer["vae_path"] = str(vae_path)
    tokenizer["bucket_name"] = ""
    tokenizer["object_store_credential_path_pretrained"] = ""
    return True
