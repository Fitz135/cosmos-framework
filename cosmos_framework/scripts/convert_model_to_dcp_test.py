# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from copy import deepcopy
from pathlib import Path

from cosmos_framework.scripts.model_conversion_utils import (
    redirect_edge_processor_to_local,
    redirect_video_vae_to_local,
)


def _model_dict() -> dict:
    return {
        "config": {
            "vlm_config": {
                "tokenizer": {
                    "_target_": "cosmos_framework.data.generator.processors.build_processor_lazy",
                    "repository": "nvidia/Cosmos3-Edge",
                    "revision": "main",
                    "subdir": "",
                }
            }
        }
    }


def test_redirect_edge_processor_to_local(tmp_path: Path):
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"processor_class": "Cosmos3EdgeProcessor"}),
        encoding="utf-8",
    )
    model_dict = _model_dict()

    assert redirect_edge_processor_to_local(model_dict, tmp_path)
    assert model_dict["config"]["vlm_config"]["tokenizer"] == {
        "_target_": "cosmos_framework.data.generator.processors.build_processor_lazy",
        "tokenizer_type": str(tmp_path),
    }


def test_redirect_edge_processor_requires_compatible_snapshot(tmp_path: Path):
    model_dict = _model_dict()
    before = deepcopy(model_dict)

    assert not redirect_edge_processor_to_local(model_dict, tmp_path)
    assert model_dict == before


def test_redirect_edge_processor_leaves_unknown_node_untouched(tmp_path: Path):
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"processor_class": "Cosmos3EdgeProcessor"}),
        encoding="utf-8",
    )
    model_dict = _model_dict()
    model_dict["config"]["vlm_config"]["tokenizer"]["_target_"] = "acme.build_processor"
    before = deepcopy(model_dict)

    assert not redirect_edge_processor_to_local(model_dict, tmp_path)
    assert model_dict == before


def test_redirect_video_vae_to_local(tmp_path: Path):
    vae_path = tmp_path / "Wan2.2_VAE.pth"
    model_dict = {
        "config": {
            "tokenizer": {
                "vae_path": "registered/vae.pth",
                "bucket_name": "bucket",
                "object_store_credential_path_pretrained": "credentials/gcp_training.secret",
            }
        }
    }

    assert redirect_video_vae_to_local(model_dict, vae_path)
    assert model_dict["config"]["tokenizer"] == {
        "vae_path": str(vae_path),
        "bucket_name": "",
        "object_store_credential_path_pretrained": "",
    }


def test_redirect_video_vae_requires_tokenizer_node(tmp_path: Path):
    model_dict = {"config": {}}

    assert not redirect_video_vae_to_local(model_dict, tmp_path / "vae.pth")
