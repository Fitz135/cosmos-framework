# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Static configuration contract for the Edge 10 FPS LIBERO recipe.

The source is inspected without importing the training stack so this test also
runs on CPU-only development hosts that do not provide CUDA runtime libraries.
The rjob dry-run separately exercises Hydra composition and model imports.
"""

import ast
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[6]
RECIPE_PATH = Path(__file__).with_name("action_policy_libero_all_edge_10fps.py")
RECIPE_SOURCE = RECIPE_PATH.read_text()
RECIPE_TREE = ast.parse(RECIPE_SOURCE)
BASE_CONFIG_SOURCE = (REPO_ROOT / "cosmos_framework/configs/base/config.py").read_text()


def _named_call(name: str) -> ast.Call:
    for node in ast.walk(RECIPE_TREE):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Call):
            continue
        lazy = node.func
        if (
            isinstance(lazy.func, ast.Name)
            and lazy.func.id == "L"
            and len(lazy.args) == 1
            and isinstance(lazy.args[0], ast.Name)
            and lazy.args[0].id == name
        ):
            return node
    raise AssertionError(f"call to {name} not found")


def _dict_call_with_keyword(keyword: str) -> ast.Call:
    for node in ast.walk(RECIPE_TREE):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dict"
            and any(item.arg == keyword for item in node.keywords)
        ):
            return node
    raise AssertionError(f"dict containing {keyword} not found")


def _literal_keyword(call: ast.Call, name: str):
    value = next(item.value for item in call.keywords if item.arg == name)
    return ast.literal_eval(value)


def test_edge_recipe_preserves_pretrained_action_heads() -> None:
    checkpoint = _dict_call_with_keyword("keys_to_skip_loading")
    optimizer = _dict_call_with_keyword("keys_to_select")

    assert "from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG" in (
        RECIPE_SOURCE
    )
    assert "copy.deepcopy(EDGE_MODEL_CONFIG)" in RECIPE_SOURCE
    assert "COSMOS3_EDGE_PROCESSOR_PATH" in RECIPE_SOURCE
    assert 'tokenizer_cfg["repository"] = None' in RECIPE_SOURCE
    assert 'tokenizer_cfg["tokenizer_type"] = processor_path' in RECIPE_SOURCE
    assert 'cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False' in RECIPE_SOURCE
    assert _literal_keyword(checkpoint, "keys_to_skip_loading") == ["net_ema."]
    assert _literal_keyword(checkpoint, "strict_resume") is True
    assert {"k_norm_und_for_gen", "action2llm", "llm2action", "action_modality_embed"} <= set(
        _literal_keyword(optimizer, "keys_to_select")
    )
    assert (
        "import cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_all_edge_10fps"
    ) in BASE_CONFIG_SOURCE


def test_edge_recipe_uses_10fps_eight_action_full_dataset() -> None:
    dataset = _named_call("get_action_libero_sft_dataset")

    assert _literal_keyword(dataset, "root") == "${oc.env:LIBERO_ROOT}"
    assert _literal_keyword(dataset, "fps") == 10
    assert _literal_keyword(dataset, "chunk_length") == 8
    assert _literal_keyword(dataset, "camera_mode") == "concat_view"
    assert _literal_keyword(dataset, "wrist_camera_key") == "observation.images.image2"
    assert _literal_keyword(dataset, "split") == "full"
    assert _literal_keyword(dataset, "action_stats_path") == (
        "libero_10fps_pm_one_native_frame_wise_relative_rot6d.json"
    )


def test_edge_toml_defines_fsdp8_global_batch_2048() -> None:
    toml_path = REPO_ROOT / "examples/toml/sft_config/action_policy_libero_all_edge_10fps.toml"
    recipe = tomllib.loads(toml_path.read_text())

    assert recipe["job"]["wandb_mode"] == "offline"
    assert recipe["model"]["parallelism"] == {
        "data_parallel_shard_degree": 8,
        "data_parallel_replicate_degree": 1,
    }
    assert recipe["trainer"]["grad_accum_iter"] == 2
    assert recipe["trainer"]["max_iter"] == 5000
    assert recipe["checkpoint"]["save_iter"] == 500
    assert 128 * 8 * recipe["trainer"]["grad_accum_iter"] == 2048
