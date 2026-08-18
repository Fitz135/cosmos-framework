# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cosmos_framework.evaluation.libero import CheckpointProfileError, resolve_checkpoint_profile


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def _stats(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "stats.json",
        {"global_raw": {"q01": [-1.0] * 10, "q99": [1.0] * 10}},
    )


def _complete_profile(stats_path: Path) -> dict:
    return {
        "schema_version": 1,
        "domain_name": "libero",
        "action_chunk_size": 8,
        "conditioning_fps": 10,
        "effective_action_dim": 10,
        "action_space": "frame_wise_relative",
        "rotation_space": "6d",
        "pose_coordinate_frame": "native",
        "action_normalization": "quantile_rot",
        "action_stats_path": str(stats_path),
        "format_prompt_as_json": True,
        "cameras": ["agentview", "wrist"],
        "image_size": 256,
        "rotate_images": False,
        "control_mode": "OSC_POSE",
        "control_frequency": 10,
        "gripper_mode": "passthrough",
        "weights_variant": "ema",
    }


def _hf_checkpoint(tmp_path: Path, *, policy: dict | None) -> Path:
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "config.json").write_text("{}")
    metadata: dict = {"use_ema_weights": True}
    if policy is not None:
        metadata["policy"] = policy
    _write_json(checkpoint / "checkpoint.json", metadata)
    return checkpoint


def test_hf_policy_and_explicit_profile_resolve_stable_hash(tmp_path: Path) -> None:
    stats_path = _stats(tmp_path)
    checkpoint = _hf_checkpoint(
        tmp_path,
        policy={"action_chunk_size": 8, "conditioning_fps": 10.0, "domain_name": "libero"},
    )
    profile_path = _write_json(tmp_path / "profile.json", _complete_profile(stats_path))

    first = resolve_checkpoint_profile(checkpoint, profile_path=profile_path)
    second = resolve_checkpoint_profile(checkpoint, profile_path=profile_path)

    assert first.profile_hash == second.profile_hash
    assert first.checkpoint_fingerprint == second.checkpoint_fingerprint
    assert len(first.checkpoint_fingerprint) == 64
    assert first.action_stats_sha256 == hashlib.sha256(stats_path.read_bytes()).hexdigest()
    assert first.checkpoint_role.value == "finetuned"
    assert first.weights_variant.value == "ema"
    assert not first.zero_shot


def test_weight_content_changes_checkpoint_fingerprint_and_profile_hash(tmp_path: Path) -> None:
    stats_path = _stats(tmp_path)
    checkpoint = _hf_checkpoint(
        tmp_path,
        policy={"action_chunk_size": 8, "conditioning_fps": 10.0, "domain_name": "libero"},
    )
    profile_path = _write_json(tmp_path / "profile.json", _complete_profile(stats_path))

    first = resolve_checkpoint_profile(checkpoint, profile_path=profile_path)
    (checkpoint / "model.safetensors").write_bytes(b"different-weights")
    second = resolve_checkpoint_profile(checkpoint, profile_path=profile_path)

    assert first.checkpoint_fingerprint != second.checkpoint_fingerprint
    assert first.profile_hash != second.profile_hash


def test_internal_hf_config_is_not_double_counted_in_checkpoint_fingerprint(tmp_path: Path) -> None:
    stats_path = _stats(tmp_path)
    checkpoint = _hf_checkpoint(
        tmp_path,
        policy={"action_chunk_size": 8, "conditioning_fps": 10.0, "domain_name": "libero"},
    )
    profile_path = _write_json(tmp_path / "profile.json", _complete_profile(stats_path))

    without_resolved_config = resolve_checkpoint_profile(checkpoint, profile_path=profile_path)
    with_resolved_config = resolve_checkpoint_profile(
        checkpoint,
        profile_path=profile_path,
        resolved_config_path=checkpoint / "config.json",
    )

    assert with_resolved_config.checkpoint_fingerprint == without_resolved_config.checkpoint_fingerprint
    assert with_resolved_config.profile_hash == without_resolved_config.profile_hash


def test_external_hf_config_changes_checkpoint_fingerprint_and_profile_hash(tmp_path: Path) -> None:
    stats_path = _stats(tmp_path)
    checkpoint = _hf_checkpoint(
        tmp_path,
        policy={"action_chunk_size": 8, "conditioning_fps": 10.0, "domain_name": "libero"},
    )
    profile_path = _write_json(tmp_path / "profile.json", _complete_profile(stats_path))
    config_path = tmp_path / "inference_config.py"
    config_path.write_text("MODEL_VARIANT = 'edge'\n")

    first = resolve_checkpoint_profile(
        checkpoint,
        profile_path=profile_path,
        resolved_config_path=config_path,
    )
    config_path.write_text("MODEL_VARIANT = 'edge-v2'\n")
    second = resolve_checkpoint_profile(
        checkpoint,
        profile_path=profile_path,
        resolved_config_path=config_path,
    )

    assert first.checkpoint_fingerprint != second.checkpoint_fingerprint
    assert first.profile_hash != second.profile_hash


def test_tokenizer_and_processor_content_change_checkpoint_fingerprint(tmp_path: Path) -> None:
    stats_path = _stats(tmp_path)
    checkpoint = _hf_checkpoint(
        tmp_path,
        policy={"action_chunk_size": 8, "conditioning_fps": 10.0, "domain_name": "libero"},
    )
    profile_path = _write_json(tmp_path / "profile.json", _complete_profile(stats_path))
    tokenizer_path = _write_json(checkpoint / "tokenizer_config.json", {"model_max_length": 128})
    processor_path = _write_json(checkpoint / "preprocessor_config.json", {"size": 256})

    first = resolve_checkpoint_profile(checkpoint, profile_path=profile_path)
    _write_json(tokenizer_path, {"model_max_length": 256})
    changed_tokenizer = resolve_checkpoint_profile(checkpoint, profile_path=profile_path)
    _write_json(processor_path, {"size": 512})
    changed_processor = resolve_checkpoint_profile(checkpoint, profile_path=profile_path)

    assert first.checkpoint_fingerprint != changed_tokenizer.checkpoint_fingerprint
    assert changed_tokenizer.checkpoint_fingerprint != changed_processor.checkpoint_fingerprint
    assert first.profile_hash != changed_tokenizer.profile_hash
    assert changed_tokenizer.profile_hash != changed_processor.profile_hash


def test_hf_checkpoint_without_weight_files_fails_explicitly(tmp_path: Path) -> None:
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    _write_json(checkpoint / "checkpoint.json", {"use_ema_weights": True})

    with pytest.raises(CheckpointProfileError, match=r"no \*\.safetensors weight files"):
        resolve_checkpoint_profile(checkpoint)


def test_conflicting_hf_policy_and_profile_fail_fast(tmp_path: Path) -> None:
    stats_path = _stats(tmp_path)
    checkpoint = _hf_checkpoint(
        tmp_path,
        policy={"action_chunk_size": 8, "conditioning_fps": 10.0, "domain_name": "libero"},
    )
    profile = _complete_profile(stats_path)
    profile["action_chunk_size"] = 16
    profile_path = _write_json(tmp_path / "profile.json", profile)

    with pytest.raises(CheckpointProfileError, match="Conflicting value for 'action_chunk_size'"):
        resolve_checkpoint_profile(checkpoint, profile_path=profile_path)


def test_missing_hf_policy_fields_fail_fast(tmp_path: Path) -> None:
    checkpoint = _hf_checkpoint(
        tmp_path,
        policy={"action_chunk_size": 8, "conditioning_fps": 10.0, "domain_name": "libero"},
    )

    with pytest.raises(CheckpointProfileError, match="missing required fields"):
        resolve_checkpoint_profile(checkpoint)


def test_dcp_resolved_config_merges_target_adapter(tmp_path: Path) -> None:
    stats_path = _stats(tmp_path)
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "iter_000010000"
    model_dir = checkpoint / "model"
    model_dir.mkdir(parents=True)
    (model_dir / ".metadata").write_bytes(b"metadata")
    (model_dir / "__0_0.distcp").write_bytes(b"weights")
    config_path = run_dir / "config.yaml"
    config_path.write_text(
        f"""
dataloader_train:
  dataloader:
    datasets:
      libero:
        dataset:
          embodiment_type: libero
          chunk_length: 8
          fps: 10
          action_space: frame_wise_relative
          rotation_space: 6d
          pose_coordinate_frame: native
          action_normalization: quantile_rot
          action_stats_path: {stats_path}
          format_prompt_as_json: true
          camera_mode: concat_view
          wrist_camera_key: observation.images.image2
          image_size: 256
"""
    )
    adapter_path = _write_json(
        tmp_path / "adapter.json",
        {
            "schema_version": 1,
            "adapter_id": "libero-native-v1",
            "rotate_images": False,
            "control_mode": "OSC_POSE",
            "control_frequency": 10,
            "gripper_mode": "passthrough",
            "weights_variant": "ema",
        },
    )

    profile = resolve_checkpoint_profile(checkpoint, target_adapter_path=adapter_path)

    assert profile.checkpoint_format.value == "dcp"
    assert profile.checkpoint_role.value == "finetuned"
    assert profile.target_adapter_id == "libero-native-v1"
    assert profile.cameras == ("agentview", "wrist")
    assert profile.control_frequency == 10
    assert profile.effective_action_dim == 10
    first_fingerprint = profile.checkpoint_fingerprint
    first_profile_hash = profile.profile_hash

    (model_dir / "__0_0.distcp").write_bytes(b"different-weights")
    changed_weights = resolve_checkpoint_profile(checkpoint, target_adapter_path=adapter_path)
    assert changed_weights.checkpoint_fingerprint != first_fingerprint
    assert changed_weights.profile_hash != first_profile_hash

    config_path.write_text(config_path.read_text() + "\n# export metadata changed\n")
    changed_config = resolve_checkpoint_profile(checkpoint, target_adapter_path=adapter_path)
    assert changed_config.checkpoint_fingerprint != changed_weights.checkpoint_fingerprint
    assert changed_config.profile_hash != changed_weights.profile_hash


def test_base_checkpoint_requires_explicit_zero_shot_target_adapter(tmp_path: Path) -> None:
    stats_path = _stats(tmp_path)
    checkpoint = _hf_checkpoint(tmp_path, policy=None)

    with pytest.raises(CheckpointProfileError, match="explicit target_adapter_path"):
        resolve_checkpoint_profile(checkpoint)

    adapter = _complete_profile(stats_path)
    adapter.update({"adapter_id": "libero-zero-shot-v1", "weights_variant": "regular"})
    adapter_path = _write_json(tmp_path / "adapter.json", adapter)
    profile = resolve_checkpoint_profile(checkpoint, target_adapter_path=adapter_path)

    assert profile.checkpoint_role.value == "base"
    assert profile.zero_shot
    assert profile.target_adapter_id == "libero-zero-shot-v1"


def test_control_frequency_is_required_and_changes_profile_hash(tmp_path: Path) -> None:
    stats_path = _stats(tmp_path)
    checkpoint = _hf_checkpoint(
        tmp_path,
        policy={"action_chunk_size": 8, "conditioning_fps": 10.0, "domain_name": "libero"},
    )
    missing_frequency = _complete_profile(stats_path)
    missing_frequency.pop("control_frequency")
    missing_path = _write_json(tmp_path / "missing-frequency.json", missing_frequency)

    with pytest.raises(CheckpointProfileError, match="control_frequency"):
        resolve_checkpoint_profile(checkpoint, profile_path=missing_path)

    ten_hz_path = _write_json(tmp_path / "ten-hz.json", _complete_profile(stats_path))
    twenty_hz = _complete_profile(stats_path)
    twenty_hz["control_frequency"] = 20
    twenty_hz_path = _write_json(tmp_path / "twenty-hz.json", twenty_hz)

    ten_hz_profile = resolve_checkpoint_profile(checkpoint, profile_path=ten_hz_path)
    twenty_hz_profile = resolve_checkpoint_profile(checkpoint, profile_path=twenty_hz_path)

    assert ten_hz_profile.control_frequency == 10
    assert twenty_hz_profile.control_frequency == 20
    assert ten_hz_profile.profile_hash != twenty_hz_profile.profile_hash


def test_stats_sha_mismatch_fails_fast(tmp_path: Path) -> None:
    stats_path = _stats(tmp_path)
    checkpoint = _hf_checkpoint(
        tmp_path,
        policy={"action_chunk_size": 8, "conditioning_fps": 10.0, "domain_name": "libero"},
    )
    profile = _complete_profile(stats_path)
    profile["action_stats_sha256"] = "0" * 64
    profile_path = _write_json(tmp_path / "profile.json", profile)

    with pytest.raises(CheckpointProfileError, match="SHA-256 mismatch"):
        resolve_checkpoint_profile(checkpoint, profile_path=profile_path)
