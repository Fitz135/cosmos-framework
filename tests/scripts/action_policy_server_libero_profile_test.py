# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from cosmos_framework.data.generator.action.action_processing import ActionProcessor
from cosmos_framework.evaluation.libero.prompt import build_libero_json_prompt
from cosmos_framework.evaluation.libero.schema import LiberoCheckpointProfile
from cosmos_framework.scripts import action_policy_server_libero as server


def _profile(*, role: str = "finetuned") -> LiberoCheckpointProfile:
    is_base = role == "base"
    return LiberoCheckpointProfile.model_validate(
        {
            "schema_version": 1,
            "checkpoint_path": "/checkpoints/edge",
            "checkpoint_format": "hf",
            "checkpoint_role": role,
            "zero_shot": is_base,
            "target_adapter_id": "libero-zero-shot-v1" if is_base else None,
            "domain_name": "libero",
            "action_chunk_size": 2,
            "conditioning_fps": 10,
            "effective_action_dim": 10,
            "action_space": "frame_wise_relative",
            "rotation_space": "6d",
            "pose_coordinate_frame": "native",
            "action_normalization": "quantile_rot",
            "action_stats_path": "/checkpoints/edge/action_stats.json",
            "action_stats_sha256": "b" * 64,
            "checkpoint_fingerprint": "c" * 64,
            "format_prompt_as_json": True,
            "cameras": ["agentview", "wrist"],
            "image_size": 256,
            "rotate_images": False,
            "control_mode": "OSC_POSE",
            "control_frequency": 10,
            "gripper_mode": "passthrough",
            "weights_variant": "ema",
            "profile_hash": "a" * 64,
            "profile_sources": ["unit-test"],
        }
    )


def _service(
    *,
    model: Any | None = None,
    normalizer: Any | None = None,
    profile: LiberoCheckpointProfile | None = None,
) -> server.ActionModelService:
    resolved_profile = profile or _profile()
    service = server.ActionModelService.__new__(server.ActionModelService)
    service.profile = resolved_profile
    service.cfg = server.ActionServerConfig(
        seed=7,
        guidance=1.0,
        num_steps=4,
        fps=10,
        action_chunk_size=2,
        max_action_dim=12,
        raw_action_dim=10,
        dump_dir=None,
        dump_every=0,
        http_400_on_error=True,
        action_stats_path=Path(resolved_profile.action_stats_path),
        action_normalization="quantile_rot",
        experiment_name="libero-unit-test",
        checkpoint_dir=resolved_profile.checkpoint_path,
        profile_hash=resolved_profile.profile_hash,
    )
    service.setup_args = SimpleNamespace(config_file=Path("/tmp/config.yaml"), config_file_type="yaml")
    service.raw_action_dim = 10
    service.action_normalization = "quantile_rot"
    service.action_normalizer = normalizer
    service.format_prompt_as_json = False
    service.append_duration_fps = False
    service.append_resolution_info = False
    service.model = model
    service._lock = threading.Lock()
    return service


def _request(seed: int) -> dict[str, Any]:
    return {
        "image": "not-decoded-by-unit-test",
        "prompt": "pick up the red block",
        "domain_name": "libero",
        "image_size": 4,
        "seed": seed,
    }


def test_action_server_disables_generic_guardrails() -> None:
    checkpoint = server.CheckpointOverrides.model_construct(checkpoint_path="/checkpoints/edge")
    args = server.ActionServerArgs(checkpoint=checkpoint)

    setup_overrides = args.build_setup_overrides()

    assert setup_overrides.guardrails is False


def _patch_cpu_preprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server,
        "_decode_base64_png_to_rgb_uint8",
        lambda _encoded: torch.zeros((3, 4, 8), dtype=torch.uint8),
    )
    monkeypatch.setattr(server, "get_vision_data_resolution", lambda shape: shape)
    monkeypatch.setattr(server, "find_closest_target_size", lambda height, width, _resolution: (width, height))

    def _reflection_pad(
        data: dict[str, Any],
        _keys: list[str],
        _enabled: bool,
        target_width: int,
        target_height: int,
    ) -> None:
        data["image_size"] = torch.tensor([target_height, target_width], dtype=torch.long)

    monkeypatch.setattr(server, "reflection_pad_to_target", _reflection_pad)
    monkeypatch.setattr(server, "build_sequence_plan_from_mode", lambda **_kwargs: object())
    monkeypatch.setattr(server, "get_domain_id", lambda _domain_name: 0)

    real_stack = torch.stack

    class _CpuOnlyStack:
        def __init__(self, tensors: list[torch.Tensor]) -> None:
            self.value = real_stack(tensors)

        def to(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
            return self.value

    monkeypatch.setattr(server.torch, "stack", lambda tensors: _CpuOnlyStack(tensors))


def test_info_exposes_resolved_raw_dim_profile_and_hash() -> None:
    profile = _profile()
    info = _service(profile=profile).get_info()

    assert info["protocol_version"] == "cosmos-libero-eval-v1"
    assert info["raw_action_dim"] == profile.effective_action_dim
    assert info["policy_profile"] == profile.model_dump(mode="json")
    assert info["policy_profile"]["profile_hash"] == profile.profile_hash
    assert info["action_stats_sha256"] == profile.action_stats_sha256
    assert info["checkpoint_fingerprint"] == profile.checkpoint_fingerprint


@pytest.mark.parametrize("path", ["/", "/predict"])
def test_legacy_single_item_endpoint_is_explicitly_gone(path: str) -> None:
    handler = server._ActionHandler.__new__(server._ActionHandler)
    responses: list[tuple[int, dict[str, Any]]] = []
    handler.path = path
    handler._send_json = lambda status, payload: responses.append((status, payload))  # type: ignore[method-assign]

    handler.do_POST()

    assert responses[0][0] == 410
    assert "/predict_batch" in responses[0][1]["error"]


def test_json_prompt_is_byte_identical_to_training_parity_builder() -> None:
    service = _service()
    video = torch.zeros((3, service.cfg.action_chunk_size + 1, 8, 16), dtype=torch.uint8)
    image_size = torch.tensor([8, 16], dtype=torch.long)

    actual = service._build_json_prompt("pick up the red block", video=video, image_size=image_size)
    expected = build_libero_json_prompt(
        "pick up the red block",
        video=video,
        image_size=image_size,
        conditioning_fps=service.cfg.fps,
        action_chunk_size=service.cfg.action_chunk_size,
        idle_frames=0,
    )

    assert actual == expected
    assert json.loads(actual)["actions"][0]["description"] == "pick up the red block."


@pytest.mark.parametrize("invalid_seed", [True, -1, 2**63])
def test_request_seed_rejects_invalid_signed_int64(invalid_seed: object) -> None:
    service = _service()
    request = _request(0)
    request["seed"] = invalid_seed

    with pytest.raises(ValueError, match="seed"):
        service._prep_policy_item(request)


def test_batch_forwards_each_validated_per_item_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cpu_preprocessing(monkeypatch)

    class _SeedCapturingModel:
        input_video_key = "video"

        def __init__(self) -> None:
            self.seeds: list[int] | None = None

        def generate_samples_from_batch(
            self,
            _batch: dict[str, Any],
            *,
            seed: list[int],
            **_kwargs: Any,
        ) -> dict[str, list[torch.Tensor]]:
            self.seeds = seed
            return {"action": [torch.zeros((2, 10), dtype=torch.float32) for _ in seed]}

    model = _SeedCapturingModel()
    service = _service(model=model)
    result = service.predict_policy_batch([_request(11), _request(22)])

    assert model.seeds == [11, 22]
    assert len(result["actions"]) == 2
    assert result["checkpoint_fingerprint"] == service.profile.checkpoint_fingerprint


def test_shared_processing_record_denormalizes_exactly_once_per_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cpu_preprocessing(monkeypatch)

    class _CountingNormalizer:
        def __init__(self) -> None:
            self.denormalize_calls = 0

        def normalize_action(self, _action: torch.Tensor) -> torch.Tensor:
            raise AssertionError("inference must not normalize generated actions")

        def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
            self.denormalize_calls += 1
            return action + 5.0

    normalizer = _CountingNormalizer()

    class _ExternalizingModel:
        input_video_key = "video"

        def __init__(self) -> None:
            self.records_were_shared = False

        def generate_samples_from_batch(
            self,
            batch: dict[str, Any],
            *,
            seed: list[int],
            **_kwargs: Any,
        ) -> dict[str, list[torch.Tensor]]:
            records = batch["action_processing_record"]
            self.records_were_shared = len(records) == 2 and records[0] is records[1]
            model_actions = [torch.ones((2, 12), dtype=torch.float32) for _ in seed]
            external_actions = [
                ActionProcessor.postprocess_action(action, record)
                for action, record in zip(model_actions, records, strict=True)
            ]
            return {"action": external_actions}

    model = _ExternalizingModel()
    service = _service(model=model, normalizer=normalizer)
    result = service.predict_policy_batch([_request(1), _request(2)])

    assert model.records_were_shared
    assert normalizer.denormalize_calls == 2
    assert torch.equal(torch.tensor(result["actions"]), torch.full((2, 2, 10), 6.0))


@pytest.mark.parametrize(
    ("prediction", "error_match"),
    [
        (torch.zeros((2, 9), dtype=torch.float32), "must have shape"),
        (torch.full((2, 10), float("nan"), dtype=torch.float32), "finite"),
    ],
)
def test_batch_rejects_invalid_action_shape_or_values(
    monkeypatch: pytest.MonkeyPatch,
    prediction: torch.Tensor,
    error_match: str,
) -> None:
    _patch_cpu_preprocessing(monkeypatch)

    class _InvalidActionModel:
        input_video_key = "video"

        def generate_samples_from_batch(self, *_args: Any, **_kwargs: Any) -> dict[str, list[torch.Tensor]]:
            return {"action": [prediction]}

    service = _service(model=_InvalidActionModel())

    with pytest.raises(ValueError, match=error_match):
        service.predict_policy_batch([_request(3)])


@pytest.mark.parametrize("role", ["base", "finetuned"])
def test_base_and_finetuned_profiles_reject_conflicting_cli_overrides(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    profile = _profile(role=role)
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(server, "resolve_checkpoint_profile", lambda *_args, **_kwargs: profile)
    args = SimpleNamespace(
        checkpoint=SimpleNamespace(config_file=None, checkpoint_path=profile.checkpoint_path),
        policy_profile_path=Path("/profiles/policy.json") if role == "finetuned" else None,
        target_adapter_path=Path("/profiles/adapter.json") if role == "base" else None,
        weights_variant=None,
        fps=11,
        action_chunk_size=None,
        raw_action_dim=None,
        action_stats_path=None,
        action_normalization=None,
        format_prompt_as_json=None,
    )

    with pytest.raises(ValueError, match="CLI override fps=11.*resolved policy profile value 10"):
        server.ActionModelService(args)
