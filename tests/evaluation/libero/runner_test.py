# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from cosmos_framework.evaluation.libero import artifacts
from cosmos_framework.evaluation.libero.runner import (
    EDGE_ACTION_CHUNK_SIZE,
    METRICS_FILENAME,
    PROTOCOL_VERSION,
    BatchPolicyClient,
    EpisodeSpec,
    RunnerInfrastructureError,
    RunnerProtocolError,
    TransitionOutcome,
    build_arg_parser,
    build_policy_item,
    classify_transition,
    collect_runtime_provenance,
    configure_mujoco_environment,
    episode_id,
    finalize_run,
    libero_env_kwargs,
    parse_policy_handshake,
    parse_predict_batch_response,
    parse_task_ids,
    prepare_run,
    run_vectorized_state_machine,
    stable_decision_seed,
    stable_episode_seed,
    validate_completion_marker,
)
from cosmos_framework.evaluation.libero.schema import LiberoCheckpointProfile

pytestmark = pytest.mark.level(0)


def _profile(**overrides: Any) -> LiberoCheckpointProfile:
    values: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_path": "/checkpoints/edge",
        "checkpoint_format": "hf",
        "checkpoint_role": "finetuned",
        "zero_shot": False,
        "target_adapter_id": None,
        "domain_name": "libero",
        "action_chunk_size": 8,
        "conditioning_fps": 10.0,
        "effective_action_dim": 10,
        "action_space": "frame_wise_relative",
        "rotation_space": "6d",
        "pose_coordinate_frame": "native",
        "action_normalization": "quantile_rot",
        "action_stats_path": "/stats/libero.json",
        "action_stats_sha256": "74f63b4aaf9bc0623e8544d8c9fe9d3da343604096082a2621928fca039010f1",
        "checkpoint_fingerprint": "d" * 64,
        "format_prompt_as_json": True,
        "cameras": ["agentview", "wrist"],
        "image_size": 256,
        "rotate_images": True,
        "control_mode": "OSC_POSE",
        "control_frequency": 10,
        "gripper_mode": "pm_one",
        "weights_variant": "ema",
        "profile_hash": "b" * 64,
        "profile_sources": ["test"],
    }
    values.update(overrides)
    return LiberoCheckpointProfile.model_validate(values)


def _info(profile: LiberoCheckpointProfile | None = None, *, protocol: str = PROTOCOL_VERSION) -> dict[str, Any]:
    result: dict[str, Any] = {"protocol_version": protocol}
    if profile is not None:
        result["policy_profile"] = profile.model_dump(mode="json")
    return result


def _valid_chunk() -> np.ndarray:
    row = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0], dtype=np.float32)
    return np.repeat(row[None, :], EDGE_ACTION_CHUNK_SIZE, axis=0)


def _observation(value: int = 0) -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.full((4, 5, 3), value, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((6, 3, 3), value + 1, dtype=np.uint8),
    }


def test_handshake_reads_only_versioned_policy_profile() -> None:
    profile = _profile()

    handshake = parse_policy_handshake(_info(profile))

    assert handshake.profile == profile
    assert handshake.server_info["protocol_version"] == PROTOCOL_VERSION

    with pytest.raises(RunnerProtocolError, match="protocol_version"):
        parse_policy_handshake(_info(profile, protocol="legacy"))
    with pytest.raises(RunnerProtocolError, match="policy_profile"):
        parse_policy_handshake({"protocol_version": PROTOCOL_VERSION, "action_chunk_size": 8})
    with pytest.raises(RunnerProtocolError, match="gripper_mode"):
        parse_policy_handshake(_info(_profile(gripper_mode="passthrough")))
    with pytest.raises(RunnerProtocolError, match="rotate_images"):
        parse_policy_handshake(_info(_profile(rotate_images=False)))
    with pytest.raises(RunnerProtocolError, match="conditioning_fps"):
        parse_policy_handshake(_info(_profile(conditioning_fps=20.0)))
    with pytest.raises(RunnerProtocolError, match="control_frequency"):
        parse_policy_handshake(_info(_profile(control_frequency=20)))
    with pytest.raises(RunnerProtocolError, match="format_prompt_as_json"):
        parse_policy_handshake(_info(_profile(format_prompt_as_json=False)))
    with pytest.raises(RunnerProtocolError, match="action_normalization"):
        parse_policy_handshake(_info(_profile(action_normalization="meanstd")))
    with pytest.raises(RunnerProtocolError, match="action_stats_sha256"):
        parse_policy_handshake(_info(_profile(action_stats_sha256="a" * 64)))


def test_decision_seed_is_stable_and_depends_on_full_episode_identity() -> None:
    seed = stable_decision_seed(7, "libero_10", 2, 3, 4)

    assert seed == 4432341604508092275
    assert seed == stable_decision_seed(7, "libero_10", 2, 3, 4)
    assert 0 <= seed <= 2**63 - 1
    assert (
        len(
            {
                seed,
                stable_decision_seed(8, "libero_10", 2, 3, 4),
                stable_decision_seed(7, "libero_goal", 2, 3, 4),
                stable_decision_seed(7, "libero_10", 1, 3, 4),
                stable_decision_seed(7, "libero_10", 2, 4, 4),
                stable_decision_seed(7, "libero_10", 2, 3, 5),
            }
        )
        == 6
    )
    with pytest.raises(ValueError, match="decision_index"):
        stable_decision_seed(7, "libero_10", 2, 3, -1)


def test_episode_seed_is_stable_slot_independent_and_uint32() -> None:
    seed = stable_episode_seed(7, "libero_10", 2, 3)

    assert seed == stable_episode_seed(7, "libero_10", 2, 3)
    assert 0 <= seed <= 2**32 - 1
    assert seed != stable_episode_seed(7, "libero_10", 2, 4)


def test_runtime_provenance_is_dependency_injected_and_complete(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked-dependencies")

    def command_runner(command: list[str] | tuple[str, ...], cwd: Path) -> tuple[int, str, str]:
        assert cwd == tmp_path
        key = tuple(command)
        responses = {
            ("git", "rev-parse", "HEAD"): (0, "abc123", ""),
            ("git", "status", "--porcelain=v1", "--untracked-files=all"): (0, " M runner.py", ""),
            ("git", "diff", "--binary", "HEAD"): (0, "diff-content", ""),
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ): (0, "NVIDIA H100, 570.00", ""),
        }
        return responses[key]

    provenance = collect_runtime_provenance(
        repo_root=tmp_path,
        environ={"COSMOS_EVAL_IMAGE": "edge:sha256", "COSMOS_EVAL_JOB_ID": "job-7"},
        command_runner=command_runner,
        version_getter=lambda distribution: f"test-{distribution}",
    )

    assert provenance["git"]["head"] == "abc123"
    assert provenance["git"]["dirty"] is True
    assert len(provenance["git"]["diff_sha256"]) == 64
    assert provenance["uv_lock"]["sha256"] == hashlib.sha256(b"locked-dependencies").hexdigest()
    assert provenance["runtime_environment"] == {
        "cosmos_eval_image": "edge:sha256",
        "cosmos_eval_job_id": "job-7",
    }
    assert provenance["gpus"] == [{"name": "NVIDIA H100", "driver_version": "570.00"}]
    assert provenance["package_versions"]["libero"] == "test-libero"


def test_policy_item_preserves_raw_prompt_and_current_canonical_frame() -> None:
    profile = _profile()
    frame = np.zeros((256, 512, 3), dtype=np.uint8)
    frame[:, :256] = [1, 2, 3]
    frame[:, 256:] = [4, 5, 6]

    item = build_policy_item(frame, "  pick up the mug  ", profile, seed=123)

    assert item.keys() == {"image", "prompt", "domain_name", "image_size", "seed"}
    assert item["prompt"] == "  pick up the mug  "
    assert item["domain_name"] == "libero"
    assert item["image_size"] == 256
    assert item["seed"] == 123
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(item["image"]))).convert("RGB"))
    np.testing.assert_array_equal(decoded, frame)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"profile_hash": "c" * 64, "actions": [_valid_chunk().tolist()]}, "profile_hash"),
        (
            {"profile_hash": "b" * 64, "checkpoint_fingerprint": "d" * 64, "actions": []},
            "action chunks",
        ),
        (
            {
                "profile_hash": "b" * 64,
                "checkpoint_fingerprint": "d" * 64,
                "actions": [np.zeros((7, 10)).tolist()],
            },
            "shape",
        ),
        (
            {
                "profile_hash": "b" * 64,
                "checkpoint_fingerprint": "d" * 64,
                "actions": [np.full((8, 10), np.nan).tolist()],
            },
            "finite",
        ),
    ],
)
def test_batch_response_rejects_hash_cardinality_shape_and_nonfinite(payload: dict[str, Any], match: str) -> None:
    with pytest.raises(RunnerProtocolError, match=match):
        parse_predict_batch_response(payload, expected_batch_size=1, profile=_profile())


def test_batch_response_rejects_server_swap_after_handshake() -> None:
    handshake_profile = _profile()
    swapped_profile = _profile(checkpoint_fingerprint="e" * 64, profile_hash="f" * 64)
    payload = {
        "profile_hash": swapped_profile.profile_hash,
        "checkpoint_fingerprint": swapped_profile.checkpoint_fingerprint,
        "actions": [_valid_chunk().tolist()],
    }

    with pytest.raises(RunnerProtocolError, match="profile_hash"):
        parse_predict_batch_response(payload, expected_batch_size=1, profile=handshake_profile)

    with pytest.raises(RunnerProtocolError, match="checkpoint_fingerprint"):
        parse_predict_batch_response(
            {**payload, "profile_hash": handshake_profile.profile_hash},
            expected_batch_size=1,
            profile=handshake_profile,
        )


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class _FakeSession:
    def __init__(self, profile: LiberoCheckpointProfile) -> None:
        self.profile = profile
        self.get_calls: list[tuple[str, float]] = []
        self.post_calls: list[tuple[str, dict[str, Any], float]] = []

    def get(self, url: str, *, timeout: float) -> _FakeResponse:
        self.get_calls.append((url, timeout))
        return _FakeResponse(_info(self.profile))

    def post(self, url: str, *, json: dict[str, Any], timeout: float) -> _FakeResponse:
        self.post_calls.append((url, json, timeout))
        return _FakeResponse(
            {
                "profile_hash": self.profile.profile_hash,
                "checkpoint_fingerprint": self.profile.checkpoint_fingerprint,
                "actions": [_valid_chunk().tolist() for _ in json["items"]],
            }
        )


def test_http_client_uses_info_and_predict_batch_even_for_one_item() -> None:
    profile = _profile()
    session = _FakeSession(profile)
    client = BatchPolicyClient("http://policy/", timeout_s=12.0, session=session)

    handshake = client.handshake()
    chunks = client.predict_batch([{"seed": 1}], handshake.profile)

    assert session.get_calls == [("http://policy/info", 12.0)]
    assert session.post_calls[0][0] == "http://policy/predict_batch"
    assert session.post_calls[0][1] == {"items": [{"seed": 1}]}
    assert chunks[0].shape == (8, 10)


@pytest.mark.parametrize(
    ("reward", "done", "info", "expected"),
    [
        (1.0, False, {}, TransitionOutcome.SUCCESS),
        (0.0, False, {"success": True}, TransitionOutcome.SUCCESS),
        (0.0, True, {}, TransitionOutcome.FAILURE),
        (0.0, False, {}, TransitionOutcome.CONTINUE),
    ],
)
def test_transition_outcome_uses_reward_or_explicit_success(
    reward: float, done: bool, info: dict[str, Any], expected: TransitionOutcome
) -> None:
    assert classify_transition(reward, done, info) == expected


def test_environment_kwargs_explicitly_pin_osc_pose_and_control_frequency() -> None:
    kwargs = libero_env_kwargs(
        bddl_file_name="task.bddl",
        image_size=256,
        render_gpu_device_id=3,
        control_frequency=10,
        control_mode="OSC_POSE",
    )

    assert kwargs == {
        "bddl_file_name": "task.bddl",
        "camera_heights": 256,
        "camera_widths": 256,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "render_gpu_device_id": 3,
        "controller": "OSC_POSE",
        "control_freq": 10,
    }
    with pytest.raises(ValueError, match="OSC_POSE"):
        libero_env_kwargs(
            bddl_file_name="task.bddl",
            image_size=256,
            render_gpu_device_id=0,
            control_frequency=10,
            control_mode="JOINT_POSITION",
        )
    with pytest.raises(ValueError, match="render_gpu_device_id"):
        libero_env_kwargs(
            bddl_file_name="task.bddl",
            image_size=256,
            render_gpu_device_id=-1,
            control_frequency=10,
            control_mode="OSC_POSE",
        )
    with pytest.raises(ValueError, match="render_gpu_device_id"):
        configure_mujoco_environment("egl", -1)


class _FakeVectorEnv:
    def __init__(self, *, warmup: bool, outcomes: dict[int, tuple[float, bool, dict[str, Any]]]) -> None:
        self.warmup = warmup
        self.outcomes = outcomes
        self.seed_values: list[list[int]] = []
        self.step_calls: list[tuple[list[int], np.ndarray]] = []

    def seed(self, seed: list[int]) -> None:
        self.seed_values.append(list(seed))

    def reset(self, *, id: list[int]) -> list[dict[str, np.ndarray]]:  # noqa: A002
        return [_observation(slot) for slot in id]

    def set_init_state(self, states: np.ndarray, *, id: list[int]) -> list[dict[str, np.ndarray]]:  # noqa: A002
        assert states.shape[0] == len(id)
        return [_observation(slot) for slot in id]

    def step(self, actions: np.ndarray, *, id: list[int]) -> tuple[list[Any], list[Any], list[Any], list[Any]]:  # noqa: A002
        self.step_calls.append((list(id), actions.copy()))
        observations = [_observation(20 + slot + len(self.step_calls)) for slot in id]
        if self.warmup and len(self.step_calls) == 1:
            return observations, [0.0] * len(id), [False] * len(id), [{} for _ in id]
        rewards, dones, infos = zip(*(self.outcomes[slot] for slot in id), strict=True)
        return observations, list(rewards), list(dones), list(infos)


class _FakePolicyClient:
    def __init__(self) -> None:
        self.items: list[list[dict[str, Any]]] = []

    def predict_batch(self, items: list[dict[str, Any]], profile: LiberoCheckpointProfile) -> list[np.ndarray]:
        assert profile.profile_hash == "b" * 64
        self.items.append(items)
        return [_valid_chunk() for _ in items]


class _FailingPolicyClient:
    def predict_batch(self, items: list[dict[str, Any]], profile: LiberoCheckpointProfile) -> list[np.ndarray]:
        raise ConnectionError("policy server unavailable")


def test_vectorized_state_machine_batches_slots_and_treats_done_without_success_as_failure() -> None:
    env = _FakeVectorEnv(
        warmup=True,
        outcomes={
            0: (1.0, False, {}),
            1: (0.0, True, {}),
        },
    )
    client = _FakePolicyClient()
    appended: list[dict[str, Any]] = []

    results = run_vectorized_state_machine(
        vector_env=env,
        policy_client=client,
        profile=_profile(),
        task_suite="libero_goal",
        task_id=2,
        task_prompt="Open the drawer",
        episodes=[EpisodeSpec(0, np.zeros(3)), EpisodeSpec(1, np.ones(3))],
        num_envs=2,
        base_seed=11,
        action_horizon=4,
        max_steps=10,
        warmup_steps=1,
        on_episode=lambda record: appended.append(dict(record)),
        on_infra_error=lambda record: pytest.fail(f"unexpected infra error: {record}"),
    )

    assert env.seed_values == [
        [
            stable_episode_seed(11, "libero_goal", 2, 0),
            stable_episode_seed(11, "libero_goal", 2, 1),
        ]
    ]
    assert len(client.items) == 1
    assert len(client.items[0]) == 2
    assert [item["prompt"] for item in client.items[0]] == ["Open the drawer", "Open the drawer"]
    assert [item["seed"] for item in client.items[0]] == [
        stable_decision_seed(11, "libero_goal", 2, 0, 0),
        stable_decision_seed(11, "libero_goal", 2, 1, 0),
    ]
    assert env.step_calls[0][1].shape == (2, 7)
    np.testing.assert_array_equal(env.step_calls[0][1], np.repeat(np.array([[0, 0, 0, 0, 0, 0, -1]]), 2, 0))
    assert env.step_calls[1][1].shape == (2, 7)
    assert [record["success"] for record in results] == [True, False]
    assert results[1]["termination"] == "done_without_success"
    assert appended == results


def test_serial_num_envs_one_uses_the_same_batch_state_machine() -> None:
    env = _FakeVectorEnv(warmup=False, outcomes={0: (0.0, True, {"success": True})})
    client = _FakePolicyClient()
    records: list[dict[str, Any]] = []

    run_vectorized_state_machine(
        vector_env=env,
        policy_client=client,
        profile=_profile(),
        task_suite="libero_spatial",
        task_id=0,
        task_prompt="Pick up the object",
        episodes=[EpisodeSpec(3, np.zeros(3))],
        num_envs=1,
        base_seed=0,
        action_horizon=1,
        max_steps=5,
        warmup_steps=0,
        on_episode=lambda record: records.append(dict(record)),
        on_infra_error=lambda record: pytest.fail(f"unexpected infra error: {record}"),
    )

    assert [len(items) for items in client.items] == [1]
    assert records[0]["success"] is True
    assert records[0]["steps"] == 1


def test_warmup_does_not_consume_policy_step_budget() -> None:
    env = _FakeVectorEnv(warmup=True, outcomes={0: (0.0, False, {})})
    client = _FakePolicyClient()
    records: list[dict[str, Any]] = []

    run_vectorized_state_machine(
        vector_env=env,
        policy_client=client,
        profile=_profile(),
        task_suite="libero_spatial",
        task_id=0,
        task_prompt="Pick up the object",
        episodes=[EpisodeSpec(0, np.zeros(3))],
        num_envs=1,
        base_seed=0,
        action_horizon=4,
        max_steps=1,
        warmup_steps=1,
        on_episode=lambda record: records.append(dict(record)),
        on_infra_error=lambda record: pytest.fail(f"unexpected infra error: {record}"),
    )

    assert len(client.items) == 1
    assert len(env.step_calls) == 2
    assert records[0]["steps"] == 1
    assert records[0]["termination"] == "max_steps"


def test_episode_and_policy_seeds_are_resume_and_slot_invariant() -> None:
    full_env = _FakeVectorEnv(
        warmup=False,
        outcomes={0: (1.0, False, {}), 1: (1.0, False, {})},
    )
    full_client = _FakePolicyClient()
    run_vectorized_state_machine(
        vector_env=full_env,
        policy_client=full_client,
        profile=_profile(),
        task_suite="libero_goal",
        task_id=4,
        task_prompt="Open the drawer",
        episodes=[EpisodeSpec(0, np.zeros(3)), EpisodeSpec(7, np.ones(3))],
        num_envs=2,
        base_seed=19,
        action_horizon=1,
        max_steps=2,
        warmup_steps=0,
        on_episode=lambda record: None,
        on_infra_error=lambda record: pytest.fail(f"unexpected infra error: {record}"),
    )

    resumed_env = _FakeVectorEnv(warmup=False, outcomes={0: (1.0, False, {})})
    resumed_client = _FakePolicyClient()
    run_vectorized_state_machine(
        vector_env=resumed_env,
        policy_client=resumed_client,
        profile=_profile(),
        task_suite="libero_goal",
        task_id=4,
        task_prompt="Open the drawer",
        episodes=[EpisodeSpec(7, np.ones(3))],
        num_envs=1,
        base_seed=19,
        action_horizon=1,
        max_steps=2,
        warmup_steps=0,
        on_episode=lambda record: None,
        on_infra_error=lambda record: pytest.fail(f"unexpected infra error: {record}"),
    )

    assert full_env.seed_values[0][1] == resumed_env.seed_values[0][0]
    assert full_client.items[0][1]["seed"] == resumed_client.items[0][0]["seed"]


@pytest.fixture
def output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr(artifacts, "OUTPUT_ROOT", root)
    return root


def test_run_artifacts_resume_aggregate_and_seal(output_root: Path) -> None:
    run_dir = output_root / "libero-eval" / "run-1"
    manifest = {"protocol_version": PROTOCOL_VERSION, "profile_hash": "b" * 64}

    first = prepare_run(run_dir, manifest)
    assert not first.complete
    assert first.completed_episode_ids == frozenset()
    artifacts.append_episode(
        run_dir,
        {
            "episode_id": episode_id("libero_10", 0, 0),
            "task_suite": "libero_10",
            "task_id": 0,
            "success": True,
            "infra_error": False,
        },
    )
    artifacts.append_episode(
        run_dir,
        {
            "episode_id": episode_id("libero_10", 0, 1),
            "task_suite": "libero_10",
            "task_id": 0,
            "success": False,
            "infra_error": False,
        },
    )

    resumed = prepare_run(run_dir, manifest)
    assert resumed.completed_episode_ids == {
        episode_id("libero_10", 0, 0),
        episode_id("libero_10", 0, 1),
    }
    summary = finalize_run(run_dir, profile_hash="b" * 64)

    assert summary["overall"]["success_rate"] == 0.5
    assert json.loads((run_dir / METRICS_FILENAME).read_text()) == summary
    assert (run_dir / METRICS_FILENAME).stat().st_mode & 0o777 == 0o644
    assert artifacts.is_complete(run_dir)
    assert artifacts.read_completion_marker(run_dir)["metrics"] == METRICS_FILENAME  # type: ignore[index]
    assert prepare_run(run_dir, manifest).complete
    with pytest.raises(ValueError, match="manifest"):
        prepare_run(run_dir, {**manifest, "profile_hash": "c" * 64})


def test_resume_allows_only_job_id_to_change_in_provenance(output_root: Path) -> None:
    run_dir = output_root / "libero-eval" / "provenance-resume"
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "provenance": {"runtime_environment": {"cosmos_eval_image": "edge:v1", "cosmos_eval_job_id": "job-1"}},
    }
    prepare_run(run_dir, manifest)

    resumed = prepare_run(
        run_dir,
        {
            **manifest,
            "provenance": {"runtime_environment": {"cosmos_eval_image": "edge:v1", "cosmos_eval_job_id": "job-2"}},
        },
    )
    assert not resumed.complete

    with pytest.raises(ValueError, match="manifest"):
        prepare_run(
            run_dir,
            {
                **manifest,
                "provenance": {"runtime_environment": {"cosmos_eval_image": "edge:v2", "cosmos_eval_job_id": "job-2"}},
            },
        )


def test_infra_attempt_is_retryable_and_does_not_seal_or_complete_episode(output_root: Path) -> None:
    run_dir = output_root / "libero-eval" / "retry"
    manifest = {"protocol_version": PROTOCOL_VERSION, "profile_hash": "b" * 64}
    prepare_run(run_dir, manifest)
    spec = EpisodeSpec(0, np.zeros(3))

    def append_infra(record: dict[str, Any]) -> None:
        artifacts.append_infra_error(run_dir, {**record, "attempt_id": "attempt-1"})

    with pytest.raises(RunnerInfrastructureError, match="policy_server"):
        run_vectorized_state_machine(
            vector_env=_FakeVectorEnv(warmup=False, outcomes={0: (0.0, False, {})}),
            policy_client=_FailingPolicyClient(),
            profile=_profile(),
            task_suite="libero_10",
            task_id=0,
            task_prompt="Move the object",
            episodes=[spec],
            num_envs=1,
            base_seed=0,
            action_horizon=1,
            max_steps=2,
            warmup_steps=0,
            on_episode=lambda record: artifacts.append_episode(run_dir, record),
            on_infra_error=append_infra,
        )

    assert artifacts.read_episodes(run_dir) == []
    assert len(artifacts.read_infra_errors(run_dir)) == 1
    assert prepare_run(run_dir, manifest).completed_episode_ids == frozenset()
    assert not artifacts.is_complete(run_dir)
    with pytest.raises(RunnerInfrastructureError, match="remain retryable"):
        finalize_run(run_dir, profile_hash="b" * 64)
    assert not artifacts.is_complete(run_dir)

    run_vectorized_state_machine(
        vector_env=_FakeVectorEnv(warmup=False, outcomes={0: (1.0, False, {})}),
        policy_client=_FakePolicyClient(),
        profile=_profile(),
        task_suite="libero_10",
        task_id=0,
        task_prompt="Move the object",
        episodes=[spec],
        num_envs=1,
        base_seed=0,
        action_horizon=1,
        max_steps=2,
        warmup_steps=0,
        on_episode=lambda record: artifacts.append_episode(run_dir, record),
        on_infra_error=lambda record: pytest.fail(f"unexpected infra error: {record}"),
    )
    summary = finalize_run(run_dir, profile_hash="b" * 64)

    assert len(artifacts.read_episodes(run_dir)) == 1
    assert summary["overall"]["evaluated_episodes"] == 1
    assert summary["overall"]["infra_errors"] == 0
    assert summary["infrastructure_errors"]["count"] == 1
    assert artifacts.is_complete(run_dir)


def test_completion_marker_validation_is_strict() -> None:
    valid = {
        "completed_at": "2026-08-18T00:00:00+00:00",
        "episode_count": 2,
        "metrics": METRICS_FILENAME,
        "profile_hash": "b" * 64,
    }
    validate_completion_marker(valid, expected_episode_count=2, profile_hash="b" * 64)

    invalid_markers = [
        None,
        {},
        {**valid, "completed_at": ""},
        {**valid, "episode_count": 1},
        {**valid, "episode_count": True},
        {**valid, "metrics": "other.json"},
        {**valid, "profile_hash": "c" * 64},
    ]
    for marker in invalid_markers:
        with pytest.raises(ValueError):
            validate_completion_marker(marker, expected_episode_count=2, profile_hash="b" * 64)


def test_task_id_selection_and_cli_surface() -> None:
    assert parse_task_ids("", num_tasks=3) == [0, 1, 2]
    assert parse_task_ids("2,0", num_tasks=3) == [2, 0]
    with pytest.raises(ValueError, match="duplicates"):
        parse_task_ids("1,1", num_tasks=3)
    with pytest.raises(ValueError, match="out of range"):
        parse_task_ids("3", num_tasks=3)

    args = build_arg_parser().parse_args(
        [
            "--server-url",
            "http://policy:8000",
            "--task-suite",
            "libero_goal",
            "--task-ids",
            "1,3",
            "--trials",
            "5",
            "--num-envs",
            "2",
            "--seed",
            "9",
            "--max-steps",
            "300",
            "--warmup-steps",
            "10",
            "--mujoco-gl",
            "egl",
            "--render-gpu-device-id",
            "1",
            "--output-dir",
            "/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/eval/run",
        ]
    )
    assert args.task_suite == "libero_goal"
    assert args.task_ids == "1,3"
    assert args.trials == 5
    assert args.output_dir.startswith(str(Path("/mnt/shared-storage-user")))
