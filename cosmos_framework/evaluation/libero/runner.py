# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Strict, resumable closed-loop LIBERO evaluation for Cosmos Edge policies."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image
from pydantic import ValidationError

from cosmos_framework.evaluation.libero import artifacts
from cosmos_framework.evaluation.libero.action import (
    framewise_rot6d_to_libero,
    remap_gripper_pm_one,
    validate_action_chunk,
)
from cosmos_framework.evaluation.libero.metrics import aggregate_episode_metrics
from cosmos_framework.evaluation.libero.observation import (
    EDGE_LIBERO_CAMERAS,
    EDGE_LIBERO_IMAGE_SIZE,
    build_libero_concat_frame,
)
from cosmos_framework.evaluation.libero.schema import LiberoCheckpointProfile

PROTOCOL_VERSION = "cosmos-libero-eval-v1"
METRICS_FILENAME = "metrics.json"
EDGE_ACTION_CHUNK_SIZE = 8
EDGE_ACTION_DIM = 10
LIBERO_ENV_ACTION_DIM = 7
EDGE_TARGET_ADAPTER_PATH = Path(__file__).with_name("profiles") / "edge_libero_target_adapter.json"

TASK_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

_DUMMY_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


class RunnerProtocolError(RuntimeError):
    """Raised when the policy server violates the evaluation protocol."""


class RunnerInfrastructureError(RuntimeError):
    """Raised after a retryable infrastructure attempt is durably recorded."""


class TransitionOutcome(str, Enum):
    CONTINUE = "continue"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class PolicyHandshake:
    profile: LiberoCheckpointProfile
    server_info: dict[str, Any]


@dataclass(frozen=True)
class EpisodeSpec:
    trial_id: int
    initial_state: np.ndarray


@dataclass
class _ActiveEpisode:
    trial_id: int
    initial_state: np.ndarray
    observation: Mapping[str, Any] | None = None
    steps: int = 0
    decisions: int = 0
    started_at: float = 0.0


@dataclass(frozen=True)
class RunResumeState:
    run_dir: Path
    completed_episode_ids: frozenset[str]
    complete: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_provenance_command(command: Sequence[str], cwd: Path) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return 1, "", str(error)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def collect_runtime_provenance(
    *,
    repo_root: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    command_runner: Callable[[Sequence[str], Path], tuple[int, str, str]] | None = None,
    version_getter: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Collect reproducibility metadata without making git or GPU availability fatal."""
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[3]
    environment = dict(os.environ if environ is None else environ)
    run_command = command_runner or _default_provenance_command
    get_version = version_getter or importlib.metadata.version

    head_rc, head, head_error = run_command(["git", "rev-parse", "HEAD"], root)
    status_rc, status, status_error = run_command(["git", "status", "--porcelain=v1", "--untracked-files=all"], root)
    git_info: dict[str, Any] = {
        "head": head if head_rc == 0 and head else None,
        "dirty": bool(status) if status_rc == 0 else None,
    }
    if head_rc != 0 or status_rc != 0:
        git_info["error"] = "; ".join(error for error in (head_error, status_error) if error)
    elif status:
        diff_rc, diff, diff_error = run_command(["git", "diff", "--binary", "HEAD"], root)
        if diff_rc == 0:
            git_info["diff_sha256"] = hashlib.sha256(f"{status}\0{diff}".encode()).hexdigest()
        elif diff_error:
            git_info["diff_error"] = diff_error

    lock_path = root / "uv.lock"
    uv_lock = {
        "path": str(lock_path),
        "sha256": _sha256_file(lock_path) if lock_path.is_file() else None,
    }
    package_versions: dict[str, str | None] = {}
    for distribution in ("cosmos-framework", "libero", "robosuite", "mujoco", "numpy", "torch"):
        try:
            package_versions[distribution] = get_version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = None
        except Exception:
            package_versions[distribution] = None

    gpu_rc, gpu_output, gpu_error = run_command(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], root
    )
    gpus: list[dict[str, str]] = []
    if gpu_rc == 0:
        for line in gpu_output.splitlines():
            name, separator, driver = line.rpartition(",")
            if separator:
                gpus.append({"name": name.strip(), "driver_version": driver.strip()})

    return {
        "git": git_info,
        "uv_lock": uv_lock,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "package_versions": package_versions,
        "runtime_environment": {
            "cosmos_eval_image": environment.get("COSMOS_EVAL_IMAGE"),
            "cosmos_eval_job_id": environment.get("COSMOS_EVAL_JOB_ID"),
        },
        "gpus": gpus,
        "nvidia_smi_error": gpu_error if gpu_rc != 0 else None,
    }


@lru_cache(maxsize=1)
def _edge_target_expectations() -> dict[str, Any]:
    """Load the canonical adapter once so runner and resolver cannot drift."""
    try:
        with EDGE_TARGET_ADAPTER_PATH.open(encoding="utf-8") as stream:
            adapter = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerProtocolError(f"Could not load canonical Edge target adapter: {error}") from error
    if not isinstance(adapter, dict):
        raise RunnerProtocolError("Canonical Edge target adapter must contain a JSON object.")
    fields = (
        "domain_name",
        "action_chunk_size",
        "conditioning_fps",
        "effective_action_dim",
        "action_space",
        "rotation_space",
        "pose_coordinate_frame",
        "action_normalization",
        "action_stats_sha256",
        "format_prompt_as_json",
        "cameras",
        "image_size",
        "rotate_images",
        "control_mode",
        "control_frequency",
        "gripper_mode",
    )
    missing = [field for field in fields if field not in adapter]
    if missing:
        raise RunnerProtocolError(f"Canonical Edge target adapter is missing fields: {missing}.")
    expected = {field: adapter[field] for field in fields}
    expected["cameras"] = tuple(expected["cameras"])
    expected["target_adapter_id"] = adapter.get("adapter_id")
    return expected


def validate_edge_policy_profile(profile: LiberoCheckpointProfile) -> None:
    """Reject any policy contract that is not the canonical Edge target."""
    expected = _edge_target_expectations()
    for name, expected_value in expected.items():
        if name == "target_adapter_id" and profile.target_adapter_id is None:
            continue
        actual_value = getattr(profile, name)
        if actual_value != expected_value:
            raise RunnerProtocolError(
                f"Unsupported policy profile: {name} must be {expected_value!r}, got {actual_value!r}."
            )


def parse_policy_handshake(info: Mapping[str, Any]) -> PolicyHandshake:
    """Parse only the versioned ``/info.policy_profile`` contract."""
    if not isinstance(info, Mapping):
        raise RunnerProtocolError("GET /info must return a JSON object.")
    protocol = info.get("protocol_version")
    if protocol != PROTOCOL_VERSION:
        raise RunnerProtocolError(f"Expected protocol_version={PROTOCOL_VERSION!r}, got {protocol!r}.")
    raw_profile = info.get("policy_profile")
    if not isinstance(raw_profile, Mapping):
        raise RunnerProtocolError("GET /info must contain an object at 'policy_profile'.")
    try:
        profile = LiberoCheckpointProfile.model_validate(raw_profile)
    except ValidationError as error:
        raise RunnerProtocolError(f"Invalid /info.policy_profile: {error}") from error
    validate_edge_policy_profile(profile)
    return PolicyHandshake(profile=profile, server_info=dict(info))


def stable_decision_seed(
    base_seed: int,
    task_suite: str,
    task_id: int,
    trial_id: int,
    decision_index: int,
) -> int:
    """Return a process-independent signed-64-bit policy seed."""
    integer_fields = {
        "base_seed": base_seed,
        "task_id": task_id,
        "trial_id": trial_id,
        "decision_index": decision_index,
    }
    for name, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    if not isinstance(task_suite, str) or not task_suite.strip():
        raise ValueError("task_suite must be a non-empty string.")
    payload = json.dumps(
        {
            "base_seed": base_seed,
            "task_suite": task_suite,
            "task_id": task_id,
            "trial_id": trial_id,
            "decision_index": decision_index,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def stable_episode_seed(base_seed: int, task_suite: str, task_id: int, trial_id: int) -> int:
    """Return a slot-independent uint32 seed for one simulator episode."""
    # Reuse the strict identity validation, while domain-separating simulator
    # randomness from policy-decision randomness.
    stable_decision_seed(base_seed, task_suite, task_id, trial_id, 0)
    payload = json.dumps(
        {
            "base_seed": base_seed,
            "purpose": "libero_environment",
            "task_suite": task_suite,
            "task_id": task_id,
            "trial_id": trial_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def episode_id(task_suite: str, task_id: int, trial_id: int) -> str:
    """Build the stable identity used by the append-only episode journal."""
    if not isinstance(task_suite, str) or not task_suite.strip():
        raise ValueError("task_suite must be a non-empty string.")
    for name, value in (("task_id", task_id), ("trial_id", trial_id)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return f"{task_suite}:task={task_id}:trial={trial_id}"


def _encode_png(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_policy_item(
    current_concat_frame: np.ndarray,
    task_prompt: str,
    profile: LiberoCheckpointProfile,
    seed: int,
) -> dict[str, Any]:
    """Build one `/predict_batch` item without augmenting the raw task text."""
    validate_edge_policy_profile(profile)
    if not isinstance(task_prompt, str) or not task_prompt.strip():
        raise ValueError("task_prompt must be a non-empty raw task string.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1:
        raise ValueError(f"seed must be a non-negative signed-64-bit integer, got {seed!r}.")
    frame = np.asarray(current_concat_frame)
    expected_shape = (EDGE_LIBERO_IMAGE_SIZE, EDGE_LIBERO_IMAGE_SIZE * len(EDGE_LIBERO_CAMERAS), 3)
    if frame.shape != expected_shape or frame.dtype != np.uint8:
        raise ValueError(
            f"current_concat_frame must be uint8 with shape {expected_shape}, got {frame.dtype} {frame.shape}."
        )
    return {
        "image": _encode_png(np.ascontiguousarray(frame)),
        "prompt": task_prompt,
        "domain_name": profile.domain_name,
        "image_size": profile.image_size,
        "seed": seed,
    }


def parse_predict_batch_response(
    payload: Mapping[str, Any],
    *,
    expected_batch_size: int,
    profile: LiberoCheckpointProfile,
) -> list[np.ndarray]:
    """Validate response identity, batch cardinality, shape, and finiteness."""
    if not isinstance(payload, Mapping):
        raise RunnerProtocolError("POST /predict_batch must return a JSON object.")
    if payload.get("profile_hash") != profile.profile_hash:
        raise RunnerProtocolError(
            f"/predict_batch profile_hash mismatch: expected {profile.profile_hash!r}, "
            f"got {payload.get('profile_hash')!r}."
        )
    if payload.get("checkpoint_fingerprint") != profile.checkpoint_fingerprint:
        raise RunnerProtocolError(
            "/predict_batch checkpoint_fingerprint mismatch: "
            f"expected {profile.checkpoint_fingerprint!r}, got {payload.get('checkpoint_fingerprint')!r}."
        )
    actions = payload.get("actions")
    if not isinstance(actions, list) or len(actions) != expected_batch_size:
        actual_size = len(actions) if isinstance(actions, list) else None
        raise RunnerProtocolError(f"/predict_batch must return {expected_batch_size} action chunks, got {actual_size}.")
    validated: list[np.ndarray] = []
    for index, chunk in enumerate(actions):
        try:
            validated.append(
                validate_action_chunk(
                    chunk,
                    expected_horizon=profile.action_chunk_size,
                    expected_action_dim=profile.effective_action_dim,
                )
            )
        except ValueError as error:
            raise RunnerProtocolError(f"Invalid action chunk at batch index {index}: {error}") from error
    return validated


class BatchPolicyClient:
    """HTTP client restricted to `/info` and `/predict_batch`."""

    def __init__(self, server_url: str, *, timeout_s: float, session: Any | None = None) -> None:
        if not isinstance(server_url, str) or not server_url.strip():
            raise ValueError("server_url must be non-empty.")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive and finite, got {timeout_s!r}.")
        self.server_url = server_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.session = session if session is not None else requests.Session()

    @staticmethod
    def _json_object(response: Any, endpoint: str) -> dict[str, Any]:
        response.raise_for_status()
        try:
            value = response.json()
        except Exception as error:
            raise RunnerProtocolError(f"{endpoint} returned invalid JSON: {error}") from error
        if not isinstance(value, dict):
            raise RunnerProtocolError(f"{endpoint} must return a JSON object.")
        return value

    def handshake(self) -> PolicyHandshake:
        response = self.session.get(f"{self.server_url}/info", timeout=self.timeout_s)
        return parse_policy_handshake(self._json_object(response, "GET /info"))

    def predict_batch(
        self,
        items: Sequence[Mapping[str, Any]],
        profile: LiberoCheckpointProfile,
    ) -> list[np.ndarray]:
        if not items:
            raise ValueError("predict_batch requires at least one item.")
        response = self.session.post(
            f"{self.server_url}/predict_batch",
            json={"items": [dict(item) for item in items]},
            timeout=self.timeout_s,
        )
        payload = self._json_object(response, "POST /predict_batch")
        return parse_predict_batch_response(payload, expected_batch_size=len(items), profile=profile)


def classify_transition(reward: Any, done: Any, info: Any) -> TransitionOutcome:
    """Classify one simulator transition using the official success signals."""
    reward_array = np.asarray(reward)
    if reward_array.size != 1:
        raise ValueError(f"reward must be scalar, got shape {reward_array.shape}.")
    reward_value = float(reward_array.reshape(-1)[0])
    if not math.isfinite(reward_value):
        raise ValueError(f"reward must be finite, got {reward_value!r}.")
    done_array = np.asarray(done)
    if done_array.size != 1:
        raise ValueError(f"done must be scalar, got shape {done_array.shape}.")
    done_value = bool(done_array.reshape(-1)[0])

    info_success = False
    if info is not None:
        if not isinstance(info, Mapping):
            raise ValueError(f"info must be a mapping or None, got {type(info).__name__}.")
        info_success = bool(info.get("success", False))
    if reward_value > 0.0 or info_success:
        return TransitionOutcome.SUCCESS
    if done_value:
        return TransitionOutcome.FAILURE
    return TransitionOutcome.CONTINUE


def _slot_values(value: Any, count: int, name: str) -> list[Any]:
    if count == 1 and (value is None or isinstance(value, Mapping) or np.asarray(value).ndim == 0):
        return [value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.shape[0] != count:
            raise ValueError(f"Vector env {name} must contain {count} values, got shape {value.shape}.")
        return [value[index] for index in range(count)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == count:
        return list(value)
    raise ValueError(f"Vector env {name} must contain {count} values, got {type(value).__name__}.")


def _vector_step(vector_env: Any, slots: list[int], actions: list[np.ndarray]) -> list[tuple[Any, Any, Any, Any]]:
    result = vector_env.step(np.stack(actions), id=slots)
    if not isinstance(result, Sequence) or len(result) != 4:
        raise ValueError("Vector env step must return (observations, rewards, dones, infos).")
    observations, rewards, dones, infos = result
    count = len(slots)
    split = [
        _slot_values(observations, count, "observations"),
        _slot_values(rewards, count, "rewards"),
        _slot_values(dones, count, "dones"),
        _slot_values(infos, count, "infos"),
    ]
    return list(zip(*split, strict=True))


def _episode_record(
    state: _ActiveEpisode,
    *,
    task_suite: str,
    task_id: int,
    task_prompt: str,
    success: bool | None,
    termination: str,
    infra_error_type: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    infra_error = infra_error_type is not None
    record: dict[str, Any] = {
        "episode_id": episode_id(task_suite, task_id, state.trial_id),
        "task_suite": task_suite,
        "task_id": task_id,
        "task_description": task_prompt,
        "trial_id": state.trial_id,
        "steps": state.steps,
        "decisions": state.decisions,
        "termination": termination,
        "infra_error": infra_error,
        "elapsed_s": round(max(0.0, time.perf_counter() - state.started_at), 3),
    }
    if infra_error:
        record["infra_error_type"] = infra_error_type
        if error is not None:
            record["error"] = error
    else:
        if not isinstance(success, bool):
            raise ValueError("Non-infrastructure episode records require boolean success.")
        record["success"] = success
    return record


def run_vectorized_state_machine(
    *,
    vector_env: Any,
    policy_client: Any,
    profile: LiberoCheckpointProfile,
    task_suite: str,
    task_id: int,
    task_prompt: str,
    episodes: Sequence[EpisodeSpec],
    num_envs: int,
    base_seed: int,
    action_horizon: int,
    max_steps: int,
    warmup_steps: int,
    on_episode: Callable[[Mapping[str, Any]], Any],
    on_infra_error: Callable[[Mapping[str, Any]], Any],
) -> list[dict[str, Any]]:
    """Run serial or parallel rollouts through one identical batched state machine."""
    validate_edge_policy_profile(profile)
    integer_args = {
        "num_envs": num_envs,
        "action_horizon": action_horizon,
        "max_steps": max_steps,
    }
    for name, value in integer_args.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if action_horizon > profile.action_chunk_size:
        raise ValueError(
            f"action_horizon must not exceed profile chunk size {profile.action_chunk_size}, got {action_horizon}."
        )
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError(f"warmup_steps must be a non-negative integer, got {warmup_steps!r}.")
    if num_envs > len(episodes) and episodes:
        num_envs = len(episodes)
    if not episodes:
        return []

    results: list[dict[str, Any]] = []
    states: dict[int, _ActiveEpisode] = {}

    def finish(
        slot: int,
        *,
        success: bool | None,
        termination: str,
        infra_error_type: str | None = None,
        error: str | None = None,
    ) -> None:
        state = states.pop(slot)
        record = _episode_record(
            state,
            task_suite=task_suite,
            task_id=task_id,
            task_prompt=task_prompt,
            success=success,
            termination=termination,
            infra_error_type=infra_error_type,
            error=error,
        )
        if infra_error_type is None:
            on_episode(record)
        else:
            on_infra_error(record)
        results.append(record)

    def finish_infra(slots: Sequence[int], error_type: str, error: BaseException | str) -> None:
        message = str(error)
        recorded = False
        for slot in list(slots):
            if slot in states:
                finish(
                    slot,
                    success=None,
                    termination="infrastructure_error",
                    infra_error_type=error_type,
                    error=message,
                )
                recorded = True
        if recorded:
            raise RunnerInfrastructureError(f"{error_type}: {message}")

    def step_slots(slots: list[int], actions: list[np.ndarray], *, count_toward_budget: bool) -> None:
        try:
            transitions = _vector_step(vector_env, slots, actions)
        except Exception as error:
            finish_infra(slots, "environment_step", error)
            return
        for slot, (observation, reward, done, info) in zip(slots, transitions, strict=True):
            if slot not in states:
                continue
            state = states[slot]
            if not isinstance(observation, Mapping):
                finish_infra([slot], "environment_observation", "observation is not a mapping")
                continue
            state.observation = observation
            if count_toward_budget:
                state.steps += 1
            try:
                outcome = classify_transition(reward, done, info)
            except ValueError as error:
                finish_infra([slot], "environment_transition", error)
                continue
            if outcome == TransitionOutcome.SUCCESS:
                finish(slot, success=True, termination="success_signal")
            elif outcome == TransitionOutcome.FAILURE:
                finish(slot, success=False, termination="done_without_success")
            elif count_toward_budget and state.steps >= max_steps:
                finish(slot, success=False, termination="max_steps")

    for wave_start in range(0, len(episodes), num_envs):
        wave = list(episodes[wave_start : wave_start + num_envs])
        slots = list(range(len(wave)))
        now = time.perf_counter()
        states = {
            slot: _ActiveEpisode(spec.trial_id, np.asarray(spec.initial_state), started_at=now)
            for slot, spec in zip(slots, wave, strict=True)
        }
        try:
            # LIBERO maps a seed list positionally to workers. Re-seed every
            # wave from episode identity so compaction during resume cannot
            # change a trial's simulator randomness.
            environment_seeds = [
                stable_episode_seed(base_seed, task_suite, task_id, states[slot].trial_id) for slot in slots
            ]
            environment_seeds.extend([0] * (num_envs - len(environment_seeds)))
            vector_env.seed(environment_seeds)
        except Exception as error:
            finish_infra(slots, "environment_seed", error)
            continue
        try:
            vector_env.reset(id=slots)
            initial_states = np.stack([states[slot].initial_state for slot in slots])
            observations = _slot_values(
                vector_env.set_init_state(initial_states, id=slots),
                len(slots),
                "initial observations",
            )
            for slot, observation in zip(slots, observations, strict=True):
                if not isinstance(observation, Mapping):
                    raise ValueError("Initial observation is not a mapping.")
                states[slot].observation = observation
        except Exception as error:
            finish_infra(slots, "environment_reset", error)
            continue

        for _ in range(warmup_steps):
            active_slots = sorted(states)
            if not active_slots:
                break
            step_slots(
                active_slots,
                [_DUMMY_ACTION.copy() for _ in active_slots],
                count_toward_budget=False,
            )

        while states:
            for slot in list(states):
                if states[slot].steps >= max_steps:
                    finish(slot, success=False, termination="max_steps")
            active_slots = sorted(states)
            if not active_slots:
                break

            request_slots: list[int] = []
            items: list[dict[str, Any]] = []
            for slot in active_slots:
                state = states[slot]
                try:
                    assert state.observation is not None
                    frame = build_libero_concat_frame(
                        state.observation,
                        cameras=profile.cameras,
                        image_size=profile.image_size,
                        rotate_180=profile.rotate_images,
                        flip_images=False,
                    )
                    seed = stable_decision_seed(
                        base_seed,
                        task_suite,
                        task_id,
                        state.trial_id,
                        state.decisions,
                    )
                    items.append(build_policy_item(frame, task_prompt, profile, seed))
                    request_slots.append(slot)
                except Exception as error:
                    finish_infra([slot], "observation_encoding", error)
            if not request_slots:
                continue

            for slot in request_slots:
                states[slot].decisions += 1
            try:
                chunks = policy_client.predict_batch(items, profile)
            except Exception as error:
                finish_infra(request_slots, "policy_server", error)
                continue

            converted: dict[int, np.ndarray] = {}
            for slot, chunk in zip(request_slots, chunks, strict=True):
                if slot not in states:
                    continue
                try:
                    libero_actions = framewise_rot6d_to_libero(chunk)
                    libero_actions = remap_gripper_pm_one(libero_actions)
                    if libero_actions.shape != (profile.action_chunk_size, LIBERO_ENV_ACTION_DIM):
                        raise ValueError(f"converted action has unexpected shape {libero_actions.shape}")
                    converted[slot] = libero_actions
                except Exception as error:
                    finish_infra([slot], "policy_action_adapter", error)

            for action_index in range(action_horizon):
                action_slots = [slot for slot in sorted(states) if slot in converted]
                if not action_slots:
                    break
                step_slots(
                    action_slots,
                    [converted[slot][action_index] for slot in action_slots],
                    count_toward_budget=True,
                )

        if states:
            raise RuntimeError("Internal error: wave ended with active episodes.")

    return sorted(results, key=lambda record: int(record["trial_id"]))


def libero_env_kwargs(
    *,
    bddl_file_name: str,
    image_size: int,
    render_gpu_device_id: int,
    control_frequency: int,
    control_mode: str,
) -> dict[str, Any]:
    """Build explicit simulator kwargs without importing LIBERO or robosuite."""
    if control_mode != "OSC_POSE":
        raise ValueError("control_mode must explicitly be OSC_POSE.")
    if isinstance(render_gpu_device_id, bool) or not isinstance(render_gpu_device_id, int) or render_gpu_device_id < 0:
        raise ValueError(f"render_gpu_device_id must be a non-negative integer, got {render_gpu_device_id!r}.")
    return {
        "bddl_file_name": bddl_file_name,
        "camera_heights": image_size,
        "camera_widths": image_size,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "render_gpu_device_id": render_gpu_device_id,
        "controller": control_mode,
        "control_freq": control_frequency,
    }


class _LiberoEnvFactory:
    """Picklable, lazy-importing factory for spawned LIBERO workers."""

    def __init__(
        self,
        *,
        bddl_file_name: str,
        image_size: int,
        render_gpu_device_id: int,
        control_mode: str,
        control_frequency: int,
        mujoco_gl: str,
    ) -> None:
        self.bddl_file_name = bddl_file_name
        self.image_size = image_size
        self.render_gpu_device_id = render_gpu_device_id
        self.control_mode = control_mode
        self.control_frequency = control_frequency
        self.mujoco_gl = mujoco_gl

    def __call__(self) -> Any:
        configure_mujoco_environment(self.mujoco_gl, self.render_gpu_device_id)

        from libero.libero.envs import OffScreenRenderEnv

        kwargs = libero_env_kwargs(
            bddl_file_name=self.bddl_file_name,
            image_size=self.image_size,
            render_gpu_device_id=self.render_gpu_device_id,
            control_frequency=self.control_frequency,
            control_mode=self.control_mode,
        )
        return OffScreenRenderEnv(**kwargs)


def create_libero_vector_env(
    *,
    bddl_file_name: str,
    num_envs: int,
    profile: LiberoCheckpointProfile,
    render_gpu_device_id: int,
    mujoco_gl: str,
) -> Any:
    """Create one SubprocVectorEnv for both serial and parallel evaluation."""
    import multiprocessing as multiprocessing_module

    from libero.libero.envs.venv import SubprocVectorEnv

    multiprocessing_module.set_start_method("spawn", force=True)
    factories = [
        _LiberoEnvFactory(
            bddl_file_name=bddl_file_name,
            image_size=profile.image_size,
            render_gpu_device_id=render_gpu_device_id,
            control_mode=profile.control_mode,
            control_frequency=profile.control_frequency,
            mujoco_gl=mujoco_gl,
        )
        for _ in range(num_envs)
    ]
    return SubprocVectorEnv(factories)


def configure_mujoco_environment(mujoco_gl: str, render_gpu_device_id: int) -> None:
    if mujoco_gl not in {"egl", "osmesa", "glfw"}:
        raise ValueError(f"Unsupported mujoco_gl={mujoco_gl!r}; use egl/osmesa/glfw.")
    if isinstance(render_gpu_device_id, bool) or not isinstance(render_gpu_device_id, int) or render_gpu_device_id < 0:
        raise ValueError(f"render_gpu_device_id must be a non-negative integer, got {render_gpu_device_id!r}.")
    os.environ["MUJOCO_GL"] = mujoco_gl
    if mujoco_gl == "egl":
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(render_gpu_device_id)
        os.environ["EGL_DEVICE_ID"] = str(render_gpu_device_id)
    elif mujoco_gl == "osmesa":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"


def parse_task_ids(raw_task_ids: str, *, num_tasks: int) -> list[int]:
    if isinstance(num_tasks, bool) or not isinstance(num_tasks, int) or num_tasks <= 0:
        raise ValueError(f"num_tasks must be positive, got {num_tasks!r}.")
    if not raw_task_ids.strip():
        return list(range(num_tasks))
    try:
        task_ids = [int(value.strip()) for value in raw_task_ids.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError(f"task_ids must be comma-separated integers, got {raw_task_ids!r}.") from error
    if not task_ids:
        raise ValueError("task_ids must select at least one task.")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"task_ids must not contain duplicates, got {task_ids}.")
    invalid = [task_id for task_id in task_ids if task_id < 0 or task_id >= num_tasks]
    if invalid:
        raise ValueError(f"task_ids out of range [0, {num_tasks}): {invalid}.")
    return task_ids


def prepare_run(run_dir: str | os.PathLike[str], manifest: Mapping[str, Any]) -> RunResumeState:
    """Create a manifest or verify it exactly before resuming a run."""
    directory = artifacts.validate_run_dir(run_dir, create=True)
    manifest_path = directory / artifacts.MANIFEST_FILENAME
    if manifest_path.exists():
        existing = artifacts.read_manifest(directory)
        existing_contract = json.loads(json.dumps(existing))
        requested_contract = json.loads(json.dumps(dict(manifest)))
        for value in (existing_contract, requested_contract):
            runtime = value.get("provenance", {}).get("runtime_environment", {})
            if isinstance(runtime, dict):
                runtime["cosmos_eval_job_id"] = None
        if existing_contract != requested_contract:
            raise ValueError("Existing manifest does not match this evaluation request; refusing unsafe resume.")
    else:
        artifacts.write_manifest(directory, manifest)
    return RunResumeState(
        run_dir=directory,
        completed_episode_ids=frozenset(artifacts.completed_episode_ids(directory)),
        complete=artifacts.is_complete(directory),
    )


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    content = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.fchmod(descriptor, 0o644)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def finalize_run(run_dir: str | os.PathLike[str], *, profile_hash: str) -> dict[str, Any]:
    """Aggregate episodes, atomically write metrics, and seal the journal."""
    directory = artifacts.validate_run_dir(run_dir)
    episodes = artifacts.read_episodes(directory)
    infra_attempts = artifacts.read_infra_errors(directory)
    terminal_ids = {str(record["episode_id"]) for record in episodes}
    unresolved_infra_ids = {
        str(record["episode_id"]) for record in infra_attempts if str(record["episode_id"]) not in terminal_ids
    }
    if unresolved_infra_ids:
        raise RunnerInfrastructureError(
            f"Cannot finalize while infrastructure-failed episodes remain retryable: {sorted(unresolved_infra_ids)}"
        )
    if not episodes:
        raise ValueError("Cannot finalize an evaluation run without terminal episode outcomes.")
    metrics = aggregate_episode_metrics(episodes)
    infra_metrics = aggregate_episode_metrics(infra_attempts)
    metrics["infrastructure_errors"] = infra_metrics["infrastructure_errors"]
    summary = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "profile_hash": profile_hash,
        **metrics,
    }
    _atomic_json_write(directory / METRICS_FILENAME, summary)
    artifacts.mark_complete(
        directory,
        metadata={
            "metrics": METRICS_FILENAME,
            "profile_hash": profile_hash,
            "overall_success_rate": summary["overall"]["success_rate"],
        },
    )
    return summary


def validate_completion_marker(
    marker: Mapping[str, Any] | None,
    *,
    expected_episode_count: int,
    profile_hash: str,
) -> None:
    """Reject malformed or stale `_SUCCESS` metadata before short-circuiting."""
    if not isinstance(marker, Mapping):
        raise ValueError("Completed run must contain a JSON object in _SUCCESS.")
    completed_at = marker.get("completed_at")
    if not isinstance(completed_at, str) or not completed_at.strip():
        raise ValueError("_SUCCESS completed_at must be a non-empty string.")
    episode_count = marker.get("episode_count")
    if isinstance(episode_count, bool) or episode_count != expected_episode_count:
        raise ValueError(f"_SUCCESS episode_count must be {expected_episode_count}, got {episode_count!r}.")
    if marker.get("metrics") != METRICS_FILENAME:
        raise ValueError(f"_SUCCESS metrics must be {METRICS_FILENAME!r}, got {marker.get('metrics')!r}.")
    if marker.get("profile_hash") != profile_hash:
        raise ValueError("_SUCCESS profile_hash does not match the active policy profile.")


def _infra_record_for_spec(
    *,
    task_suite: str,
    task_id: int,
    task_prompt: str,
    trial_id: int,
    error_type: str,
    error: BaseException | str,
) -> dict[str, Any]:
    state = _ActiveEpisode(trial_id, np.empty(0), started_at=time.perf_counter())
    return _episode_record(
        state,
        task_suite=task_suite,
        task_id=task_id,
        task_prompt=task_prompt,
        success=None,
        termination="infrastructure_error",
        infra_error_type=error_type,
        error=str(error),
    )


def _manifest(
    args: argparse.Namespace, handshake: PolicyHandshake, task_ids: list[int], max_steps: int
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "server_url": args.server_url.rstrip("/"),
        "server_info": handshake.server_info,
        "policy_profile": handshake.profile.model_dump(mode="json"),
        "profile_hash": handshake.profile.profile_hash,
        "checkpoint_fingerprint": handshake.profile.checkpoint_fingerprint,
        "task_suite": args.task_suite,
        "task_ids": task_ids,
        "trials_per_task": args.trials,
        "base_seed": args.seed,
        "num_envs": args.num_envs,
        "action_horizon": args.action_horizon,
        "max_steps": max_steps,
        "warmup_steps": args.warmup_steps,
        "mujoco_gl": args.mujoco_gl,
        "render_gpu_device_id": args.render_gpu_device_id,
        "provenance": collect_runtime_provenance(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one resumable evaluation run from parsed CLI arguments."""
    if (
        args.trials <= 0
        or args.num_envs <= 0
        or args.seed < 0
        or args.seed > 2**63 - 1
        or args.max_steps < 0
        or args.warmup_steps < 0
        or args.render_gpu_device_id < 0
    ):
        raise ValueError(
            "trials and num_envs must be positive; seed must fit signed 64-bit; "
            "max_steps, warmup_steps, and render_gpu_device_id must be non-negative."
        )
    artifacts.validate_run_dir(args.output_dir)
    client = BatchPolicyClient(args.server_url, timeout_s=args.request_timeout)
    handshake = client.handshake()
    profile = handshake.profile
    if not 1 <= args.action_horizon <= profile.action_chunk_size:
        raise ValueError(f"action_horizon must be in [1, {profile.action_chunk_size}], got {args.action_horizon}.")
    configure_mujoco_environment(args.mujoco_gl, args.render_gpu_device_id)

    # Optional dependencies are imported only after CLI and server contracts pass.
    try:
        from libero.libero import benchmark, get_libero_path
    except ImportError as error:  # pragma: no cover - depends on evaluation environment
        raise RuntimeError("LIBERO is not installed in the active evaluation environment.") from error

    benchmark_types = benchmark.get_benchmark_dict()
    if args.task_suite not in benchmark_types:
        raise ValueError(f"Unknown task_suite={args.task_suite!r}; available={sorted(benchmark_types)}.")
    task_suite = benchmark_types[args.task_suite]()
    task_ids = parse_task_ids(args.task_ids, num_tasks=int(task_suite.n_tasks))
    max_steps = args.max_steps if args.max_steps > 0 else TASK_MAX_STEPS[args.task_suite]
    manifest = _manifest(args, handshake, task_ids, max_steps)
    resume = prepare_run(args.output_dir, manifest)
    expected_ids = {
        episode_id(args.task_suite, task_id, trial_id) for task_id in task_ids for trial_id in range(args.trials)
    }
    completed = set(resume.completed_episode_ids)
    unexpected_ids = completed - expected_ids
    if unexpected_ids:
        raise ValueError(f"Episode journal contains IDs outside the manifest selection: {sorted(unexpected_ids)}")
    if resume.complete:
        missing_ids = expected_ids - completed
        if missing_ids:
            raise ValueError(f"Completed run is missing expected episode IDs: {sorted(missing_ids)}")
        validate_completion_marker(
            artifacts.read_completion_marker(resume.run_dir),
            expected_episode_count=len(expected_ids),
            profile_hash=profile.profile_hash,
        )
        with (resume.run_dir / METRICS_FILENAME).open(encoding="utf-8") as stream:
            summary = json.load(stream)
        if not isinstance(summary, dict) or summary.get("profile_hash") != profile.profile_hash:
            raise ValueError("Completed run metrics do not match the manifest profile_hash.")
        return summary

    def append_terminal(record: Mapping[str, Any]) -> None:
        artifacts.append_episode(resume.run_dir, record)
        completed.add(str(record["episode_id"]))
        print(
            f"{record['episode_id']} success={record.get('success')} infra_error={record['infra_error']} "
            f"steps={record['steps']}",
            flush=True,
        )

    invocation_id = uuid.uuid4().hex
    infra_attempt_index = 0

    def append_infra(record: Mapping[str, Any]) -> None:
        nonlocal infra_attempt_index
        value = dict(record)
        value["attempt_id"] = f"{invocation_id}:{infra_attempt_index:08d}"
        infra_attempt_index += 1
        artifacts.append_infra_error(resume.run_dir, value)
        print(
            f"{value['episode_id']} infra_error={value['infra_error']} type={value.get('infra_error_type')}",
            flush=True,
        )

    for task_id in task_ids:
        pending_trials = [
            trial_id
            for trial_id in range(args.trials)
            if episode_id(args.task_suite, task_id, trial_id) not in completed
        ]
        if not pending_trials:
            continue
        try:
            task = task_suite.get_task(task_id)
            task_prompt = str(task.language)
        except Exception as error:
            for trial_id in pending_trials:
                append_infra(
                    _infra_record_for_spec(
                        task_suite=args.task_suite,
                        task_id=task_id,
                        task_prompt=f"unavailable task {task_id}",
                        trial_id=trial_id,
                        error_type="task_metadata",
                        error=error,
                    )
                )
            raise RunnerInfrastructureError(f"task_metadata: {error}") from error
        try:
            default_initial_states = task_suite.get_task_init_states(task_id)
        except Exception as error:
            for trial_id in pending_trials:
                append_infra(
                    _infra_record_for_spec(
                        task_suite=args.task_suite,
                        task_id=task_id,
                        task_prompt=task_prompt,
                        trial_id=trial_id,
                        error_type="initial_state",
                        error=error,
                    )
                )
            raise RunnerInfrastructureError(f"initial_state: {error}") from error

        episode_specs: list[EpisodeSpec] = []
        for trial_id in pending_trials:
            try:
                initial_state = np.asarray(default_initial_states[trial_id], dtype=np.float64)
                if initial_state.size == 0 or not np.isfinite(initial_state).all():
                    raise ValueError("default initial state must be non-empty and finite")
                episode_specs.append(EpisodeSpec(trial_id=trial_id, initial_state=initial_state))
            except Exception as error:
                append_infra(
                    _infra_record_for_spec(
                        task_suite=args.task_suite,
                        task_id=task_id,
                        task_prompt=task_prompt,
                        trial_id=trial_id,
                        error_type="initial_state",
                        error=error,
                    )
                )
                raise RunnerInfrastructureError(f"initial_state: {error}") from error
        if not episode_specs:
            continue

        bddl_file_name = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        task_num_envs = min(args.num_envs, len(episode_specs))
        try:
            vector_env = create_libero_vector_env(
                bddl_file_name=bddl_file_name,
                num_envs=task_num_envs,
                profile=profile,
                render_gpu_device_id=args.render_gpu_device_id,
                mujoco_gl=args.mujoco_gl,
            )
        except Exception as error:
            for spec in episode_specs:
                append_infra(
                    _infra_record_for_spec(
                        task_suite=args.task_suite,
                        task_id=task_id,
                        task_prompt=task_prompt,
                        trial_id=spec.trial_id,
                        error_type="environment_create",
                        error=error,
                    )
                )
            raise RunnerInfrastructureError(f"environment_create: {error}") from error
        try:
            run_vectorized_state_machine(
                vector_env=vector_env,
                policy_client=client,
                profile=profile,
                task_suite=args.task_suite,
                task_id=task_id,
                task_prompt=task_prompt,
                episodes=episode_specs,
                num_envs=task_num_envs,
                base_seed=args.seed,
                action_horizon=args.action_horizon,
                max_steps=max_steps,
                warmup_steps=args.warmup_steps,
                on_episode=append_terminal,
                on_infra_error=append_infra,
            )
        finally:
            try:
                vector_env.close()
            except Exception:
                pass

    missing_ids = expected_ids - completed
    if missing_ids:
        raise RuntimeError(f"Evaluation ended without records for episodes: {sorted(missing_ids)}")
    return finalize_run(resume.run_dir, profile_hash=profile.profile_hash)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict Cosmos Edge LIBERO closed-loop evaluation")
    parser.add_argument("--server-url", required=True, help="Action policy server base URL")
    parser.add_argument("--task-suite", default="libero_spatial", choices=sorted(TASK_MAX_STEPS))
    parser.add_argument("--task-ids", default="", help="Comma-separated IDs; empty selects the full suite")
    parser.add_argument("--trials", type=int, default=10, help="Trials per selected task")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel vector environment slots")
    parser.add_argument("--action-horizon", type=int, default=EDGE_ACTION_CHUNK_SIZE)
    parser.add_argument("--seed", type=int, default=0, help="Base seed for deterministic decision seeds")
    parser.add_argument("--max-steps", type=int, default=0, help="0 selects the suite's canonical limit")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--mujoco-gl", default="egl", choices=["egl", "osmesa", "glfw"])
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--output-dir",
        required=True,
        help=f"Absolute per-run directory below canonical root {artifacts.OUTPUT_ROOT}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
