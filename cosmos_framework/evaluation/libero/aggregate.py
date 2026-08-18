# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validate and aggregate sealed Cosmos Edge LIBERO evaluation runs."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from pydantic import ValidationError

from cosmos_framework.evaluation.libero import artifacts
from cosmos_framework.evaluation.libero.action import GRIPPER_ADAPTER_CONTRACT
from cosmos_framework.evaluation.libero.checkpoint_profile import compute_profile_hash
from cosmos_framework.evaluation.libero.metrics import aggregate_episode_metrics, aggregate_gripper_adapter_metrics
from cosmos_framework.evaluation.libero.runner import (
    METRICS_FILENAME,
    PROTOCOL_VERSION,
    TASK_MAX_STEPS,
    RunnerProtocolError,
    episode_id,
    validate_completion_marker,
    validate_edge_policy_profile,
)
from cosmos_framework.evaluation.libero.schema import LiberoCheckpointProfile
from cosmos_framework.evaluation.libero.seeding import SAMPLING_SEED_CONTRACT, stable_episode_seed

REPORT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 3
METRICS_SCHEMA_VERSION = 2
REPORT_KIND = "libero-checkpoint-matrix"
PRIMARY_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
# Keep this light-weight mirror synchronized with
# inference.common.args.ConfigFileType without importing the GPU inference stack.
SUPPORTED_CONFIG_FILE_TYPES = frozenset({"module", "yaml", "json"})

_REQUIRED_SOURCE_FILENAMES = (
    artifacts.MANIFEST_FILENAME,
    artifacts.EPISODES_FILENAME,
    METRICS_FILENAME,
    artifacts.COMPLETION_FILENAME,
)
_SOURCE_FILENAMES = (*_REQUIRED_SOURCE_FILENAMES, artifacts.INFRA_ERRORS_FILENAME)

_SUITE_VARIANT_PATHS = (
    ("task_suite",),
    ("max_steps",),
    ("provenance", "runtime_environment", "cosmos_eval_job_id"),
)
_CHECKPOINT_VARIANT_PATHS = (
    ("checkpoint_fingerprint",),
    ("profile_hash",),
    ("policy_profile", "checkpoint_fingerprint"),
    ("policy_profile", "checkpoint_path"),
    ("policy_profile", "checkpoint_role"),
    ("policy_profile", "profile_hash"),
    ("policy_profile", "profile_sources"),
    ("policy_profile", "weights_variant"),
    ("policy_profile", "zero_shot"),
    ("server_info", "checkpoint"),
    ("server_info", "checkpoint_fingerprint"),
    ("server_info", "config_file"),
    ("server_info", "policy_profile", "checkpoint_fingerprint"),
    ("server_info", "policy_profile", "checkpoint_path"),
    ("server_info", "policy_profile", "checkpoint_role"),
    ("server_info", "policy_profile", "profile_hash"),
    ("server_info", "policy_profile", "profile_sources"),
    ("server_info", "policy_profile", "weights_variant"),
    ("server_info", "policy_profile", "zero_shot"),
    ("server_info", "weights_variant"),
)


class AggregationError(ValueError):
    """Raised when sealed evaluation inputs cannot form one valid report."""


@dataclass(frozen=True)
class SealedSuiteRun:
    """Validated immutable inputs for one checkpoint and one task suite."""

    checkpoint_id: str
    suite: str
    run_dir: Path
    manifest: dict[str, Any]
    completion: dict[str, Any]
    metrics: dict[str, Any]
    episodes: tuple[dict[str, Any], ...]
    infra_attempts: tuple[dict[str, Any], ...]
    source_sha256: dict[str, str | None]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AggregationError(f"{name} must be a JSON object")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AggregationError(f"{name} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AggregationError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _sha256(value: Any, name: str, *, length: int = 64) -> str:
    result = _nonempty_string(value, name)
    if len(result) != length or any(character not in "0123456789abcdef" for character in result):
        raise AggregationError(f"{name} must be a lowercase {length}-character hexadecimal digest")
    return result


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise AggregationError(f"JSON artifact contains non-finite constant {value!r}")


def _read_regular_source_bytes(path: Path, name: str) -> bytes:
    """Read one regular file without following a final-component symlink."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise AggregationError(f"Sealed run requires a regular {name} file: {path}")
        stream = os.fdopen(descriptor, "rb")
        descriptor = -1
        with stream:
            return stream.read()
    except AggregationError:
        raise
    except OSError as error:
        raise AggregationError(f"Cannot read regular sealed {name} artifact: {path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _parse_json_object(content: bytes, path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AggregationError(f"Cannot parse valid JSON from {name}: {path}") from error
    return dict(_mapping(value, name))


def _parse_sealed_jsonl(
    content: bytes,
    path: Path,
    name: str,
    *,
    identity_field: str,
) -> tuple[dict[str, Any], ...]:
    """Parse an immutable journal snapshot without opening its append lock."""
    if not content:
        return ()
    if not content.endswith(b"\n"):
        raise AggregationError(f"Sealed {name} has an incomplete trailing record: {path}")

    records: list[dict[str, Any]] = []
    identities: set[str] = set()
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        if not raw_line.strip():
            raise AggregationError(f"Blank JSONL record in {path}:{line_number}")
        try:
            value = json.loads(raw_line.decode("utf-8"), parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AggregationError(f"Malformed JSONL record in {path}:{line_number}") from error
        record = dict(_mapping(value, f"{name} record {line_number}"))
        identity = _nonempty_string(record.get(identity_field), f"{name} record {line_number} {identity_field}")
        if identity in identities:
            raise AggregationError(f"Duplicate {identity_field}={identity!r} in sealed {name}")
        identities.add(identity)
        records.append(record)
    return tuple(records)


def _verify_source_snapshot(paths: Mapping[str, Path], snapshot: Mapping[str, bytes | None]) -> None:
    for name, path in paths.items():
        expected = snapshot[name]
        if expected is None:
            if path.is_symlink() or path.exists():
                raise AggregationError(f"Sealed source {name} changed while validating: {path}")
            continue
        try:
            actual = _read_regular_source_bytes(path, name)
        except AggregationError as error:
            raise AggregationError(f"Sealed source {name} changed while validating: {path}") from error
        if actual != expected:
            raise AggregationError(f"Sealed source {name} changed while validating: {path}")


def _validate_provenance(manifest: Mapping[str, Any]) -> None:
    provenance = _mapping(manifest.get("provenance"), "manifest.provenance")
    git = _mapping(provenance.get("git"), "manifest.provenance.git")
    _sha256(git.get("head"), "manifest.provenance.git.head", length=40)
    if git.get("dirty") is not False:
        raise AggregationError("Formal aggregation requires provenance.git.dirty=false")

    uv_lock = _mapping(provenance.get("uv_lock"), "manifest.provenance.uv_lock")
    _nonempty_string(uv_lock.get("path"), "manifest.provenance.uv_lock.path")
    _sha256(uv_lock.get("sha256"), "manifest.provenance.uv_lock.sha256")

    python = _mapping(provenance.get("python"), "manifest.provenance.python")
    for field in ("version", "implementation", "executable", "platform"):
        _nonempty_string(python.get(field), f"manifest.provenance.python.{field}")
    package_versions = _mapping(provenance.get("package_versions"), "manifest.provenance.package_versions")
    for package in ("cosmos-framework", "libero", "robosuite", "mujoco", "numpy", "torch"):
        _nonempty_string(package_versions.get(package), f"manifest.provenance.package_versions.{package}")

    runtime = _mapping(provenance.get("runtime_environment"), "manifest.provenance.runtime_environment")
    _nonempty_string(runtime.get("cosmos_eval_image"), "manifest.provenance.runtime_environment.cosmos_eval_image")
    _nonempty_string(runtime.get("cosmos_eval_job_id"), "manifest.provenance.runtime_environment.cosmos_eval_job_id")

    gpus = provenance.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        raise AggregationError("manifest.provenance.gpus must contain at least one GPU")
    for index, raw_gpu in enumerate(gpus):
        gpu = _mapping(raw_gpu, f"manifest.provenance.gpus[{index}]")
        _nonempty_string(gpu.get("name"), f"manifest.provenance.gpus[{index}].name")
        _nonempty_string(gpu.get("driver_version"), f"manifest.provenance.gpus[{index}].driver_version")


def _validate_server_info(
    server_info: Mapping[str, Any],
    profile: LiberoCheckpointProfile,
    raw_profile: Mapping[str, Any],
) -> None:
    expected_fields = {
        "protocol_version": PROTOCOL_VERSION,
        "sampling_seed_contract": SAMPLING_SEED_CONTRACT,
        "gripper_adapter_contract": GRIPPER_ADAPTER_CONTRACT,
        "checkpoint": profile.checkpoint_path,
        "checkpoint_fingerprint": profile.checkpoint_fingerprint,
        "weights_variant": profile.weights_variant.value,
        "action_chunk_size": profile.action_chunk_size,
        "raw_action_dim": profile.effective_action_dim,
        "fps": profile.conditioning_fps,
        "format_prompt_as_json": profile.format_prompt_as_json,
        "action_normalization": profile.action_normalization,
        "action_stats_path": profile.action_stats_path,
        "action_stats_sha256": profile.action_stats_sha256,
    }
    for field, expected in expected_fields.items():
        if server_info.get(field) != expected:
            raise AggregationError(f"manifest.server_info.{field} does not match the validated policy profile")
    if server_info.get("policy_profile") != dict(raw_profile):
        raise AggregationError("manifest.server_info.policy_profile does not match manifest.policy_profile")

    _nonempty_string(server_info.get("config_file"), "manifest.server_info.config_file")
    if server_info.get("config_file_type") not in SUPPORTED_CONFIG_FILE_TYPES:
        raise AggregationError(
            f"manifest.server_info.config_file_type must be one of {sorted(SUPPORTED_CONFIG_FILE_TYPES)}"
        )
    max_action_dim = _nonnegative_int(server_info.get("max_action_dim"), "manifest.server_info.max_action_dim")
    if max_action_dim < profile.effective_action_dim:
        raise AggregationError("manifest.server_info.max_action_dim must cover the effective policy action dimension")

    num_steps = _nonnegative_int(server_info.get("num_steps"), "manifest.server_info.num_steps")
    if num_steps == 0:
        raise AggregationError("manifest.server_info.num_steps must be positive")
    guidance = server_info.get("guidance")
    if (
        isinstance(guidance, bool)
        or not isinstance(guidance, (int, float))
        or not math.isfinite(float(guidance))
        or guidance < 0
    ):
        raise AggregationError("manifest.server_info.guidance must be a finite non-negative number")
    server_seed = _nonnegative_int(server_info.get("seed"), "manifest.server_info.seed")
    if server_seed > 2**63 - 1:
        raise AggregationError("manifest.server_info.seed must fit signed 64-bit")
    if not isinstance(server_info.get("run_name"), str):
        raise AggregationError("manifest.server_info.run_name must be a string")


def _validate_manifest_contract(manifest: Mapping[str, Any]) -> LiberoCheckpointProfile:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise AggregationError(f"Manifest schema_version must be {MANIFEST_SCHEMA_VERSION}")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise AggregationError(f"Manifest protocol_version must be {PROTOCOL_VERSION!r}")
    if manifest.get("sampling_seed_contract") != SAMPLING_SEED_CONTRACT:
        raise AggregationError("Manifest sampling_seed_contract is incompatible")
    if manifest.get("gripper_adapter_contract") != GRIPPER_ADAPTER_CONTRACT:
        raise AggregationError("Manifest gripper_adapter_contract is incompatible")

    raw_profile = _mapping(manifest.get("policy_profile"), "manifest.policy_profile")
    try:
        profile = LiberoCheckpointProfile.model_validate(raw_profile)
    except ValidationError as error:
        raise AggregationError(f"Invalid manifest.policy_profile: {error}") from error
    expected_profile_hash = compute_profile_hash(raw_profile)
    if profile.profile_hash != expected_profile_hash:
        raise AggregationError(
            f"Semantic profile_hash mismatch: recorded={profile.profile_hash}, expected={expected_profile_hash}"
        )
    try:
        validate_edge_policy_profile(profile)
    except RunnerProtocolError as error:
        raise AggregationError(f"Policy profile is not the canonical Edge LIBERO contract: {error}") from error
    if manifest.get("profile_hash") != profile.profile_hash:
        raise AggregationError("Manifest profile_hash does not match policy_profile")
    if manifest.get("checkpoint_fingerprint") != profile.checkpoint_fingerprint:
        raise AggregationError("Manifest checkpoint_fingerprint does not match policy_profile")

    server_info = _mapping(manifest.get("server_info"), "manifest.server_info")
    _validate_server_info(server_info, profile, raw_profile)
    _validate_provenance(manifest)
    return profile


def _validate_episode(
    record: Mapping[str, Any],
    *,
    suite: str,
    task_id: int,
    trial_id: int,
    base_seed: int,
    action_horizon: int,
    max_steps: int,
    action_chunk_size: int,
) -> None:
    identity = episode_id(suite, task_id, trial_id)
    if record.get("episode_id") != identity:
        raise AggregationError(f"Episode identity mismatch for {identity}")
    if record.get("task_suite") != suite or record.get("task_id") != task_id or record.get("trial_id") != trial_id:
        raise AggregationError(f"Episode coordinates do not match {identity}")
    if record.get("episode_seed") != stable_episode_seed(base_seed, suite, task_id, trial_id):
        raise AggregationError(f"Episode seed does not match the manifest for {identity}")
    if record.get("infra_error") is not False or not isinstance(record.get("success"), bool):
        raise AggregationError(f"Episode {identity} is not a terminal non-infrastructure outcome")
    _nonempty_string(record.get("task_description"), f"episode {identity} task_description")

    steps = _nonnegative_int(record.get("steps"), f"episode {identity} steps")
    decisions = _nonnegative_int(record.get("decisions"), f"episode {identity} decisions")
    if steps > max_steps:
        raise AggregationError(f"Episode {identity} exceeds max_steps={max_steps}")
    expected_decisions = math.ceil(steps / action_horizon)
    if decisions != expected_decisions:
        raise AggregationError(
            f"Episode {identity} decisions={decisions} does not equal ceil(steps/action_horizon)={expected_decisions}"
        )

    success = record["success"]
    termination = record.get("termination")
    if success and termination != "success_signal":
        raise AggregationError(f"Successful episode {identity} must terminate with success_signal")
    if not success and termination not in {"done_without_success", "max_steps"}:
        raise AggregationError(f"Failed episode {identity} has invalid termination={termination!r}")
    if termination == "max_steps" and steps != max_steps:
        raise AggregationError(f"Episode {identity} terminated at max_steps with steps={steps}")

    # Reuse the strict episode-level gripper validator before checking its
    # relationship to the number of generated policy chunks.
    aggregate_gripper_adapter_metrics([record])
    telemetry = _mapping(record.get("gripper_adapter_telemetry"), f"episode {identity} gripper telemetry")
    generated = _nonnegative_int(telemetry.get("generated_value_count"), f"episode {identity} generated_value_count")
    expected_generated = decisions * action_chunk_size
    if generated != expected_generated:
        raise AggregationError(
            f"Episode {identity} generated_value_count={generated} does not equal "
            f"decisions*action_chunk_size={expected_generated}"
        )


def load_sealed_suite_run(
    checkpoint_id: str,
    suite: str,
    run_dir: str | os.PathLike[str],
    *,
    expected_task_ids: Sequence[int],
    trials_per_task: int,
    require_canonical_max_steps: bool = True,
    require_zero_infra_attempts: bool = False,
) -> SealedSuiteRun:
    """Load one sealed run and re-derive every published suite metric."""
    _nonempty_string(checkpoint_id, "checkpoint_id")
    _nonempty_string(suite, "suite")
    if isinstance(trials_per_task, bool) or not isinstance(trials_per_task, int) or trials_per_task <= 0:
        raise AggregationError("trials_per_task must be a positive integer")
    task_ids = tuple(_nonnegative_int(value, "expected_task_id") for value in expected_task_ids)
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise AggregationError("expected_task_ids must be a non-empty sequence of unique integers")

    directory = artifacts.validate_run_dir(run_dir)
    source_paths = {name: directory / name for name in _SOURCE_FILENAMES}
    source_snapshot: dict[str, bytes | None] = {
        name: _read_regular_source_bytes(source_paths[name], name) for name in _REQUIRED_SOURCE_FILENAMES
    }
    infra_path = source_paths[artifacts.INFRA_ERRORS_FILENAME]
    source_snapshot[artifacts.INFRA_ERRORS_FILENAME] = (
        _read_regular_source_bytes(infra_path, artifacts.INFRA_ERRORS_FILENAME)
        if infra_path.is_symlink() or infra_path.exists()
        else None
    )

    completion_content = source_snapshot[artifacts.COMPLETION_FILENAME]
    manifest_content = source_snapshot[artifacts.MANIFEST_FILENAME]
    episodes_content = source_snapshot[artifacts.EPISODES_FILENAME]
    metrics_content = source_snapshot[METRICS_FILENAME]
    assert completion_content is not None
    assert manifest_content is not None
    assert episodes_content is not None
    assert metrics_content is not None
    completion = _parse_json_object(
        completion_content,
        source_paths[artifacts.COMPLETION_FILENAME],
        artifacts.COMPLETION_FILENAME,
    )
    manifest = _parse_json_object(
        manifest_content,
        source_paths[artifacts.MANIFEST_FILENAME],
        artifacts.MANIFEST_FILENAME,
    )
    profile = _validate_manifest_contract(manifest)

    if manifest.get("task_suite") != suite:
        raise AggregationError(f"Manifest task_suite does not match expected suite {suite!r}")
    manifest_task_ids = manifest.get("task_ids")
    if not isinstance(manifest_task_ids, list) or tuple(manifest_task_ids) != task_ids:
        raise AggregationError(f"Manifest task_ids must be exactly {list(task_ids)}")
    if manifest.get("trials_per_task") != trials_per_task:
        raise AggregationError(f"Manifest trials_per_task must be {trials_per_task}")
    max_steps = _nonnegative_int(manifest.get("max_steps"), "manifest.max_steps")
    if max_steps == 0:
        raise AggregationError("manifest.max_steps must contain the resolved positive limit")
    if require_canonical_max_steps:
        expected_max_steps = TASK_MAX_STEPS.get(suite)
        if expected_max_steps is None or max_steps != expected_max_steps:
            raise AggregationError(
                f"Manifest max_steps for {suite!r} must be canonical value {expected_max_steps}, got {max_steps}"
            )
    base_seed = _nonnegative_int(manifest.get("base_seed"), "manifest.base_seed")
    if base_seed > 2**63 - 1:
        raise AggregationError("manifest.base_seed must fit signed 64-bit")
    action_horizon = _nonnegative_int(manifest.get("action_horizon"), "manifest.action_horizon")
    if action_horizon == 0 or action_horizon > profile.action_chunk_size:
        raise AggregationError("manifest.action_horizon is incompatible with the policy action chunk")

    expected_count = len(task_ids) * trials_per_task
    try:
        validate_completion_marker(completion, expected_episode_count=expected_count, profile_hash=profile.profile_hash)
    except ValueError as error:
        raise AggregationError(f"Invalid completion marker for {directory}: {error}") from error

    # The producing worker may own a mode-0600 `.episodes.lock`. Once `_SUCCESS`
    # exists, append APIs refuse mutation, so aggregation reads the sealed
    # journals directly and never opens that append lock in `a+b` mode.
    episodes = _parse_sealed_jsonl(
        episodes_content,
        source_paths[artifacts.EPISODES_FILENAME],
        artifacts.EPISODES_FILENAME,
        identity_field="episode_id",
    )
    infra_content = source_snapshot[artifacts.INFRA_ERRORS_FILENAME]
    infra_attempts = (
        _parse_sealed_jsonl(
            infra_content,
            infra_path,
            artifacts.INFRA_ERRORS_FILENAME,
            identity_field="attempt_id",
        )
        if infra_content is not None
        else ()
    )
    expected_coordinates = {
        episode_id(suite, task_id, trial_id): (task_id, trial_id)
        for task_id in task_ids
        for trial_id in range(trials_per_task)
    }
    for index, record in enumerate(episodes):
        if record.get("infra_error") is not False or not isinstance(record.get("success"), bool):
            raise AggregationError(f"Episode record {index} is not a terminal non-infrastructure outcome")
    for index, record in enumerate(infra_attempts):
        identity = _nonempty_string(record.get("episode_id"), f"infrastructure record {index} episode_id")
        if record.get("infra_error") is not True or "success" in record:
            raise AggregationError(f"Infrastructure record {index} must have infra_error=true and no success field")
        task_suite = _nonempty_string(record.get("task_suite"), f"infrastructure record {index} task_suite")
        task_id = _nonnegative_int(record.get("task_id"), f"infrastructure record {index} task_id")
        trial_id = _nonnegative_int(record.get("trial_id"), f"infrastructure record {index} trial_id")
        if (
            task_suite != suite
            or identity != episode_id(suite, task_id, trial_id)
            or expected_coordinates.get(identity) != (task_id, trial_id)
        ):
            raise AggregationError(f"Infrastructure record {index} does not belong to the manifest episode grid")
        if record.get("episode_seed") != stable_episode_seed(
            base_seed,
            suite,
            task_id,
            trial_id,
        ):
            raise AggregationError(f"Infrastructure record {index} seed does not match the manifest episode grid")
    if require_zero_infra_attempts and infra_attempts:
        raise AggregationError(
            f"Formal run contains {len(infra_attempts)} historical infrastructure attempts: {directory}"
        )

    by_id = {str(record.get("episode_id")): record for record in episodes}
    if set(by_id) != set(expected_coordinates) or len(episodes) != expected_count:
        missing = sorted(set(expected_coordinates) - set(by_id))
        extra = sorted(set(by_id) - set(expected_coordinates))
        raise AggregationError(
            f"Episode journal does not match the manifest selection: expected={expected_count}, actual={len(episodes)}, "
            f"missing={missing}, extra={extra}"
        )

    descriptions: dict[int, str] = {}
    for identity, (task_id, trial_id) in expected_coordinates.items():
        record = by_id[identity]
        _validate_episode(
            record,
            suite=suite,
            task_id=task_id,
            trial_id=trial_id,
            base_seed=base_seed,
            action_horizon=action_horizon,
            max_steps=max_steps,
            action_chunk_size=profile.action_chunk_size,
        )
        description = str(record["task_description"])
        previous = descriptions.setdefault(task_id, description)
        if previous != description:
            raise AggregationError(f"Task description is inconsistent across trials for {suite} task {task_id}")

    metrics = _parse_json_object(metrics_content, source_paths[METRICS_FILENAME], METRICS_FILENAME)
    terminal_metrics = aggregate_episode_metrics(episodes)
    infra_metrics = aggregate_episode_metrics(infra_attempts)
    terminal_metrics["infrastructure_errors"] = infra_metrics["infrastructure_errors"]
    expected_metrics = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "profile_hash": profile.profile_hash,
        "gripper_adapter": aggregate_gripper_adapter_metrics(episodes),
        **terminal_metrics,
    }
    if metrics != expected_metrics:
        raise AggregationError(f"Published metrics do not exactly match re-derived episode metrics: {directory}")

    marker_rate = completion.get("overall_success_rate")
    if (
        isinstance(marker_rate, bool)
        or not isinstance(marker_rate, (int, float))
        or not math.isfinite(float(marker_rate))
    ):
        raise AggregationError("_SUCCESS overall_success_rate must be a finite number")
    if float(marker_rate) != metrics["overall"]["success_rate"]:
        raise AggregationError("_SUCCESS overall_success_rate does not match metrics.json")
    source_sha256 = {
        name: _sha256_bytes(content) if content is not None else None for name, content in source_snapshot.items()
    }
    _verify_source_snapshot(source_paths, source_snapshot)
    return SealedSuiteRun(
        checkpoint_id=checkpoint_id,
        suite=suite,
        run_dir=directory,
        manifest=manifest,
        completion=dict(completion),
        metrics=metrics,
        episodes=episodes,
        infra_attempts=infra_attempts,
        source_sha256=source_sha256,
    )


def _delete_path(value: dict[str, Any], path: tuple[str, ...]) -> None:
    current: dict[str, Any] = value
    for component in path[:-1]:
        child = current.get(component)
        if not isinstance(child, dict):
            raise AggregationError(f"Manifest is missing contract path {'.'.join(path)}")
        current = child
    if path[-1] not in current:
        raise AggregationError(f"Manifest is missing contract path {'.'.join(path)}")
    del current[path[-1]]


def _normalized_manifest(manifest: Mapping[str, Any], paths: Sequence[tuple[str, ...]]) -> dict[str, Any]:
    result = copy.deepcopy(dict(manifest))
    for path in paths:
        _delete_path(result, path)
    return result


def _validate_matrix_contract(runs_by_checkpoint: Mapping[str, Sequence[SealedSuiteRun]]) -> None:
    if not runs_by_checkpoint:
        raise AggregationError("At least one checkpoint is required")
    profile_hashes: set[str] = set()
    fingerprints: set[str] = set()
    descriptions: dict[tuple[str, int], str] | None = None

    for checkpoint_id, runs in runs_by_checkpoint.items():
        if not runs:
            raise AggregationError(f"Checkpoint {checkpoint_id!r} has no suite runs")
        baseline = _normalized_manifest(runs[0].manifest, _SUITE_VARIANT_PATHS)
        suites: set[str] = set()
        current_descriptions: dict[tuple[str, int], str] = {}
        for run in runs:
            if run.checkpoint_id != checkpoint_id:
                raise AggregationError("Internal checkpoint label mismatch")
            if run.suite in suites:
                raise AggregationError(f"Duplicate suite {run.suite!r} for checkpoint {checkpoint_id!r}")
            suites.add(run.suite)
            if _normalized_manifest(run.manifest, _SUITE_VARIANT_PATHS) != baseline:
                raise AggregationError(f"Suite manifests disagree for checkpoint {checkpoint_id!r}")
            for record in run.episodes:
                key = (run.suite, int(record["task_id"]))
                description = str(record["task_description"])
                previous = current_descriptions.setdefault(key, description)
                if previous != description:
                    raise AggregationError(f"Task description mismatch for {key}")

        profile_hash = str(runs[0].manifest["profile_hash"])
        fingerprint = str(runs[0].manifest["checkpoint_fingerprint"])
        if profile_hash in profile_hashes or fingerprint in fingerprints:
            raise AggregationError("Checkpoint groups must have distinct profile hashes and fingerprints")
        profile_hashes.add(profile_hash)
        fingerprints.add(fingerprint)
        if descriptions is None:
            descriptions = current_descriptions
        elif descriptions != current_descriptions:
            raise AggregationError("Task descriptions differ across checkpoint groups")

    checkpoint_ids = sorted(runs_by_checkpoint)
    baseline_runs = {run.suite: run for run in runs_by_checkpoint[checkpoint_ids[0]]}
    comparison_paths = (*_CHECKPOINT_VARIANT_PATHS, ("provenance", "runtime_environment", "cosmos_eval_job_id"))
    for checkpoint_id in checkpoint_ids[1:]:
        candidate_runs = {run.suite: run for run in runs_by_checkpoint[checkpoint_id]}
        if set(candidate_runs) != set(baseline_runs):
            raise AggregationError("Checkpoint groups do not contain the same suite set")
        for suite, baseline_run in baseline_runs.items():
            if _normalized_manifest(baseline_run.manifest, comparison_paths) != _normalized_manifest(
                candidate_runs[suite].manifest, comparison_paths
            ):
                raise AggregationError(
                    f"Evaluation/provenance contract differs for checkpoint {checkpoint_id!r}, {suite}"
                )


def _field_summary(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [int(record[field]) for record in records]
    successful = [int(record[field]) for record in records if record["success"] is True]
    failed = [int(record[field]) for record in records if record["success"] is False]
    return {
        "total": sum(values),
        "mean": fmean(values),
        "min": min(values),
        "max": max(values),
        "success_mean": fmean(successful) if successful else None,
        "failure_mean": fmean(failed) if failed else None,
    }


def _efficiency_fields(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise AggregationError("Cannot summarize rollout efficiency without terminal episodes")
    return {"steps": _field_summary(records, "steps"), "decisions": _field_summary(records, "decisions")}


def aggregate_checkpoint_runs(checkpoint_id: str, runs: Sequence[SealedSuiteRun]) -> dict[str, Any]:
    """Aggregate all suites for one checkpoint without crossing policy identity."""
    episodes = tuple(record for run in runs for record in run.episodes)
    infra_attempts = tuple(record for run in runs for record in run.infra_attempts)
    rates = aggregate_episode_metrics(episodes)
    rates["infrastructure_errors"] = aggregate_episode_metrics(infra_attempts)["infrastructure_errors"]

    for row in rates["tasks"]:
        selected = [
            record
            for record in episodes
            if record["task_suite"] == row["task_suite"] and record["task_id"] == row["task_id"]
        ]
        row.update(_efficiency_fields(selected))
    for row in rates["suites"]:
        selected = [record for record in episodes if record["task_suite"] == row["task_suite"]]
        row.update(_efficiency_fields(selected))
        row["gripper_adapter"] = aggregate_gripper_adapter_metrics(selected)
    rates["overall"].update(_efficiency_fields(episodes))
    rates["overall"]["aggregation"] = "micro_episode_weighted"

    first_manifest = runs[0].manifest
    profile = _mapping(first_manifest["policy_profile"], "policy_profile")
    source_runs = []
    for run in runs:
        runtime = _mapping(run.manifest["provenance"]["runtime_environment"], "runtime_environment")
        source_runs.append(
            {
                "task_suite": run.suite,
                "run_dir": str(run.run_dir),
                "job_id": runtime["cosmos_eval_job_id"],
                "completed_at": run.completion["completed_at"],
                "episode_count": len(run.episodes),
                "source_sha256": run.source_sha256,
            }
        )
    return {
        "checkpoint_id": checkpoint_id,
        "checkpoint_fingerprint": first_manifest["checkpoint_fingerprint"],
        "profile_hash": first_manifest["profile_hash"],
        "checkpoint_role": profile["checkpoint_role"],
        "zero_shot": profile["zero_shot"],
        "weights_variant": profile["weights_variant"],
        "policy_profile": dict(profile),
        "episode_count": len(episodes),
        "source_runs": source_runs,
        "gripper_adapter": aggregate_gripper_adapter_metrics(episodes),
        **rates,
    }


def aggregate_matrix(
    checkpoint_roots: Mapping[str, str | os.PathLike[str]],
    *,
    suites: Sequence[str] = PRIMARY_SUITES,
    expected_task_ids: Sequence[int] = tuple(range(10)),
    trials_per_task: int = 50,
    require_canonical_max_steps: bool = True,
    require_zero_infra_attempts: bool = False,
) -> dict[str, Any]:
    """Validate and aggregate a checkpoint matrix into independent results."""
    suite_names = tuple(_nonempty_string(suite, "suite") for suite in suites)
    if not suite_names or len(set(suite_names)) != len(suite_names):
        raise AggregationError("suites must be a non-empty sequence of unique names")
    task_ids = tuple(expected_task_ids)
    roots: dict[str, Path] = {}
    for checkpoint_id, raw_root in checkpoint_roots.items():
        label = _nonempty_string(checkpoint_id, "checkpoint_id")
        if label in roots:
            raise AggregationError(f"Duplicate checkpoint_id={label!r}")
        roots[label] = artifacts.validate_run_dir(raw_root)
    if not roots:
        raise AggregationError("At least one checkpoint root is required")

    runs_by_checkpoint: dict[str, tuple[SealedSuiteRun, ...]] = {}
    seen_run_dirs: set[Path] = set()
    for checkpoint_id in sorted(roots):
        runs = tuple(
            load_sealed_suite_run(
                checkpoint_id,
                suite,
                roots[checkpoint_id] / suite,
                expected_task_ids=task_ids,
                trials_per_task=trials_per_task,
                require_canonical_max_steps=require_canonical_max_steps,
                require_zero_infra_attempts=require_zero_infra_attempts,
            )
            for suite in suite_names
        )
        for run in runs:
            if run.run_dir in seen_run_dirs:
                raise AggregationError(f"Run directory is reused across checkpoint groups: {run.run_dir}")
            seen_run_dirs.add(run.run_dir)
        runs_by_checkpoint[checkpoint_id] = runs

    _validate_matrix_contract(runs_by_checkpoint)
    checkpoint_results = [
        aggregate_checkpoint_runs(checkpoint_id, runs_by_checkpoint[checkpoint_id])
        for checkpoint_id in sorted(runs_by_checkpoint)
    ]
    baseline_manifest = next(iter(runs_by_checkpoint.values()))[0].manifest
    common_paths = tuple(dict.fromkeys((*_SUITE_VARIANT_PATHS, *_CHECKPOINT_VARIANT_PATHS)))
    expected_per_checkpoint = len(suite_names) * len(task_ids) * trials_per_task
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": REPORT_KIND,
        "protocol_version": PROTOCOL_VERSION,
        "input_contract": {
            "suites": list(suite_names),
            "task_ids": list(task_ids),
            "trials_per_task": trials_per_task,
            "episodes_per_checkpoint": expected_per_checkpoint,
            "require_canonical_max_steps": require_canonical_max_steps,
            "require_zero_infra_attempts": require_zero_infra_attempts,
        },
        "source_run_count": len(suite_names) * len(checkpoint_results),
        "total_episode_count": sum(result["episode_count"] for result in checkpoint_results),
        "common_contract": _normalized_manifest(baseline_manifest, common_paths),
        "checkpoints": checkpoint_results,
    }


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_report_sources(report: Mapping[str, Any]) -> None:
    """Revalidate formal report inputs immediately before accepting publication."""
    if report.get("report_kind") != REPORT_KIND:
        return
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise AggregationError("Formal report must contain checkpoint results")
    for checkpoint_index, raw_checkpoint in enumerate(checkpoints):
        checkpoint = _mapping(raw_checkpoint, f"report.checkpoints[{checkpoint_index}]")
        source_runs = checkpoint.get("source_runs")
        if not isinstance(source_runs, list) or not source_runs:
            raise AggregationError(f"report.checkpoints[{checkpoint_index}].source_runs must be non-empty")
        for run_index, raw_source_run in enumerate(source_runs):
            source_run = _mapping(
                raw_source_run,
                f"report.checkpoints[{checkpoint_index}].source_runs[{run_index}]",
            )
            raw_run_dir = _nonempty_string(source_run.get("run_dir"), "report source run_dir")
            run_dir = Path(raw_run_dir)
            if not run_dir.is_absolute():
                raise AggregationError(f"Report source run_dir must be absolute: {raw_run_dir}")
            source_sha256 = _mapping(source_run.get("source_sha256"), "report source_sha256")
            if set(source_sha256) != set(_SOURCE_FILENAMES):
                raise AggregationError(f"Report source hashes do not cover the sealed artifact set: {run_dir}")
            for name in _SOURCE_FILENAMES:
                expected = source_sha256[name]
                source_path = run_dir / name
                if expected is None:
                    if name != artifacts.INFRA_ERRORS_FILENAME:
                        raise AggregationError(f"Report source hash cannot be null for {name}: {run_dir}")
                    if source_path.is_symlink() or source_path.exists():
                        raise AggregationError(f"Report source {name} changed since aggregation: {run_dir}")
                    continue
                digest = _sha256(expected, f"report source hash for {name}")
                try:
                    actual = _sha256_bytes(_read_regular_source_bytes(source_path, name))
                except AggregationError as error:
                    raise AggregationError(f"Report source {name} changed since aggregation: {run_dir}") from error
                if actual != digest:
                    raise AggregationError(f"Report source {name} changed since aggregation: {run_dir}")


def write_report_atomic(path: str | os.PathLike[str], report: Mapping[str, Any]) -> Path:
    """Atomically publish deterministic JSON, refusing a conflicting overwrite."""
    candidate = Path(path)
    if not candidate.is_absolute() or not candidate.name:
        raise AggregationError("Report output must be an absolute file path")
    directory = artifacts.validate_run_dir(candidate.parent, create=True)
    destination = directory / candidate.name
    content = (json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    lock_path = directory / f".{destination.name}.lock"
    with lock_path.open("a+b") as lock_stream:
        os.fchmod(lock_stream.fileno(), 0o664)
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            if destination.is_symlink() or (destination.exists() and not destination.is_file()):
                raise AggregationError(f"Report output is not a regular file: {destination}")
            if destination.exists():
                _verify_report_sources(report)
                if destination.read_bytes() == content:
                    return destination
                raise AggregationError(f"Refusing to overwrite a different report: {destination}")

            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=directory)
            temporary_path = Path(temporary_name)
            os.fchmod(descriptor, 0o644)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                _verify_report_sources(report)
                os.replace(temporary_path, destination)
                _fsync_directory(directory)
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    return destination


def _safe_segment(value: str, name: str) -> str:
    result = _nonempty_string(value, name)
    if Path(result).name != result or result in {".", ".."}:
        raise AggregationError(f"{name} must be one path segment")
    return result


def _checkpoint_mapping(value: str) -> tuple[str, Path]:
    checkpoint_id, separator, raw_path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("expected CHECKPOINT_ID=/absolute/checkpoint/run/root")
    try:
        label = _safe_segment(checkpoint_id, "checkpoint_id")
    except AggregationError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    path = Path(raw_path)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("checkpoint run root must be absolute")
    return label, path


def resolve_checkpoint_roots(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve deterministic layout discovery or explicit checkpoint roots."""
    if args.checkpoint_run:
        if args.run_id is not None or args.checkpoint_id:
            raise AggregationError("--checkpoint-run cannot be combined with --run-id or --checkpoint-id")
        pairs = args.checkpoint_run
    else:
        if args.runs_root is None or args.run_id is None or not args.checkpoint_id:
            raise AggregationError("Discovery mode requires --runs-root, --run-id, and --checkpoint-id")
        run_id = _safe_segment(args.run_id, "run_id")
        pairs = [
            (_safe_segment(checkpoint_id, "checkpoint_id"), args.runs_root / checkpoint_id / run_id)
            for checkpoint_id in args.checkpoint_id
        ]
    result: dict[str, Path] = {}
    for checkpoint_id, path in pairs:
        if checkpoint_id in result:
            raise AggregationError(f"Duplicate checkpoint_id={checkpoint_id!r}")
        result[checkpoint_id] = path
    return result


def _csv_strings(value: str, name: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values or len(set(values)) != len(values):
        raise AggregationError(f"{name} must be a comma-separated list of unique values")
    return values


def _csv_ints(value: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise AggregationError(f"{name} must be a comma-separated list of integers") from error
    if not values or any(item < 0 for item in values) or len(set(values)) != len(values):
        raise AggregationError(f"{name} must contain unique non-negative integers")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs-root", type=Path, help="Root with <checkpoint>/<run-id>/<suite> layout")
    source.add_argument(
        "--checkpoint-run",
        action="append",
        type=_checkpoint_mapping,
        default=[],
        metavar="CHECKPOINT_ID=PATH",
        help="Explicit checkpoint run root containing suite directories; repeat per checkpoint",
    )
    parser.add_argument("--run-id", help="Run ID used with --runs-root")
    parser.add_argument("--checkpoint-id", action="append", default=[], help="Checkpoint ID; repeat in discovery mode")
    parser.add_argument("--suites", default=",".join(PRIMARY_SUITES))
    parser.add_argument("--task-ids", default=",".join(str(value) for value in range(10)))
    parser.add_argument("--trials-per-task", type=int, default=50)
    parser.add_argument(
        "--require-canonical-max-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--require-zero-infra-attempts", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    roots = resolve_checkpoint_roots(args)
    output = args.output.absolute()
    for root in roots.values():
        resolved_root = artifacts.validate_run_dir(root)
        try:
            output.relative_to(resolved_root)
        except ValueError:
            pass
        else:
            raise AggregationError(f"Report output must not be inside an input run tree: {resolved_root}")
    report = aggregate_matrix(
        roots,
        suites=_csv_strings(args.suites, "suites"),
        expected_task_ids=_csv_ints(args.task_ids, "task_ids"),
        trials_per_task=args.trials_per_task,
        require_canonical_max_steps=args.require_canonical_max_steps,
        require_zero_infra_attempts=args.require_zero_infra_attempts,
    )
    destination = write_report_atomic(output, report)
    print(
        json.dumps(
            {
                "checkpoints": len(report["checkpoints"]),
                "episodes_per_checkpoint": report["input_contract"]["episodes_per_checkpoint"],
                "output": str(destination),
                "source_runs": report["source_run_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
