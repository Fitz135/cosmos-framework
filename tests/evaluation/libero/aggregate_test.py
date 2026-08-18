# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

from cosmos_framework.evaluation.libero import aggregate as aggregate_module
from cosmos_framework.evaluation.libero import artifacts
from cosmos_framework.evaluation.libero.action import GRIPPER_ADAPTER_CONTRACT
from cosmos_framework.evaluation.libero.aggregate import (
    AggregationError,
    aggregate_matrix,
    build_arg_parser,
    load_sealed_suite_run,
    resolve_checkpoint_roots,
    write_report_atomic,
)
from cosmos_framework.evaluation.libero.checkpoint_profile import compute_profile_hash
from cosmos_framework.evaluation.libero.runner import PROTOCOL_VERSION, TASK_MAX_STEPS, episode_id, finalize_run
from cosmos_framework.evaluation.libero.schema import LiberoCheckpointProfile
from cosmos_framework.evaluation.libero.seeding import SAMPLING_SEED_CONTRACT, stable_episode_seed


@pytest.fixture
def output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr(artifacts, "OUTPUT_ROOT", root)
    return root


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _profile(checkpoint_id: str) -> LiberoCheckpointProfile:
    is_base = checkpoint_id.startswith("base")
    profile = LiberoCheckpointProfile(
        schema_version=1,
        checkpoint_path=f"/checkpoints/{checkpoint_id}",
        checkpoint_format="hf",
        checkpoint_role="base" if is_base else "finetuned",
        zero_shot=is_base,
        target_adapter_id="cosmos3-edge-libero-10fps-v1",
        domain_name="libero",
        action_chunk_size=8,
        conditioning_fps=10,
        effective_action_dim=10,
        action_space="frame_wise_relative",
        rotation_space="6d",
        pose_coordinate_frame="native",
        action_normalization="quantile_rot",
        action_stats_path="/stats/libero.json",
        action_stats_sha256="74f63b4aaf9bc0623e8544d8c9fe9d3da343604096082a2621928fca039010f1",
        checkpoint_fingerprint=_digest(f"fingerprint:{checkpoint_id}"),
        format_prompt_as_json=True,
        cameras=("agentview", "wrist"),
        image_size=256,
        rotate_images=True,
        control_mode="OSC_POSE",
        control_frequency=10,
        gripper_mode="pm_one",
        weights_variant="regular" if is_base else "ema",
        profile_hash=_digest(f"profile:{checkpoint_id}"),
        profile_sources=(f"checkpoint:{checkpoint_id}", "adapter:edge-libero-v1"),
    )
    values = profile.model_dump(mode="json")
    values["profile_hash"] = compute_profile_hash(values)
    return LiberoCheckpointProfile.model_validate(values)


def _provenance(job_id: str) -> dict[str, Any]:
    return {
        "git": {"head": "a" * 40, "dirty": False},
        "uv_lock": {"path": "/repo/uv.lock", "sha256": "b" * 64},
        "python": {
            "version": "3.13.14",
            "implementation": "CPython",
            "executable": "/repo/.venv-libero/bin/python",
            "platform": "Linux-test",
        },
        "package_versions": {
            "cosmos-framework": "1.2.2",
            "libero": "0.1.1",
            "robosuite": "1.4.0",
            "mujoco": "3.3.2",
            "numpy": "2.2.6",
            "torch": "2.13.0+cu130",
        },
        "runtime_environment": {"cosmos_eval_image": "registry/eval:fixed", "cosmos_eval_job_id": job_id},
        "gpus": [{"name": "NVIDIA H200", "driver_version": "595.71.05"}],
        "nvidia_smi_error": None,
    }


def _manifest(
    checkpoint_id: str,
    suite: str,
    *,
    task_ids: tuple[int, ...],
    trials: int,
) -> dict[str, Any]:
    profile = _profile(checkpoint_id)
    raw_profile = profile.model_dump(mode="json")
    server_info = {
        "protocol_version": PROTOCOL_VERSION,
        "sampling_seed_contract": SAMPLING_SEED_CONTRACT,
        "gripper_adapter_contract": GRIPPER_ADAPTER_CONTRACT,
        "checkpoint": profile.checkpoint_path,
        "checkpoint_fingerprint": profile.checkpoint_fingerprint,
        "config_file": f"/configs/{checkpoint_id}.json",
        "config_file_type": "json",
        "weights_variant": profile.weights_variant.value,
        "policy_profile": raw_profile,
        "num_steps": 8,
        "guidance": 1.0,
        "fps": 10,
        "action_chunk_size": 8,
        "raw_action_dim": 10,
        "max_action_dim": 64,
        "action_normalization": "quantile_rot",
        "action_stats_path": "/stats/libero.json",
        "action_stats_sha256": "74f63b4aaf9bc0623e8544d8c9fe9d3da343604096082a2621928fca039010f1",
        "format_prompt_as_json": True,
        "run_name": "",
        "seed": 0,
    }
    return {
        "schema_version": 3,
        "protocol_version": PROTOCOL_VERSION,
        "sampling_seed_contract": SAMPLING_SEED_CONTRACT,
        "gripper_adapter_contract": GRIPPER_ADAPTER_CONTRACT,
        "server_url": "http://127.0.0.1:8000",
        "server_info": server_info,
        "policy_profile": raw_profile,
        "profile_hash": profile.profile_hash,
        "checkpoint_fingerprint": profile.checkpoint_fingerprint,
        "task_suite": suite,
        "task_ids": list(task_ids),
        "trials_per_task": trials,
        "base_seed": 0,
        "num_envs": 8,
        "action_horizon": 8,
        "max_steps": TASK_MAX_STEPS[suite],
        "warmup_steps": 10,
        "mujoco_gl": "egl",
        "render_gpu_device_id": 0,
        "provenance": _provenance(f"job-{checkpoint_id}-{suite}"),
    }


def _episode(suite: str, task_id: int, trial_id: int) -> dict[str, Any]:
    success = trial_id % 2 == 0
    return {
        "episode_id": episode_id(suite, task_id, trial_id),
        "task_suite": suite,
        "task_id": task_id,
        "task_description": f"description:{suite}:{task_id}",
        "trial_id": trial_id,
        "episode_seed": stable_episode_seed(0, suite, task_id, trial_id),
        "steps": 8,
        "decisions": 1,
        "termination": "success_signal" if success else "done_without_success",
        "infra_error": False,
        "success": success,
        "gripper_adapter_contract": GRIPPER_ADAPTER_CONTRACT,
        "gripper_adapter_telemetry": {
            "raw_min": -1.25,
            "raw_max": 0.5,
            "generated_value_count": 8,
            "clipped_generated_value_count": 1,
            "clipped_generated_value_rate": 0.125,
            "max_abs_overshoot": 0.25,
        },
        "elapsed_s": 1.0,
    }


def _create_suite_run(
    output_root: Path,
    checkpoint_id: str,
    suite: str,
    *,
    task_ids: tuple[int, ...] = (0, 1),
    trials: int = 2,
    historical_infra: bool = False,
) -> Path:
    run_dir = output_root / "runs" / checkpoint_id / "full" / suite
    manifest = _manifest(checkpoint_id, suite, task_ids=task_ids, trials=trials)
    artifacts.write_manifest(run_dir, manifest)
    if historical_infra:
        artifacts.append_infra_error(
            run_dir,
            {
                "attempt_id": "old-attempt:00000000",
                "episode_id": episode_id(suite, task_ids[0], 0),
                "task_suite": suite,
                "task_id": task_ids[0],
                "trial_id": 0,
                "episode_seed": stable_episode_seed(0, suite, task_ids[0], 0),
                "infra_error": True,
                "infra_error_type": "policy_server",
            },
        )
    for task_id in task_ids:
        for trial_id in range(trials):
            artifacts.append_episode(run_dir, _episode(suite, task_id, trial_id))
    finalize_run(run_dir, profile_hash=str(manifest["profile_hash"]))
    return run_dir


def _create_matrix(
    output_root: Path,
    *,
    checkpoint_ids: tuple[str, ...] = ("base", "edge-5k"),
    suites: tuple[str, ...] = ("libero_spatial", "libero_object"),
    task_ids: tuple[int, ...] = (0, 1),
    trials: int = 2,
) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for checkpoint_id in checkpoint_ids:
        roots[checkpoint_id] = output_root / "runs" / checkpoint_id / "full"
        for suite in suites:
            _create_suite_run(output_root, checkpoint_id, suite, task_ids=task_ids, trials=trials)
    return roots


def test_matrix_recomputes_micro_metrics_and_never_pools_checkpoints(output_root: Path) -> None:
    suites = ("libero_spatial", "libero_object")
    roots = _create_matrix(output_root, suites=suites)

    report = aggregate_matrix(
        roots,
        suites=suites,
        expected_task_ids=(0, 1),
        trials_per_task=2,
        require_zero_infra_attempts=True,
    )

    assert "overall" not in report
    assert report["source_run_count"] == 4
    assert report["total_episode_count"] == 16
    assert report["input_contract"]["episodes_per_checkpoint"] == 8
    assert [result["checkpoint_id"] for result in report["checkpoints"]] == ["base", "edge-5k"]
    for result in report["checkpoints"]:
        assert result["episode_count"] == 8
        assert result["overall"]["aggregation"] == "micro_episode_weighted"
        assert result["overall"]["evaluated_episodes"] == 8
        assert result["overall"]["successes"] == 4
        assert result["overall"]["success_rate"] == 0.5
        assert result["overall"]["steps"] == {
            "total": 64,
            "mean": 8.0,
            "min": 8,
            "max": 8,
            "success_mean": 8.0,
            "failure_mean": 8.0,
        }
        assert result["overall"]["decisions"]["total"] == 8
        assert len(result["suites"]) == 2
        assert len(result["tasks"]) == 4
        assert result["gripper_adapter"]["generated_value_count"] == 64
        assert result["gripper_adapter"]["clipped_generated_value_rate"] == 0.125
        assert all(source["source_sha256"][artifacts.INFRA_ERRORS_FILENAME] is None for source in result["source_runs"])


def test_sealed_loader_rejects_active_run_and_tampered_metrics(output_root: Path) -> None:
    run_dir = _create_suite_run(output_root, "base", "libero_spatial")
    marker = run_dir / artifacts.COMPLETION_FILENAME
    marker_bytes = marker.read_bytes()
    marker.unlink()
    with pytest.raises(AggregationError, match="_SUCCESS"):
        load_sealed_suite_run(
            "base",
            "libero_spatial",
            run_dir,
            expected_task_ids=(0, 1),
            trials_per_task=2,
        )
    marker.write_bytes(marker_bytes)

    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["overall"]["successes"] += 1
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(AggregationError, match="re-derived"):
        load_sealed_suite_run(
            "base",
            "libero_spatial",
            run_dir,
            expected_task_ids=(0, 1),
            trials_per_task=2,
        )


def test_episode_decision_and_generated_chunk_contract_is_strict(output_root: Path) -> None:
    run_dir = _create_suite_run(output_root, "base", "libero_spatial")
    episodes_path = run_dir / artifacts.EPISODES_FILENAME
    records = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    records[0]["decisions"] = 2
    episodes_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(AggregationError, match=r"ceil\(steps/action_horizon\)"):
        load_sealed_suite_run(
            "base",
            "libero_spatial",
            run_dir,
            expected_task_ids=(0, 1),
            trials_per_task=2,
        )


def test_missing_infra_journal_is_zero_but_strict_mode_rejects_history(output_root: Path) -> None:
    clean_dir = _create_suite_run(output_root, "base", "libero_spatial")
    assert not (clean_dir / artifacts.INFRA_ERRORS_FILENAME).exists()
    clean = load_sealed_suite_run(
        "base",
        "libero_spatial",
        clean_dir,
        expected_task_ids=(0, 1),
        trials_per_task=2,
        require_zero_infra_attempts=True,
    )
    assert clean.infra_attempts == ()

    history_dir = _create_suite_run(output_root, "edge-5k", "libero_spatial", historical_infra=True)
    permissive = load_sealed_suite_run(
        "edge-5k",
        "libero_spatial",
        history_dir,
        expected_task_ids=(0, 1),
        trials_per_task=2,
    )
    assert len(permissive.infra_attempts) == 1
    with pytest.raises(AggregationError, match="historical infrastructure"):
        load_sealed_suite_run(
            "edge-5k",
            "libero_spatial",
            history_dir,
            expected_task_ids=(0, 1),
            trials_per_task=2,
            require_zero_infra_attempts=True,
        )


def test_loader_rejects_infrastructure_attempt_outside_manifest_grid(output_root: Path) -> None:
    run_dir = _create_suite_run(output_root, "base", "libero_spatial", historical_infra=True)
    infra_path = run_dir / artifacts.INFRA_ERRORS_FILENAME
    record = json.loads(infra_path.read_text())
    record["task_suite"] = "wrong_suite"
    record["task_id"] = 999
    record["trial_id"] = 999
    infra_path.write_text(json.dumps(record) + "\n")

    with pytest.raises(AggregationError, match="manifest episode grid"):
        load_sealed_suite_run(
            "base",
            "libero_spatial",
            run_dir,
            expected_task_ids=(0, 1),
            trials_per_task=2,
        )


def test_loader_recomputes_infrastructure_attempt_seed(output_root: Path) -> None:
    run_dir = _create_suite_run(output_root, "base", "libero_spatial", historical_infra=True)
    infra_path = run_dir / artifacts.INFRA_ERRORS_FILENAME
    record = json.loads(infra_path.read_text())
    record["episode_seed"] = stable_episode_seed(0, "libero_spatial", 0, 0) + 1
    infra_path.write_text(json.dumps(record) + "\n")

    with pytest.raises(AggregationError, match="seed does not match"):
        load_sealed_suite_run(
            "base",
            "libero_spatial",
            run_dir,
            expected_task_ids=(0, 1),
            trials_per_task=2,
        )


def test_loader_requires_infrastructure_attempt_seed(output_root: Path) -> None:
    run_dir = _create_suite_run(output_root, "base", "libero_spatial", historical_infra=True)
    infra_path = run_dir / artifacts.INFRA_ERRORS_FILENAME
    record = json.loads(infra_path.read_text())
    del record["episode_seed"]
    infra_path.write_text(json.dumps(record) + "\n")

    with pytest.raises(AggregationError, match="seed does not match"):
        load_sealed_suite_run(
            "base",
            "libero_spatial",
            run_dir,
            expected_task_ids=(0, 1),
            trials_per_task=2,
        )


def test_sealed_loader_does_not_open_root_owned_append_lock(output_root: Path) -> None:
    run_dir = _create_suite_run(output_root, "base", "libero_spatial")
    lock_path = run_dir / ".episodes.lock"
    lock_path.chmod(0)
    run_dir.chmod(0o555)
    try:
        sealed = load_sealed_suite_run(
            "base",
            "libero_spatial",
            run_dir,
            expected_task_ids=(0, 1),
            trials_per_task=2,
            require_zero_infra_attempts=True,
        )
    finally:
        run_dir.chmod(0o755)
        lock_path.chmod(0o644)
    assert len(sealed.episodes) == 4


def test_loader_rejects_source_mutation_after_snapshot(
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _create_suite_run(output_root, "base", "libero_spatial")
    episodes_path = run_dir / artifacts.EPISODES_FILENAME
    original_validate = aggregate_module._validate_episode
    mutated = False

    def mutate_after_snapshot(*args: Any, **kwargs: Any) -> None:
        nonlocal mutated
        if not mutated:
            episodes_path.write_bytes(episodes_path.read_bytes() + b" ")
            mutated = True
        original_validate(*args, **kwargs)

    monkeypatch.setattr(aggregate_module, "_validate_episode", mutate_after_snapshot)
    with pytest.raises(AggregationError, match="changed while validating"):
        load_sealed_suite_run(
            "base",
            "libero_spatial",
            run_dir,
            expected_task_ids=(0, 1),
            trials_per_task=2,
        )


def test_loader_recomputes_profile_hash_and_requires_canonical_edge_semantics(output_root: Path) -> None:
    run_dir = _create_suite_run(output_root, "base", "libero_spatial")
    manifest = artifacts.read_manifest(run_dir)
    manifest["profile_hash"] = "0" * 64
    manifest["policy_profile"]["profile_hash"] = "0" * 64
    manifest["server_info"]["policy_profile"]["profile_hash"] = "0" * 64
    artifacts.write_manifest(run_dir, manifest)
    with pytest.raises(AggregationError, match="Semantic profile_hash mismatch"):
        load_sealed_suite_run(
            "base",
            "libero_spatial",
            run_dir,
            expected_task_ids=(0, 1),
            trials_per_task=2,
        )

    manifest = _manifest("base", "libero_spatial", task_ids=(0, 1), trials=2)
    manifest["policy_profile"]["image_size"] = 128
    semantic_hash = compute_profile_hash(manifest["policy_profile"])
    manifest["profile_hash"] = semantic_hash
    manifest["policy_profile"]["profile_hash"] = semantic_hash
    manifest["server_info"]["policy_profile"] = dict(manifest["policy_profile"])
    artifacts.write_manifest(run_dir, manifest)
    with pytest.raises(AggregationError, match="canonical Edge LIBERO contract"):
        load_sealed_suite_run(
            "base",
            "libero_spatial",
            run_dir,
            expected_task_ids=(0, 1),
            trials_per_task=2,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint", "/checkpoints/wrong"),
        ("checkpoint_fingerprint", "0" * 64),
        ("weights_variant", "regular"),
        ("action_chunk_size", 999),
        ("raw_action_dim", 999),
        ("fps", 999),
        ("format_prompt_as_json", False),
        ("action_normalization", "meanstd"),
        ("action_stats_path", "/stats/wrong.json"),
        ("action_stats_sha256", "0" * 64),
        ("config_file", None),
        ("config_file_type", "unsupported"),
        ("max_action_dim", 9),
    ],
)
def test_loader_rejects_server_info_contract_drift(output_root: Path, field: str, value: Any) -> None:
    run_dir = _create_suite_run(output_root, "edge-5k", "libero_spatial")
    manifest = artifacts.read_manifest(run_dir)
    manifest["server_info"][field] = value
    artifacts.write_manifest(run_dir, manifest)

    with pytest.raises(AggregationError, match=rf"server_info\.{field}"):
        load_sealed_suite_run(
            "edge-5k",
            "libero_spatial",
            run_dir,
            expected_task_ids=(0, 1),
            trials_per_task=2,
        )


def test_loader_accepts_relative_module_config_identity(output_root: Path) -> None:
    run_dir = _create_suite_run(output_root, "edge-5k", "libero_spatial")
    manifest = artifacts.read_manifest(run_dir)
    manifest["server_info"]["config_file"] = "cosmos_framework.configs.libero"
    manifest["server_info"]["config_file_type"] = "module"
    artifacts.write_manifest(run_dir, manifest)

    sealed = load_sealed_suite_run(
        "edge-5k",
        "libero_spatial",
        run_dir,
        expected_task_ids=(0, 1),
        trials_per_task=2,
    )
    assert sealed.manifest["server_info"]["config_file_type"] == "module"


def test_loader_rejects_seed_outside_runner_signed63_contract(output_root: Path) -> None:
    run_dir = _create_suite_run(output_root, "base", "libero_spatial")
    manifest = artifacts.read_manifest(run_dir)
    manifest["base_seed"] = 2**63
    artifacts.write_manifest(run_dir, manifest)
    with pytest.raises(AggregationError, match="fit signed 64-bit"):
        load_sealed_suite_run(
            "base",
            "libero_spatial",
            run_dir,
            expected_task_ids=(0, 1),
            trials_per_task=2,
        )


def test_matrix_rejects_provenance_drift(output_root: Path) -> None:
    suites = ("libero_spatial", "libero_object")
    roots = _create_matrix(output_root, suites=suites)
    drifted_dir = roots["edge-5k"] / "libero_object"
    manifest = artifacts.read_manifest(drifted_dir)
    manifest["provenance"]["package_versions"]["numpy"] = "different"
    artifacts.write_manifest(drifted_dir, manifest)

    with pytest.raises(AggregationError, match="Suite manifests disagree"):
        aggregate_matrix(
            roots,
            suites=suites,
            expected_task_ids=(0, 1),
            trials_per_task=2,
        )


def test_atomic_report_is_deterministic_idempotent_and_refuses_conflict(output_root: Path) -> None:
    destination = output_root / "reports" / "full" / "summary.json"
    report = {"schema_version": 1, "checkpoints": [{"checkpoint_id": "base"}]}

    first = write_report_atomic(destination, report)
    first_bytes = first.read_bytes()
    assert stat.S_IMODE(first.stat().st_mode) == 0o644
    assert stat.S_IMODE((first.parent / ".summary.json.lock").stat().st_mode) == 0o664
    assert write_report_atomic(destination, report) == first
    assert first.read_bytes() == first_bytes
    assert not list(first.parent.glob(".summary.json.*.tmp"))
    with pytest.raises(AggregationError, match="Refusing to overwrite"):
        write_report_atomic(destination, {**report, "schema_version": 2})


def test_atomic_report_serializes_concurrent_conflicting_publishers(output_root: Path) -> None:
    destination = output_root / "reports" / "race" / "summary.json"
    barrier = Barrier(2)

    def publish(schema_version: int) -> Path | AggregationError:
        barrier.wait()
        try:
            return write_report_atomic(destination, {"schema_version": schema_version})
        except AggregationError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, (1, 2)))

    assert sum(isinstance(outcome, Path) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, AggregationError) for outcome in outcomes) == 1
    assert json.loads(destination.read_text())["schema_version"] in {1, 2}


def test_atomic_report_revalidates_sources_before_publish(output_root: Path) -> None:
    roots = _create_matrix(output_root, checkpoint_ids=("base",), suites=("libero_spatial",))
    report = aggregate_matrix(
        roots,
        suites=("libero_spatial",),
        expected_task_ids=(0, 1),
        trials_per_task=2,
    )
    episodes_path = roots["base"] / "libero_spatial" / artifacts.EPISODES_FILENAME
    episodes_path.write_bytes(episodes_path.read_bytes() + b" ")
    destination = output_root / "reports" / "stale" / "summary.json"

    with pytest.raises(AggregationError, match="changed since aggregation"):
        write_report_atomic(destination, report)
    assert not destination.exists()


def test_discovery_and_explicit_checkpoint_roots() -> None:
    parser = build_arg_parser()
    discovery = parser.parse_args(
        [
            "--runs-root",
            "/output/runs",
            "--run-id",
            "full-v1",
            "--checkpoint-id",
            "base",
            "--checkpoint-id",
            "edge-5k",
            "--output",
            "/output/reports/summary.json",
        ]
    )
    assert resolve_checkpoint_roots(discovery) == {
        "base": Path("/output/runs/base/full-v1"),
        "edge-5k": Path("/output/runs/edge-5k/full-v1"),
    }

    explicit = parser.parse_args(
        [
            "--checkpoint-run",
            "base=/output/runs/base/full-v1",
            "--checkpoint-run",
            "edge-5k=/output/runs/edge-5k/full-v1",
            "--output",
            "/output/reports/summary.json",
        ]
    )
    assert resolve_checkpoint_roots(explicit) == {
        "base": Path("/output/runs/base/full-v1"),
        "edge-5k": Path("/output/runs/edge-5k/full-v1"),
    }

    invalid = argparse.Namespace(
        checkpoint_run=[("base", Path("/output/base"))],
        runs_root=None,
        run_id="full-v1",
        checkpoint_id=[],
    )
    with pytest.raises(AggregationError, match="cannot be combined"):
        resolve_checkpoint_roots(invalid)
