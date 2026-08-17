# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cosmos_framework.evaluation.libero import artifacts, preflight
from cosmos_framework.evaluation.libero.schema import LiberoTargetAdapterSpec


class _Array:
    def __init__(self, shape: tuple[int, ...], dtype: str = "float64") -> None:
        self.shape = shape
        self.dtype = dtype


class _FakeTask:
    def __init__(self, suite_name: str) -> None:
        self.problem_folder = suite_name
        self.bddl_file = "task_0.bddl"
        self.language = f"task zero for {suite_name}"


class _FakeSuite:
    n_tasks = 10

    def __init__(self, suite_name: str) -> None:
        self.task = _FakeTask(suite_name)

    def get_task(self, task_id: int) -> _FakeTask:
        assert task_id == 0
        return self.task

    def get_task_init_states(self, task_id: int) -> list[_Array]:
        assert task_id == 0
        return [_Array((42,))]


class _FakeBenchmark:
    def get_benchmark_dict(self) -> dict[str, Any]:
        return {name: (lambda suite_name=name: _FakeSuite(suite_name)) for name in preflight.PRIMARY_SUITES}


class _FakeEnv:
    instances: list["_FakeEnv"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.seed_value: int | None = None
        self.step_count = 0
        self.closed = False
        self.instances.append(self)

    @staticmethod
    def _observation() -> dict[str, _Array]:
        return {
            "agentview_image": _Array((256, 256, 3), "uint8"),
            "robot0_eye_in_hand_image": _Array((256, 256, 3), "uint8"),
        }

    def seed(self, seed: int) -> None:
        self.seed_value = seed

    def reset(self) -> dict[str, _Array]:
        return self._observation()

    def set_init_state(self, state: _Array) -> dict[str, _Array]:
        assert state.shape == (42,)
        return self._observation()

    def step(self, action: list[float]) -> tuple[dict[str, _Array], float, bool, dict[str, Any]]:
        assert action == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        self.step_count += 1
        return self._observation(), 0.0, False, {}

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_dependencies(tmp_path: Path) -> preflight.SimulatorDependencies:
    _FakeEnv.instances.clear()
    bddl_root = tmp_path / "bddl_files"
    init_states = tmp_path / "init_states"
    assets = tmp_path / "assets"
    for path in (bddl_root, init_states, assets):
        path.mkdir()
    for suite_name in preflight.PRIMARY_SUITES:
        suite_dir = bddl_root / suite_name
        suite_dir.mkdir()
        (suite_dir / "task_0.bddl").write_text("(define (problem task-0))")

    paths = {"bddl_files": bddl_root, "init_states": init_states, "assets": assets}

    return preflight.SimulatorDependencies(
        benchmark=_FakeBenchmark(),
        get_libero_path=lambda name: str(paths[name]),
        offscreen_render_env=_FakeEnv,
    )


def _version(distribution: str) -> str:
    return {"libero": "0.1.1", "robosuite": "1.4.0", "mujoco": "3.3.2"}[distribution]


def _egl() -> dict[str, Any]:
    return {"library": "libEGL.so.1", "loaded": True}


def _dri() -> dict[str, Any]:
    return {"path": "/dev/dri", "exists": False, "nodes": [], "accessible_nodes": []}


def test_edge_target_adapter_locks_the_canonical_contract() -> None:
    raw = json.loads(preflight.TARGET_ADAPTER_PATH.read_text())
    adapter = LiberoTargetAdapterSpec.model_validate(raw)

    assert "weights_variant" not in raw
    assert adapter.domain_name == "libero"
    assert adapter.action_chunk_size == 8
    assert adapter.conditioning_fps == 10
    assert adapter.effective_action_dim == 10
    assert adapter.action_space == "frame_wise_relative"
    assert adapter.rotation_space == "6d"
    assert adapter.pose_coordinate_frame == "native"
    assert adapter.action_normalization == "quantile_rot"
    assert adapter.format_prompt_as_json is True
    assert adapter.cameras == ("agentview", "wrist")
    assert adapter.image_size == 256
    assert adapter.rotate_images is True
    assert adapter.control_mode == "OSC_POSE"
    assert adapter.control_frequency == 10
    assert adapter.gripper_mode == "pm_one"

    stats_path = (preflight.TARGET_ADAPTER_PATH.parent / adapter.action_stats_path).resolve()
    assert stats_path.is_file()
    assert preflight._sha256(stats_path) == adapter.action_stats_sha256


def test_run_preflight_checks_all_four_suites_with_explicit_control(
    fake_dependencies: preflight.SimulatorDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "2")
    monkeypatch.setenv("EGL_DEVICE_ID", "2")

    manifest = preflight.run_preflight(
        seed=17,
        render_gpu_device_id=2,
        dependencies=fake_dependencies,
        version_getter=_version,
        egl_probe=_egl,
        dri_probe=_dri,
    )

    assert manifest["status"] == "passed"
    assert manifest["package_versions"] == {"libero": "0.1.1", "robosuite": "1.4.0", "mujoco": "3.3.2"}
    assert tuple(manifest["suites"]) == preflight.PRIMARY_SUITES
    assert all(suite["num_tasks"] == 10 for suite in manifest["suites"].values())
    assert all(suite["warmup_steps"] == 10 for suite in manifest["suites"].values())
    assert all(suite["render_passed"] for suite in manifest["suites"].values())
    assert manifest["rendering"]["egl"]["four_suite_render_smoke"] is True
    assert manifest["environment"]["mujoco_gl"] == "egl"

    assert len(_FakeEnv.instances) == 4
    for env in _FakeEnv.instances:
        assert env.kwargs["controller"] == "OSC_POSE"
        assert env.kwargs["control_freq"] == 10
        assert env.kwargs["render_gpu_device_id"] == 2
        assert env.kwargs["camera_names"] == ["agentview", "robot0_eye_in_hand"]
        assert env.seed_value == 17
        assert env.step_count == 10
        assert env.closed


def test_cli_writes_manifest_only_below_canonical_root(
    tmp_path: Path,
    fake_dependencies: preflight.SimulatorDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(artifacts, "OUTPUT_ROOT", output_root)
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
    monkeypatch.setenv("EGL_DEVICE_ID", "0")
    run_dir = output_root / "cosmos-framework-eval" / "preflight-run"

    result = preflight.main(
        ["--output-dir", str(run_dir), "--seed", "3", "--render-gpu-device-id", "0"],
        dependencies=fake_dependencies,
        version_getter=_version,
        egl_probe=_egl,
        dri_probe=_dri,
    )

    assert result == 0
    manifest_path = run_dir / preflight.PREFLIGHT_MANIFEST_FILENAME
    assert json.loads(manifest_path.read_text())["status"] == "passed"
    assert manifest_path.stat().st_mode & 0o777 == 0o644
    assert not list(run_dir.glob(".preflight.json.*.tmp"))

    outside = tmp_path / "outside"
    assert (
        preflight.main(
            ["--output-dir", str(outside)],
            dependencies=fake_dependencies,
            version_getter=_version,
            egl_probe=_egl,
            dri_probe=_dri,
        )
        == 1
    )
    assert not outside.exists()


def test_cli_fails_before_lazy_import_when_mujoco_gl_is_not_egl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(artifacts, "OUTPUT_ROOT", output_root)
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    imported = False

    def forbidden_import() -> preflight.SimulatorDependencies:
        nonlocal imported
        imported = True
        raise AssertionError("simulator dependencies must not be imported")

    monkeypatch.setattr(preflight, "_load_simulator_dependencies", forbidden_import)
    run_dir = output_root / "eval" / "failed-preflight"
    result = preflight.main(
        ["--output-dir", str(run_dir)],
        version_getter=_version,
        egl_probe=_egl,
        dri_probe=_dri,
    )

    assert result == 1
    assert not imported
    manifest = json.loads((run_dir / preflight.PREFLIGHT_MANIFEST_FILENAME).read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "PreflightError"
    assert "MUJOCO_GL" in manifest["error"]["message"]


def test_suite_task_count_mismatch_is_a_hard_failure(
    fake_dependencies: preflight.SimulatorDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
    monkeypatch.setenv("EGL_DEVICE_ID", "0")
    benchmark_dict = fake_dependencies.benchmark.get_benchmark_dict()
    bad_suite = _FakeSuite("libero_spatial")
    bad_suite.n_tasks = 9
    benchmark_dict["libero_spatial"] = lambda: bad_suite
    fake_dependencies = preflight.SimulatorDependencies(
        benchmark=SimpleNamespace(get_benchmark_dict=lambda: benchmark_dict),
        get_libero_path=fake_dependencies.get_libero_path,
        offscreen_render_env=fake_dependencies.offscreen_render_env,
    )

    with pytest.raises(preflight.PreflightError, match="exactly 10 tasks"):
        preflight.run_preflight(
            dependencies=fake_dependencies,
            version_getter=_version,
            egl_probe=_egl,
            dri_probe=_dri,
        )
