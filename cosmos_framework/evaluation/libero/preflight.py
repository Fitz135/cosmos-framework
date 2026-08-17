# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Headless LIBERO simulator preflight with a machine-readable manifest.

Simulator dependencies are intentionally imported only after CLI parsing and
the EGL environment checks.  This keeps ``--help`` and pure-CPU unit tests
usable when the optional LIBERO dependency group is not installed.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cosmos_framework.evaluation.libero.artifacts import validate_run_dir

PRIMARY_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EXPECTED_TASKS_PER_SUITE = 10
WARMUP_STEPS = 10
PREFLIGHT_MANIFEST_FILENAME = "preflight.json"
TARGET_ADAPTER_PATH = Path(__file__).resolve().parent / "profiles" / "edge_libero_target_adapter.json"

_CAMERA_OBSERVATION_KEYS = {
    "agentview": "agentview_image",
    "wrist": "robot0_eye_in_hand_image",
}


class PreflightError(RuntimeError):
    """Raised when the runtime cannot safely execute the canonical evaluation."""


@dataclass(frozen=True)
class SimulatorDependencies:
    """Late-bound simulator entry points, injectable for CPU-only tests."""

    benchmark: Any
    get_libero_path: Callable[[str], str]
    offscreen_render_env: Callable[..., Any]


def _libero_config_file() -> Path:
    config_root = Path(os.environ.get("LIBERO_CONFIG_PATH", "~/.libero")).expanduser()
    return config_root / "config.yaml"


def _load_simulator_dependencies() -> SimulatorDependencies:
    """Import LIBERO and robosuite only after EGL has been validated."""

    config_file = _libero_config_file()
    if not config_file.is_file():
        raise PreflightError(
            f"LIBERO config does not exist: {config_file}. Initialize it before running non-interactive preflight."
        )
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as error:
        raise PreflightError(
            "Could not import the locked LIBERO simulator stack. Install the repository's 'libero' dependency group."
        ) from error
    return SimulatorDependencies(
        benchmark=benchmark,
        get_libero_path=get_libero_path,
        offscreen_render_env=OffScreenRenderEnv,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_target_adapter() -> tuple[dict[str, Any], Path]:
    try:
        adapter = json.loads(TARGET_ADAPTER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"Could not read Edge LIBERO target adapter {TARGET_ADAPTER_PATH}: {error}") from error
    if not isinstance(adapter, dict):
        raise PreflightError(f"Target adapter must be a JSON object: {TARGET_ADAPTER_PATH}")
    raw_stats_path = adapter.get("action_stats_path")
    if not isinstance(raw_stats_path, str) or not raw_stats_path:
        raise PreflightError("Target adapter is missing action_stats_path")
    stats_path = (TARGET_ADAPTER_PATH.parent / raw_stats_path).resolve()
    if not stats_path.is_file():
        raise PreflightError(f"Target-adapter action statistics do not exist: {stats_path}")
    expected_sha = adapter.get("action_stats_sha256")
    actual_sha = _sha256(stats_path)
    if expected_sha != actual_sha:
        raise PreflightError(
            f"Target-adapter action statistics SHA-256 mismatch: expected {expected_sha!r}, got {actual_sha}"
        )
    return adapter, stats_path


def _validate_egl_environment(render_gpu_device_id: int) -> dict[str, Any]:
    mujoco_gl = os.environ.get("MUJOCO_GL")
    if mujoco_gl != "egl":
        raise PreflightError(f"MUJOCO_GL must be exactly 'egl', got {mujoco_gl!r}")

    pyopengl_platform = os.environ.get("PYOPENGL_PLATFORM")
    if pyopengl_platform not in (None, "egl"):
        raise PreflightError(f"PYOPENGL_PLATFORM must be unset or 'egl', got {pyopengl_platform!r}")
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(render_gpu_device_id))
    os.environ.setdefault("EGL_DEVICE_ID", str(render_gpu_device_id))
    return {
        "mujoco_gl": mujoco_gl,
        "pyopengl_platform": os.environ["PYOPENGL_PLATFORM"],
        "mujoco_egl_device_id": os.environ["MUJOCO_EGL_DEVICE_ID"],
        "egl_device_id": os.environ["EGL_DEVICE_ID"],
    }


def _probe_dri() -> dict[str, Any]:
    dri_root = Path("/dev/dri")
    nodes: list[dict[str, Any]] = []
    if dri_root.is_dir():
        for node in sorted((*dri_root.glob("renderD*"), *dri_root.glob("card*"))):
            nodes.append(
                {
                    "path": str(node),
                    "readable": os.access(node, os.R_OK),
                    "writable": os.access(node, os.W_OK),
                }
            )
    return {
        "path": str(dri_root),
        "exists": dri_root.is_dir(),
        "nodes": nodes,
        "accessible_nodes": [node["path"] for node in nodes if node["readable"] and node["writable"]],
    }


def _probe_egl_library() -> dict[str, Any]:
    library = ctypes.util.find_library("EGL")
    if library is None:
        raise PreflightError("Could not locate libEGL")
    try:
        ctypes.CDLL(library)
    except OSError as error:
        raise PreflightError(f"Could not load EGL library {library!r}: {error}") from error
    return {"library": library, "loaded": True}


def _package_versions(version_getter: Callable[[str], str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("libero", "robosuite", "mujoco"):
        try:
            versions[distribution] = version_getter(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise PreflightError(f"Required distribution is not installed: {distribution}") from error
    return versions


def _validate_assets(dependencies: SimulatorDependencies) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for name in ("bddl_files", "init_states", "assets"):
        try:
            path = Path(dependencies.get_libero_path(name)).expanduser().resolve()
        except Exception as error:
            raise PreflightError(f"Could not resolve LIBERO asset path {name!r}: {error}") from error
        if not path.is_dir():
            raise PreflightError(f"LIBERO asset path {name!r} is not a directory: {path}")
        assets[name] = {"path": str(path), "exists": True}
    return assets


def _shape(value: Any, *, label: str) -> list[int]:
    raw_shape = getattr(value, "shape", None)
    if raw_shape is None:
        raise PreflightError(f"{label} has no shape")
    try:
        return [int(dimension) for dimension in raw_shape]
    except (TypeError, ValueError) as error:
        raise PreflightError(f"{label} has an invalid shape: {raw_shape!r}") from error


def _smoke_suite(
    suite_name: str,
    suite: Any,
    *,
    dependencies: SimulatorDependencies,
    adapter: Mapping[str, Any],
    seed: int,
    render_gpu_device_id: int,
) -> dict[str, Any]:
    num_tasks = int(suite.n_tasks)
    if num_tasks != EXPECTED_TASKS_PER_SUITE:
        raise PreflightError(
            f"Suite {suite_name!r} must contain exactly {EXPECTED_TASKS_PER_SUITE} tasks, got {num_tasks}"
        )

    task = suite.get_task(0)
    bddl_file = (
        Path(dependencies.get_libero_path("bddl_files")) / str(task.problem_folder) / str(task.bddl_file)
    ).resolve()
    if not bddl_file.is_file():
        raise PreflightError(f"Suite {suite_name!r} task 0 BDDL file is missing: {bddl_file}")

    initial_states = suite.get_task_init_states(0)
    if len(initial_states) == 0:
        raise PreflightError(f"Suite {suite_name!r} task 0 has no default initial states")
    initial_state = initial_states[0]

    control_mode = str(adapter["control_mode"])
    control_frequency = int(adapter["control_frequency"])
    image_size = int(adapter["image_size"])
    cameras = tuple(str(camera) for camera in adapter["cameras"])
    env: Any | None = None
    try:
        env = dependencies.offscreen_render_env(
            bddl_file_name=str(bddl_file),
            camera_heights=image_size,
            camera_widths=image_size,
            camera_names=["agentview", "robot0_eye_in_hand"],
            controller=control_mode,
            control_freq=control_frequency,
            render_gpu_device_id=render_gpu_device_id,
        )
        env.seed(seed)
        env.reset()
        observation = env.set_init_state(initial_state)
        for _ in range(WARMUP_STEPS):
            observation, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

        camera_shapes: dict[str, list[int]] = {}
        camera_dtypes: dict[str, str] = {}
        for camera in cameras:
            observation_key = _CAMERA_OBSERVATION_KEYS.get(camera)
            if observation_key is None:
                raise PreflightError(f"Unsupported adapter camera: {camera!r}")
            if observation_key not in observation:
                raise PreflightError(
                    f"Suite {suite_name!r} task 0 observation is missing camera key {observation_key!r}"
                )
            image = observation[observation_key]
            shape = _shape(image, label=f"{suite_name}.{observation_key}")
            if shape != [image_size, image_size, 3]:
                raise PreflightError(
                    f"Suite {suite_name!r} task 0 camera {camera!r} has shape {shape}; "
                    f"expected {[image_size, image_size, 3]}"
                )
            camera_shapes[camera] = shape
            camera_dtypes[camera] = str(getattr(image, "dtype", "unknown"))
    except PreflightError:
        raise
    except Exception as error:
        raise PreflightError(f"Suite {suite_name!r} task 0 simulator smoke failed: {error}") from error
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    return {
        "num_tasks": num_tasks,
        "task_id": 0,
        "task_description": str(task.language),
        "bddl_file": str(bddl_file),
        "default_initial_states": len(initial_states),
        "initial_state_shape": _shape(initial_state, label=f"{suite_name}.initial_state"),
        "control_mode": control_mode,
        "control_frequency": control_frequency,
        "warmup_steps": WARMUP_STEPS,
        "camera_shapes": camera_shapes,
        "camera_dtypes": camera_dtypes,
        "render_passed": True,
    }


def run_preflight(
    *,
    seed: int = 0,
    render_gpu_device_id: int = 0,
    dependencies: SimulatorDependencies | None = None,
    version_getter: Callable[[str], str] = importlib.metadata.version,
    egl_probe: Callable[[], dict[str, Any]] = _probe_egl_library,
    dri_probe: Callable[[], dict[str, Any]] = _probe_dri,
) -> dict[str, Any]:
    """Execute the canonical four-suite task-0 simulator smoke test."""

    if render_gpu_device_id < 0:
        raise PreflightError("render_gpu_device_id must be an explicit non-negative EGL device index")
    egl_environment = _validate_egl_environment(render_gpu_device_id)
    adapter, stats_path = _load_target_adapter()
    egl = egl_probe()
    dri = dri_probe()
    versions = _package_versions(version_getter)
    simulator = dependencies if dependencies is not None else _load_simulator_dependencies()
    assets = _validate_assets(simulator)

    benchmark_dict = simulator.benchmark.get_benchmark_dict()
    suites: dict[str, Any] = {}
    for suite_name in PRIMARY_SUITES:
        suite_factory = benchmark_dict.get(suite_name)
        if suite_factory is None:
            raise PreflightError(f"LIBERO benchmark registry is missing suite {suite_name!r}")
        suites[suite_name] = _smoke_suite(
            suite_name,
            suite_factory(),
            dependencies=simulator,
            adapter=adapter,
            seed=seed,
            render_gpu_device_id=render_gpu_device_id,
        )

    return {
        "schema_version": 1,
        "status": "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "package_versions": versions,
        "environment": egl_environment,
        "rendering": {"dev_dri": dri, "egl": {**egl, "four_suite_render_smoke": True}},
        "assets": {
            **assets,
            "action_stats": {
                "path": str(stats_path),
                "sha256": adapter["action_stats_sha256"],
                "exists": True,
            },
        },
        "target_adapter": {
            "path": str(TARGET_ADAPTER_PATH),
            "sha256": _sha256(TARGET_ADAPTER_PATH),
            "adapter_id": adapter["adapter_id"],
            "control_mode": adapter["control_mode"],
            "control_frequency": adapter["control_frequency"],
            "cameras": adapter["cameras"],
            "image_size": adapter["image_size"],
        },
        "seed": seed,
        "render_gpu_device_id": render_gpu_device_id,
        "suites": suites,
    }


def _write_manifest(output_dir: str | os.PathLike[str], manifest: Mapping[str, Any]) -> Path:
    directory = validate_run_dir(output_dir, create=True)
    output_path = directory / PREFLIGHT_MANIFEST_FILENAME
    payload = (json.dumps(dict(manifest), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{PREFLIGHT_MANIFEST_FILENAME}.", suffix=".tmp", dir=directory
    )
    os.fchmod(descriptor, 0o644)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
        directory_descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def _failure_manifest(error: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Per-run directory below the canonical output root")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    dependencies: SimulatorDependencies | None = None,
    version_getter: Callable[[str], str] = importlib.metadata.version,
    egl_probe: Callable[[], dict[str, Any]] = _probe_egl_library,
    dri_probe: Callable[[], dict[str, Any]] = _probe_dri,
) -> int:
    """CLI entry point.  Every runtime failure produces a non-zero exit."""

    args = _parse_args(argv)
    try:
        output_dir = validate_run_dir(args.output_dir, create=True)
    except Exception as error:
        print(json.dumps(_failure_manifest(error), sort_keys=True), file=sys.stderr, flush=True)
        return 1

    try:
        manifest = run_preflight(
            seed=args.seed,
            render_gpu_device_id=args.render_gpu_device_id,
            dependencies=dependencies,
            version_getter=version_getter,
            egl_probe=egl_probe,
            dri_probe=dri_probe,
        )
    except Exception as error:
        manifest = _failure_manifest(error)
        _write_manifest(output_dir, manifest)
        print(json.dumps(manifest, sort_keys=True), file=sys.stderr, flush=True)
        return 1

    _write_manifest(output_dir, manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
