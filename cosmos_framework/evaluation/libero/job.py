# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Single-checkpoint orchestration for strict Cosmos Edge LIBERO evaluation.

This module deliberately stays thin: it resolves the expected checkpoint
profile, starts the existing action-policy server, validates its versioned
/info handshake, and then invokes the existing runner in a subprocess.
Server and runner output is appended to files inside the canonical run
directory so failed and resumed invocations retain their diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import requests

from cosmos_framework.evaluation.libero import artifacts
from cosmos_framework.evaluation.libero.checkpoint_profile import resolve_checkpoint_profile
from cosmos_framework.evaluation.libero.runner import (
    EDGE_ACTION_CHUNK_SIZE,
    METRICS_FILENAME,
    TASK_MAX_STEPS,
    PolicyHandshake,
    parse_policy_handshake,
)
from cosmos_framework.evaluation.libero.schema import CheckpointFormat, LiberoCheckpointProfile

SERVER_MODULE = "cosmos_framework.scripts.action_policy_server_libero"
RUNNER_MODULE = "cosmos_framework.evaluation.libero.runner"
SERVER_LOG_FILENAME = "server.log"
RUNNER_LOG_FILENAME = "runner.log"
JOB_LOG_FILENAME = "job.log"
SERVER_RUNTIME_DIRNAME = "server_runtime"


class LiberoJobError(RuntimeError):
    """Base error for orchestration failures."""


class LiberoProcessError(LiberoJobError):
    """A managed server or runner subprocess exited unsuccessfully."""

    def __init__(self, phase: str, returncode: int, log_path: Path) -> None:
        self.phase = phase
        self.returncode = int(returncode)
        self.log_path = log_path
        super().__init__(f"{phase} exited with status {returncode}; see {log_path}")


class LiberoJobInterrupted(KeyboardInterrupt):
    """Raised inside the orchestrator so SIGTERM unwinds managed children."""

    def __init__(self) -> None:
        super().__init__("orchestrator received SIGTERM")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(stream: TextIO, message: str) -> None:
    stream.write(f"[{_timestamp()}] {message}\n")
    stream.flush()


def _set_shared_log_mode(stream: TextIO) -> None:
    os.fchmod(stream.fileno(), 0o644)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _raise_on_sigterm(_signal_number: int, _frame: Any) -> None:
    raise LiberoJobInterrupted()


@contextmanager
def _sigterm_as_interrupt() -> Any:
    """Turn SIGTERM into an unwind while managed subprocesses are live."""
    try:
        previous_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _raise_on_sigterm)
    except ValueError:
        # signal.signal is restricted to the main thread. Programmatic callers
        # from another thread still retain KeyboardInterrupt/finally cleanup.
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def build_server_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    """Build the existing action-policy server command without a shell."""
    command = [
        sys.executable,
        "-m",
        SERVER_MODULE,
        "--checkpoint-path",
        str(args.checkpoint_path),
        "--output-dir",
        str(run_dir / SERVER_RUNTIME_DIRNAME),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.server_port),
        "--num-steps",
        str(args.num_steps),
        "--guidance",
        str(args.guidance),
    ]
    optional_paths = (
        ("--config-file", args.config_file),
        ("--policy-profile-path", args.policy_profile_path),
        ("--target-adapter-path", args.target_adapter_path),
    )
    for flag, value in optional_paths:
        if value is not None:
            command.extend((flag, str(value)))
    if args.weights_variant is not None:
        command.extend(("--weights-variant", args.weights_variant))
    return command


def build_runner_command(args: argparse.Namespace, run_dir: Path, server_url: str) -> list[str]:
    """Build the stable LIBERO runner CLI, optionally in another Python env."""
    runner_python = os.path.expanduser(args.runner_python) if args.runner_python else sys.executable
    return [
        str(runner_python),
        "-m",
        RUNNER_MODULE,
        "--server-url",
        server_url,
        "--output-dir",
        str(run_dir),
        "--task-suite",
        args.task_suite,
        "--task-ids",
        args.task_ids,
        "--trials",
        str(args.trials),
        "--num-envs",
        str(args.num_envs),
        "--action-horizon",
        str(args.action_horizon),
        "--seed",
        str(args.seed),
        "--max-steps",
        str(args.max_steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--mujoco-gl",
        args.mujoco_gl,
        "--render-gpu-device-id",
        str(args.render_gpu_device_id),
        "--request-timeout",
        str(args.request_timeout),
    ]


def _fetch_server_info(server_url: str, *, timeout_s: float) -> Mapping[str, Any]:
    response = requests.get(f"{server_url.rstrip('/')}/info", timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise LiberoJobError("GET /info must return a JSON object")
    return payload


def wait_for_policy_server(
    server_url: str,
    server_process: subprocess.Popen[Any],
    expected_profile: LiberoCheckpointProfile,
    *,
    startup_timeout_s: float,
    request_timeout_s: float,
    poll_interval_s: float,
    server_log_path: Path,
) -> PolicyHandshake:
    """Poll the server until a matching strict handshake is available."""
    deadline = time.monotonic() + startup_timeout_s
    last_transport_error: BaseException | None = None

    def _timeout_error() -> LiberoJobError:
        detail = f": {last_transport_error}" if last_transport_error is not None else ""
        return LiberoJobError(
            f"Timed out after {startup_timeout_s:.1f}s waiting for {server_url}/info{detail}; see {server_log_path}"
        )

    while True:
        returncode = server_process.poll()
        if returncode is not None:
            raise LiberoProcessError("policy server", returncode, server_log_path)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _timeout_error()
        try:
            info = _fetch_server_info(
                server_url,
                timeout_s=min(request_timeout_s, remaining),
            )
        except requests.RequestException as error:
            last_transport_error = error
        else:
            handshake = parse_policy_handshake(info)
            if handshake.profile != expected_profile:
                raise LiberoJobError(
                    "Policy server handshake does not match the locally resolved checkpoint profile: "
                    f"expected hash={expected_profile.profile_hash}, got hash={handshake.profile.profile_hash}"
                )
            return handshake

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _timeout_error()
        time.sleep(min(poll_interval_s, remaining))


def _signal_process_group(process: subprocess.Popen[Any], signal_number: int) -> None:
    """Signal an isolated child process group, with a direct-process fallback."""
    try:
        process_id = process.pid
    except AttributeError:
        process_id = None
    if process_id is not None:
        try:
            os.killpg(process_id, signal_number)
            return
        except ProcessLookupError:
            if process.poll() is not None:
                return
    try:
        if signal_number == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        return


def _terminate_process(
    process: subprocess.Popen[Any],
    *,
    timeout_s: float,
) -> None:
    """Terminate an isolated child group, escalating to kill after timeout."""
    if process.poll() is not None:
        return
    _signal_process_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGKILL)
        process.wait(timeout=timeout_s)


def _cleanup_processes(
    processes: Sequence[tuple[str, subprocess.Popen[Any] | None]],
    *,
    timeout_s: float,
    job_log: TextIO,
    preserve_active_error: bool,
) -> None:
    """Best-effort cleanup without allowing one child to strand another."""
    cleanup_errors: list[BaseException] = []
    for phase, process in processes:
        if process is None:
            continue
        try:
            _terminate_process(process, timeout_s=timeout_s)
        except BaseException as error:
            cleanup_errors.append(error)
            _log(job_log, f"failed to stop {phase}: {type(error).__name__}: {error}")
    if cleanup_errors and not preserve_active_error:
        raise LiberoJobError("Failed to stop one or more managed subprocesses") from cleanup_errors[0]


def _ensure_port_available(port: int) -> None:
    """Reject an already-bound port before an expensive model load."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise LiberoJobError(f"Policy server port 127.0.0.1:{port} is already in use") from error


def _validate_runner_python(args: argparse.Namespace) -> None:
    if args.runner_python is None:
        return
    candidate = Path(args.runner_python).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"runner_python must resolve to an absolute executable path, got {candidate}")
    candidate = candidate.resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise ValueError(f"runner_python is not an executable file: {candidate}")
    args.runner_python = str(candidate)


def _validate_runner_metrics(run_dir: Path, expected_profile: LiberoCheckpointProfile) -> Mapping[str, Any]:
    """Reject a nominally successful runner with no valid evaluation episodes."""
    metrics_path = run_dir / METRICS_FILENAME
    try:
        with metrics_path.open(encoding="utf-8") as stream:
            metrics = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise LiberoJobError(f"Runner did not produce valid metrics at {metrics_path}") from error
    if not isinstance(metrics, Mapping):
        raise LiberoJobError(f"Runner metrics must contain a JSON object: {metrics_path}")
    if metrics.get("profile_hash") != expected_profile.profile_hash:
        raise LiberoJobError("Runner metrics profile_hash does not match the validated policy profile")
    overall = metrics.get("overall")
    infrastructure = metrics.get("infrastructure_errors")
    if not isinstance(overall, Mapping) or not isinstance(infrastructure, Mapping):
        raise LiberoJobError("Runner metrics are missing overall or infrastructure_errors objects")
    evaluated = overall.get("evaluated_episodes")
    overall_infra = overall.get("infra_errors")
    infrastructure_count = infrastructure.get("count")
    for name, value in (
        ("overall.evaluated_episodes", evaluated),
        ("overall.infra_errors", overall_infra),
        ("infrastructure_errors.count", infrastructure_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LiberoJobError(f"Runner metrics field {name} must be a non-negative integer, got {value!r}")
    if evaluated == 0 or overall_infra != 0:
        raise LiberoJobError(
            f"Runner produced no clean successful evaluation result: "
            f"evaluated_episodes={evaluated}, infra_errors={overall_infra}"
        )
    return metrics


def _normalize_job_paths(args: argparse.Namespace) -> None:
    for name in ("checkpoint_path", "config_file", "policy_profile_path", "target_adapter_path"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value).expanduser().resolve())


def _validate_job_args(args: argparse.Namespace) -> None:
    _normalize_job_paths(args)
    if (
        isinstance(args.server_port, bool)
        or not isinstance(args.server_port, int)
        or not 1 <= args.server_port <= 65535
    ):
        raise ValueError(f"server_port must be in [1, 65535], got {args.server_port}")
    positive_scalars = {
        "server_startup_timeout": args.server_startup_timeout,
        "server_info_timeout": args.server_info_timeout,
        "server_poll_interval": args.server_poll_interval,
        "server_stop_timeout": args.server_stop_timeout,
        "request_timeout": args.request_timeout,
        "guidance": args.guidance,
    }
    for name, value in positive_scalars.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{name} must be positive, got {value!r}")
    for name in ("trials", "num_envs", "action_horizon", "num_steps"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if args.action_horizon > EDGE_ACTION_CHUNK_SIZE:
        raise ValueError(f"action_horizon must not exceed {EDGE_ACTION_CHUNK_SIZE}, got {args.action_horizon}")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or not 0 <= args.seed <= 2**63 - 1:
        raise ValueError(f"seed must fit in a non-negative signed 64-bit integer, got {args.seed}")
    for name in ("max_steps", "warmup_steps", "render_gpu_device_id"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if args.task_suite not in TASK_MAX_STEPS:
        raise ValueError(f"task_suite must be one of {sorted(TASK_MAX_STEPS)}, got {args.task_suite!r}")
    _validate_runner_python(args)


def _resolve_expected_profile(args: argparse.Namespace) -> LiberoCheckpointProfile:
    return resolve_checkpoint_profile(
        args.checkpoint_path,
        profile_path=args.policy_profile_path,
        target_adapter_path=args.target_adapter_path,
        resolved_config_path=args.config_file,
        weights_variant=args.weights_variant,
    )


def run_job(args: argparse.Namespace) -> int:
    """Run one checkpoint evaluation and return zero only after runner success."""
    _validate_job_args(args)
    run_dir = artifacts.validate_run_dir(args.output_dir, create=True)
    server_log_path = run_dir / SERVER_LOG_FILENAME
    runner_log_path = run_dir / RUNNER_LOG_FILENAME
    job_log_path = run_dir / JOB_LOG_FILENAME
    server_process: subprocess.Popen[Any] | None = None
    runner_process: subprocess.Popen[Any] | None = None

    with job_log_path.open("a", encoding="utf-8") as job_log:
        _set_shared_log_mode(job_log)
        _log(job_log, f"starting single-checkpoint LIBERO job in {run_dir}")
        try:
            expected_profile = _resolve_expected_profile(args)
            if expected_profile.checkpoint_format == CheckpointFormat.DCP and args.config_file is None:
                raise LiberoJobError("DCP evaluation requires an explicit --config-file for server parity")
            _log(
                job_log,
                f"resolved profile hash={expected_profile.profile_hash} "
                f"fingerprint={expected_profile.checkpoint_fingerprint} "
                f"role={expected_profile.checkpoint_role.value} weights={expected_profile.weights_variant.value}",
            )
            _ensure_port_available(args.server_port)
            server_url = f"http://127.0.0.1:{args.server_port}"
            server_command = build_server_command(args, run_dir)
            _log(job_log, f"server command: {shlex.join(server_command)}")

            with server_log_path.open("a", encoding="utf-8") as server_log, _sigterm_as_interrupt():
                _set_shared_log_mode(server_log)
                server_process = subprocess.Popen(
                    server_command,
                    cwd=_repository_root(),
                    env=_subprocess_environment(),
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                try:
                    handshake = wait_for_policy_server(
                        server_url,
                        server_process,
                        expected_profile,
                        startup_timeout_s=args.server_startup_timeout,
                        request_timeout_s=args.server_info_timeout,
                        poll_interval_s=args.server_poll_interval,
                        server_log_path=server_log_path,
                    )
                    _log(
                        job_log,
                        f"validated server handshake protocol={handshake.server_info['protocol_version']} "
                        f"profile_hash={handshake.profile.profile_hash} "
                        f"fingerprint={handshake.profile.checkpoint_fingerprint}",
                    )

                    runner_command = build_runner_command(args, run_dir, server_url)
                    _log(job_log, f"runner command: {shlex.join(runner_command)}")
                    with runner_log_path.open("a", encoding="utf-8") as runner_log:
                        _set_shared_log_mode(runner_log)
                        runner_process = subprocess.Popen(
                            runner_command,
                            cwd=_repository_root(),
                            env=_subprocess_environment(),
                            stdout=runner_log,
                            stderr=subprocess.STDOUT,
                            text=True,
                            start_new_session=True,
                        )
                        runner_returncode = runner_process.wait()
                    if runner_returncode != 0:
                        raise LiberoProcessError("LIBERO runner", runner_returncode, runner_log_path)
                    _validate_runner_metrics(run_dir, expected_profile)
                    _log(job_log, "LIBERO runner completed successfully")
                    return 0
                finally:
                    active_error = sys.exc_info()[0] is not None
                    _cleanup_processes(
                        (
                            ("LIBERO runner", runner_process),
                            ("policy server", server_process),
                        ),
                        timeout_s=args.server_stop_timeout,
                        job_log=job_log,
                        preserve_active_error=active_error,
                    )
                    _log(job_log, "managed subprocess cleanup completed")
        except BaseException as error:
            _log(job_log, f"job failed: {type(error).__name__}: {error}")
            traceback.print_exc(file=job_log)
            job_log.flush()
            raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint = parser.add_argument_group("checkpoint and policy server")
    checkpoint.add_argument("--checkpoint-path", type=Path, required=True)
    checkpoint.add_argument("--config-file", type=Path)
    checkpoint.add_argument("--policy-profile-path", type=Path)
    checkpoint.add_argument("--target-adapter-path", type=Path)
    checkpoint.add_argument("--weights-variant", choices=["ema", "regular"])
    checkpoint.add_argument("--server-port", type=int, default=8000)
    checkpoint.add_argument("--server-startup-timeout", type=float, default=900.0)
    checkpoint.add_argument("--server-info-timeout", type=float, default=5.0)
    checkpoint.add_argument("--server-poll-interval", type=float, default=1.0)
    checkpoint.add_argument("--server-stop-timeout", type=float, default=20.0)

    sampling = parser.add_argument_group("policy sampling")
    sampling.add_argument("--num-steps", type=int, default=8)
    sampling.add_argument("--guidance", type=float, default=1.0)

    evaluation = parser.add_argument_group("LIBERO runner")
    evaluation.add_argument("--runner-python", help="Python executable for the LIBERO runner environment")
    evaluation.add_argument("--task-suite", choices=sorted(TASK_MAX_STEPS), default="libero_spatial")
    evaluation.add_argument("--task-ids", default="", help="Comma-separated IDs; empty selects the full suite")
    evaluation.add_argument("--trials", type=int, default=10)
    evaluation.add_argument("--num-envs", type=int, default=1)
    evaluation.add_argument("--action-horizon", type=int, default=EDGE_ACTION_CHUNK_SIZE)
    evaluation.add_argument("--seed", type=int, default=0)
    evaluation.add_argument("--max-steps", type=int, default=0)
    evaluation.add_argument("--warmup-steps", type=int, default=10)
    evaluation.add_argument("--mujoco-gl", choices=["egl", "osmesa", "glfw"], default="egl")
    evaluation.add_argument("--render-gpu-device-id", type=int, default=0)
    evaluation.add_argument("--request-timeout", type=float, default=120.0)
    evaluation.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=f"Absolute canonical run directory below {artifacts.OUTPUT_ROOT}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    try:
        returncode = run_job(args)
    except LiberoProcessError as error:
        print(f"LIBERO job failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(error.returncode if 1 <= error.returncode <= 255 else 1) from error
    except BaseException as error:
        print(f"LIBERO job failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
