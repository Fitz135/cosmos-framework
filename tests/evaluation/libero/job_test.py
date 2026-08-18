# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import argparse
import io
import json
import stat
import subprocess
from pathlib import Path
from typing import Any, TextIO

import pytest
import requests

from cosmos_framework.evaluation.libero import job
from cosmos_framework.evaluation.libero import runner as libero_runner
from cosmos_framework.evaluation.libero.schema import LiberoCheckpointProfile

pytestmark = pytest.mark.level(0)


def _profile(checkpoint_path: Path, **overrides: Any) -> LiberoCheckpointProfile:
    values: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_fingerprint": "d" * 64,
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


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class _FakeProcess:
    def __init__(
        self,
        *,
        completion_returncode: int | None = None,
        initial_returncode: int | None = None,
        timeout_after_terminate: bool = False,
        wait_error: BaseException | None = None,
        wait_callback: Any | None = None,
        pid: int | None = None,
    ) -> None:
        self.completion_returncode = completion_returncode
        self.returncode = initial_returncode
        self.timeout_after_terminate = timeout_after_terminate
        self.wait_error = wait_error
        self.wait_callback = wait_callback
        self.terminated = False
        self.killed = False
        self.wait_timeouts: list[float | None] = []
        if pid is not None:
            self.pid = pid

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self.wait_callback is not None and not self.terminated and not self.killed:
            self.wait_callback()
        if self.wait_error is not None and not self.terminated and not self.killed:
            raise self.wait_error
        if self.timeout_after_terminate and self.terminated and not self.killed:
            raise subprocess.TimeoutExpired("fake", timeout)
        if self.returncode is None:
            if self.killed:
                self.returncode = -9
            elif self.terminated:
                self.returncode = -15
            elif self.completion_returncode is not None:
                self.returncode = self.completion_returncode
            else:
                raise AssertionError("live fake process cannot finish without termination")
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class _PopenFactory:
    def __init__(
        self,
        *,
        server: _FakeProcess | None = None,
        runner: _FakeProcess | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.server = server or _FakeProcess()
        self.runner = runner or _FakeProcess(completion_returncode=0)
        self.metrics = metrics or {
            "profile_hash": "b" * 64,
            "overall": {
                "evaluated_episodes": 1,
                "infra_errors": 0,
            },
            "infrastructure_errors": {
                "count": 2,
                "by_suite": {"libero_goal": 2},
                "by_type": {"transient": 2},
            },
        }
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> _FakeProcess:
        self.calls.append((command, kwargs))
        stream: TextIO = kwargs["stdout"]
        stream.write(f"mock spawn: {' '.join(command)}\n")
        stream.flush()
        if job.SERVER_MODULE in command:
            return self.server
        if job.RUNNER_MODULE in command:
            run_dir = Path(command[command.index("--output-dir") + 1])
            metrics_path = run_dir / libero_runner.METRICS_FILENAME
            metrics_path.write_text(json.dumps(self.metrics), encoding="utf-8")
            return self.runner
        raise AssertionError(f"unexpected command: {command}")


def _args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[argparse.Namespace, Path]:
    output_root = tmp_path / "canonical-output"
    output_root.mkdir()
    monkeypatch.setattr(job.artifacts, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(job, "_ensure_port_available", lambda _port: None)
    checkpoint = tmp_path / "checkpoint"
    config = tmp_path / "config.yaml"
    policy_profile = tmp_path / "profile.json"
    target_adapter = tmp_path / "adapter.json"
    run_dir = output_root / "libero" / "run-001"
    runner_python = tmp_path / "libero-python"
    runner_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner_python.chmod(0o755)
    args = job.build_arg_parser().parse_args(
        [
            "--checkpoint-path",
            str(checkpoint),
            "--config-file",
            str(config),
            "--policy-profile-path",
            str(policy_profile),
            "--target-adapter-path",
            str(target_adapter),
            "--weights-variant",
            "ema",
            "--server-port",
            "8123",
            "--server-startup-timeout",
            "1",
            "--server-info-timeout",
            "0.2",
            "--server-poll-interval",
            "0.001",
            "--server-stop-timeout",
            "0.01",
            "--num-steps",
            "6",
            "--guidance",
            "1.25",
            "--runner-python",
            str(runner_python),
            "--task-suite",
            "libero_goal",
            "--task-ids",
            "1,3",
            "--trials",
            "3",
            "--num-envs",
            "2",
            "--action-horizon",
            "4",
            "--seed",
            "5",
            "--max-steps",
            "6",
            "--warmup-steps",
            "7",
            "--mujoco-gl",
            "osmesa",
            "--render-gpu-device-id",
            "2",
            "--request-timeout",
            "9.5",
            "--output-dir",
            str(run_dir),
        ]
    )
    return args, checkpoint


def _info(profile: LiberoCheckpointProfile) -> dict[str, Any]:
    return {
        "protocol_version": "cosmos-libero-eval-v1",
        "policy_profile": profile.model_dump(mode="json"),
    }


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_job_polls_handshake_forwards_commands_logs_and_stops_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, checkpoint = _args(tmp_path, monkeypatch)
    args.output_dir.mkdir(parents=True)
    for filename in (job.JOB_LOG_FILENAME, job.SERVER_LOG_FILENAME, job.RUNNER_LOG_FILENAME):
        path = args.output_dir / filename
        path.touch()
        path.chmod(0o600)
    profile = _profile(checkpoint)
    resolve_calls: list[tuple[Path, dict[str, Any]]] = []

    def _resolve(path: Path, **kwargs: Any) -> LiberoCheckpointProfile:
        resolve_calls.append((path, kwargs))
        return profile

    monkeypatch.setattr(job, "resolve_checkpoint_profile", _resolve)
    http_calls: list[tuple[str, float]] = []

    def _get(url: str, *, timeout: float) -> _FakeResponse:
        http_calls.append((url, timeout))
        if len(http_calls) == 1:
            raise requests.ConnectionError("server is still loading")
        return _FakeResponse(_info(profile))

    monkeypatch.setattr(job.requests, "get", _get)
    factory = _PopenFactory()
    monkeypatch.setattr(job.subprocess, "Popen", factory)

    assert job.run_job(args) == 0

    assert len(resolve_calls) == 1
    assert resolve_calls[0][0] == checkpoint
    assert resolve_calls[0][1] == {
        "profile_path": args.policy_profile_path,
        "target_adapter_path": args.target_adapter_path,
        "resolved_config_path": args.config_file,
        "weights_variant": "ema",
    }
    assert http_calls == [
        ("http://127.0.0.1:8123/info", 0.2),
        ("http://127.0.0.1:8123/info", 0.2),
    ]
    assert len(factory.calls) == 2
    server_command, server_kwargs = factory.calls[0]
    runner_command, runner_kwargs = factory.calls[1]
    assert server_command[:3] == [job.sys.executable, "-m", job.SERVER_MODULE]
    assert _flag(server_command, "--checkpoint-path") == str(checkpoint)
    assert _flag(server_command, "--config-file") == str(args.config_file)
    assert _flag(server_command, "--policy-profile-path") == str(args.policy_profile_path)
    assert _flag(server_command, "--target-adapter-path") == str(args.target_adapter_path)
    assert _flag(server_command, "--weights-variant") == "ema"
    assert _flag(server_command, "--port") == "8123"
    assert _flag(server_command, "--num-steps") == "6"
    assert _flag(server_command, "--guidance") == "1.25"
    assert runner_command[:3] == [str(runner_python := Path(args.runner_python)), "-m", job.RUNNER_MODULE]
    assert runner_python.is_absolute() and runner_python.stat().st_mode & 0o111
    assert _flag(runner_command, "--server-url") == "http://127.0.0.1:8123"
    assert _flag(runner_command, "--output-dir") == str(args.output_dir)
    assert _flag(runner_command, "--task-suite") == "libero_goal"
    assert _flag(runner_command, "--task-ids") == "1,3"
    assert _flag(runner_command, "--trials") == "3"
    assert _flag(runner_command, "--num-envs") == "2"
    assert _flag(runner_command, "--action-horizon") == "4"
    assert _flag(runner_command, "--seed") == "5"
    assert _flag(runner_command, "--max-steps") == "6"
    assert _flag(runner_command, "--warmup-steps") == "7"
    assert _flag(runner_command, "--mujoco-gl") == "osmesa"
    assert _flag(runner_command, "--render-gpu-device-id") == "2"
    assert _flag(runner_command, "--request-timeout") == "9.5"
    assert server_kwargs["stderr"] is subprocess.STDOUT
    assert server_kwargs["start_new_session"] is True
    assert runner_kwargs["stderr"] is subprocess.STDOUT
    assert runner_kwargs["start_new_session"] is True
    assert factory.server.terminated and not factory.server.killed
    assert not factory.runner.terminated

    run_dir = args.output_dir
    assert (run_dir / job.SERVER_LOG_FILENAME).is_file()
    assert (run_dir / job.RUNNER_LOG_FILENAME).is_file()
    for filename in (job.JOB_LOG_FILENAME, job.SERVER_LOG_FILENAME, job.RUNNER_LOG_FILENAME):
        assert stat.S_IMODE((run_dir / filename).stat().st_mode) == 0o644
    job_log = (run_dir / job.JOB_LOG_FILENAME).read_text()
    assert "validated server handshake" in job_log
    assert "LIBERO runner completed successfully" in job_log


def test_mismatched_handshake_fails_before_runner_and_preserves_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, checkpoint = _args(tmp_path, monkeypatch)
    expected = _profile(checkpoint)
    served = _profile(checkpoint, profile_hash="c" * 64)
    monkeypatch.setattr(job, "resolve_checkpoint_profile", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(job.requests, "get", lambda *_args, **_kwargs: _FakeResponse(_info(served)))
    factory = _PopenFactory()
    monkeypatch.setattr(job.subprocess, "Popen", factory)

    with pytest.raises(job.LiberoJobError, match="handshake does not match"):
        job.run_job(args)

    assert len(factory.calls) == 1
    assert factory.server.terminated
    assert (args.output_dir / job.SERVER_LOG_FILENAME).is_file()
    assert "job failed: LiberoJobError" in (args.output_dir / job.JOB_LOG_FILENAME).read_text()


def test_runner_nonzero_is_propagated_and_server_is_stopped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, checkpoint = _args(tmp_path, monkeypatch)
    profile = _profile(checkpoint)
    monkeypatch.setattr(job, "resolve_checkpoint_profile", lambda *_args, **_kwargs: profile)
    monkeypatch.setattr(job.requests, "get", lambda *_args, **_kwargs: _FakeResponse(_info(profile)))
    factory = _PopenFactory(runner=_FakeProcess(completion_returncode=7))
    monkeypatch.setattr(job.subprocess, "Popen", factory)

    with pytest.raises(job.LiberoProcessError) as raised:
        job.run_job(args)

    assert raised.value.phase == "LIBERO runner"
    assert raised.value.returncode == 7
    assert raised.value.log_path == args.output_dir / job.RUNNER_LOG_FILENAME
    assert factory.server.terminated
    assert (args.output_dir / job.RUNNER_LOG_FILENAME).is_file()
    assert "exited with status 7" in (args.output_dir / job.JOB_LOG_FILENAME).read_text()


def test_server_early_exit_is_nonzero_and_skips_http_and_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, checkpoint = _args(tmp_path, monkeypatch)
    profile = _profile(checkpoint)
    monkeypatch.setattr(job, "resolve_checkpoint_profile", lambda *_args, **_kwargs: profile)
    http_called = False

    def _unexpected_http(*_args: Any, **_kwargs: Any) -> _FakeResponse:
        nonlocal http_called
        http_called = True
        raise AssertionError("HTTP must not be called after server exit")

    monkeypatch.setattr(job.requests, "get", _unexpected_http)
    factory = _PopenFactory(server=_FakeProcess(initial_returncode=3))
    monkeypatch.setattr(job.subprocess, "Popen", factory)

    with pytest.raises(job.LiberoProcessError) as raised:
        job.run_job(args)

    assert raised.value.phase == "policy server"
    assert raised.value.returncode == 3
    assert not http_called
    assert len(factory.calls) == 1


def test_termination_escalates_whole_process_group_to_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(timeout_after_terminate=True, pid=4242)
    signals: list[tuple[int, int]] = []

    def _killpg(pid: int, signal_number: int) -> None:
        signals.append((pid, signal_number))
        if signal_number == job.signal.SIGTERM:
            process.terminated = True
        else:
            process.killed = True

    monkeypatch.setattr(job.os, "killpg", _killpg)

    job._terminate_process(process, timeout_s=0.01)

    assert signals == [(4242, job.signal.SIGTERM), (4242, job.signal.SIGKILL)]
    assert process.terminated
    assert process.killed
    assert process.returncode == -9


def test_runner_keyboard_interrupt_still_stops_both_process_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, checkpoint = _args(tmp_path, monkeypatch)
    profile = _profile(checkpoint)
    monkeypatch.setattr(job, "resolve_checkpoint_profile", lambda *_args, **_kwargs: profile)
    monkeypatch.setattr(job.requests, "get", lambda *_args, **_kwargs: _FakeResponse(_info(profile)))
    factory = _PopenFactory(runner=_FakeProcess(wait_error=KeyboardInterrupt()))
    monkeypatch.setattr(job.subprocess, "Popen", factory)

    with pytest.raises(KeyboardInterrupt):
        job.run_job(args)

    assert factory.runner.terminated
    assert factory.server.terminated
    assert "job failed: KeyboardInterrupt" in (args.output_dir / job.JOB_LOG_FILENAME).read_text()


def test_orchestrator_sigterm_unwinds_and_stops_both_process_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, checkpoint = _args(tmp_path, monkeypatch)
    profile = _profile(checkpoint)
    monkeypatch.setattr(job, "resolve_checkpoint_profile", lambda *_args, **_kwargs: profile)
    monkeypatch.setattr(job.requests, "get", lambda *_args, **_kwargs: _FakeResponse(_info(profile)))
    installed_handlers: list[Any] = []
    previous_handler = object()
    monkeypatch.setattr(job.signal, "getsignal", lambda _signal_number: previous_handler)

    def _install_handler(_signal_number: int, handler: Any) -> Any:
        installed_handlers.append(handler)
        return previous_handler

    monkeypatch.setattr(job.signal, "signal", _install_handler)

    def _send_sigterm() -> None:
        installed_handlers[-1](job.signal.SIGTERM, None)

    factory = _PopenFactory(runner=_FakeProcess(wait_callback=_send_sigterm))
    monkeypatch.setattr(job.subprocess, "Popen", factory)

    with pytest.raises(job.LiberoJobInterrupted):
        job.run_job(args)

    assert factory.runner.terminated
    assert factory.server.terminated
    assert installed_handlers[-1] is previous_handler


def test_runner_spawn_error_still_stops_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, checkpoint = _args(tmp_path, monkeypatch)
    profile = _profile(checkpoint)
    monkeypatch.setattr(job, "resolve_checkpoint_profile", lambda *_args, **_kwargs: profile)
    monkeypatch.setattr(job.requests, "get", lambda *_args, **_kwargs: _FakeResponse(_info(profile)))
    factory = _PopenFactory()

    def _popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        if job.RUNNER_MODULE in command:
            raise OSError("runner executable failed to spawn")
        return factory(command, **kwargs)

    monkeypatch.setattr(job.subprocess, "Popen", _popen)

    with pytest.raises(OSError, match="failed to spawn"):
        job.run_job(args)

    assert factory.server.terminated
    assert (args.output_dir / job.RUNNER_LOG_FILENAME).is_file()


def test_server_spawn_error_is_logged_without_starting_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, checkpoint = _args(tmp_path, monkeypatch)
    profile = _profile(checkpoint)
    monkeypatch.setattr(job, "resolve_checkpoint_profile", lambda *_args, **_kwargs: profile)

    def _spawn_failure(_command: list[str], **_kwargs: Any) -> _FakeProcess:
        raise OSError("policy server failed to spawn")

    monkeypatch.setattr(job.subprocess, "Popen", _spawn_failure)

    with pytest.raises(OSError, match="failed to spawn"):
        job.run_job(args)

    assert (args.output_dir / job.SERVER_LOG_FILENAME).is_file()
    assert "job failed: OSError" in (args.output_dir / job.JOB_LOG_FILENAME).read_text()


def test_cleanup_attempts_every_group_without_masking_active_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _FakeProcess()
    server = _FakeProcess()
    attempted: list[_FakeProcess] = []

    def _failed_cleanup(process: _FakeProcess, *, timeout_s: float) -> None:
        assert timeout_s == 0.25
        attempted.append(process)
        raise RuntimeError("cleanup failure")

    monkeypatch.setattr(job, "_terminate_process", _failed_cleanup)
    log = io.StringIO()

    job._cleanup_processes(
        (("LIBERO runner", runner), ("policy server", server)),
        timeout_s=0.25,
        job_log=log,
        preserve_active_error=True,
    )

    assert attempted == [runner, server]
    assert log.getvalue().count("cleanup failure") == 2


def test_occupied_server_port_is_rejected() -> None:
    with job.socket.socket(job.socket.AF_INET, job.socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        with pytest.raises(job.LiberoJobError, match="already in use"):
            job._ensure_port_available(port)


def test_startup_timeout_reports_last_transport_error_and_server_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    profile = _profile(checkpoint)
    process = _FakeProcess()
    attempts = 0

    def _unavailable(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(job, "_fetch_server_info", _unavailable)
    log_path = tmp_path / "server.log"

    with pytest.raises(job.LiberoJobError, match="connection refused.*server.log"):
        job.wait_for_policy_server(
            "http://127.0.0.1:8123",
            process,
            profile,
            startup_timeout_s=0.002,
            request_timeout_s=1.0,
            poll_interval_s=0.001,
            server_log_path=log_path,
        )

    assert attempts >= 1


def test_checkpoint_inputs_are_resolved_before_server_cwd_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _checkpoint = _args(tmp_path, monkeypatch)
    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    monkeypatch.chdir(caller_dir)
    relative_inputs = {
        "checkpoint_path": Path("relative/checkpoint"),
        "config_file": Path("relative/config.yaml"),
        "policy_profile_path": Path("relative/profile.json"),
        "target_adapter_path": Path("relative/adapter.json"),
    }
    for name, value in relative_inputs.items():
        setattr(args, name, value)

    job._validate_job_args(args)
    server_command = job.build_server_command(args, args.output_dir)

    for name, value in relative_inputs.items():
        assert getattr(args, name) == (caller_dir / value).resolve()
    assert _flag(server_command, "--checkpoint-path") == str(args.checkpoint_path)
    assert _flag(server_command, "--config-file") == str(args.config_file)
    assert _flag(server_command, "--policy-profile-path") == str(args.policy_profile_path)
    assert _flag(server_command, "--target-adapter-path") == str(args.target_adapter_path)


def test_generated_runner_command_parses_with_runner_cli_and_optional_server_flags_are_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _checkpoint = _args(tmp_path, monkeypatch)
    args.config_file = None
    args.policy_profile_path = None
    args.target_adapter_path = None
    args.weights_variant = None
    args.task_ids = ""
    home = tmp_path / "home"
    runner_python = home / "libero-env" / "bin" / "python"
    runner_python.parent.mkdir(parents=True)
    runner_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner_python.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    args.runner_python = "~/libero-env/bin/python"
    job._validate_job_args(args)

    server_command = job.build_server_command(args, args.output_dir)
    runner_command = job.build_runner_command(args, args.output_dir, "http://127.0.0.1:8123")
    parsed = libero_runner.build_arg_parser().parse_args(runner_command[3:])

    for flag in ("--config-file", "--policy-profile-path", "--target-adapter-path", "--weights-variant"):
        assert flag not in server_command
    assert runner_command[0] == str(runner_python.resolve())
    assert parsed.server_url == "http://127.0.0.1:8123"
    assert parsed.output_dir == str(args.output_dir)
    assert parsed.task_ids == ""
    assert parsed.task_suite == args.task_suite
    assert parsed.action_horizon == args.action_horizon


def test_runner_python_preserves_virtualenv_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _checkpoint = _args(tmp_path, monkeypatch)
    interpreter = tmp_path / "runtime" / "python3.13"
    interpreter.parent.mkdir()
    interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    interpreter.chmod(0o755)
    runner_python = tmp_path / "libero-env" / "bin" / "python"
    runner_python.parent.mkdir(parents=True)
    runner_python.symlink_to(interpreter)
    args.runner_python = str(runner_python)

    job._validate_job_args(args)
    runner_command = job.build_runner_command(args, args.output_dir, "http://127.0.0.1:8123")

    assert runner_command[0] == str(runner_python.absolute())
    assert Path(runner_command[0]).is_symlink()
    assert Path(runner_command[0]).resolve() == interpreter.resolve()


def test_relative_runner_python_becomes_absolute_without_dereferencing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _checkpoint = _args(tmp_path, monkeypatch)
    caller_dir = tmp_path / "caller"
    runner_python = caller_dir / "libero-env" / "bin" / "python"
    runner_python.parent.mkdir(parents=True)
    runner_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner_python.chmod(0o755)
    monkeypatch.chdir(caller_dir)
    args.runner_python = "libero-env/bin/python"

    job._validate_job_args(args)

    assert args.runner_python == str(runner_python.absolute())


@pytest.mark.parametrize("invalid_kind", ["non_executable", "directory"])
def test_runner_python_rejects_non_executable_files_and_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_kind: str,
) -> None:
    args, _checkpoint = _args(tmp_path, monkeypatch)
    candidate = tmp_path / invalid_kind
    if invalid_kind == "directory":
        candidate.mkdir()
        candidate.chmod(0o755)
    else:
        candidate.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        candidate.chmod(0o644)
    args.runner_python = str(candidate)

    with pytest.raises(ValueError, match="runner_python is not an executable file"):
        job._validate_job_args(args)


def test_dcp_requires_explicit_config_before_server_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, checkpoint = _args(tmp_path, monkeypatch)
    args.config_file = None
    profile = _profile(checkpoint, checkpoint_format="dcp")
    monkeypatch.setattr(job, "resolve_checkpoint_profile", lambda *_args, **_kwargs: profile)
    factory = _PopenFactory()
    monkeypatch.setattr(job.subprocess, "Popen", factory)

    with pytest.raises(job.LiberoJobError, match="DCP evaluation requires an explicit --config-file"):
        job.run_job(args)

    assert factory.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("server_info_timeout", float("nan")),
        ("server_stop_timeout", 0.0),
        ("num_steps", 0),
        ("guidance", float("nan")),
        ("guidance", 0.0),
        ("action_horizon", 9),
        ("action_horizon", 1.5),
        ("task_suite", "libero_unknown"),
        ("runner_python", "python"),
    ],
)
def test_invalid_job_args_fail_before_profile_or_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    args, _checkpoint = _args(tmp_path, monkeypatch)
    setattr(args, field, value)
    profile_called = False

    def _unexpected_profile(*_args: Any, **_kwargs: Any) -> LiberoCheckpointProfile:
        nonlocal profile_called
        profile_called = True
        raise AssertionError("profile resolution must not run")

    monkeypatch.setattr(job, "resolve_checkpoint_profile", _unexpected_profile)

    with pytest.raises(ValueError, match=field):
        job.run_job(args)

    assert not profile_called


def test_runner_metrics_accept_historical_infra_attempts_after_clean_resume(tmp_path: Path) -> None:
    profile = _profile(tmp_path / "checkpoint")
    metrics = {
        "profile_hash": profile.profile_hash,
        "overall": {"evaluated_episodes": 2, "infra_errors": 0},
        "infrastructure_errors": {"count": 4},
    }
    (tmp_path / libero_runner.METRICS_FILENAME).write_text(json.dumps(metrics), encoding="utf-8")

    assert job._validate_runner_metrics(tmp_path, profile) == metrics


@pytest.mark.parametrize(
    "metrics",
    [
        {
            "profile_hash": "b" * 64,
            "overall": {"evaluated_episodes": 0, "infra_errors": 0},
            "infrastructure_errors": {"count": 0},
        },
        {
            "profile_hash": "b" * 64,
            "overall": {"evaluated_episodes": 2, "infra_errors": 1},
            "infrastructure_errors": {"count": 4},
        },
    ],
)
def test_runner_zero_exit_still_rejects_no_clean_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metrics: dict[str, Any],
) -> None:
    args, checkpoint = _args(tmp_path, monkeypatch)
    profile = _profile(checkpoint)
    monkeypatch.setattr(job, "resolve_checkpoint_profile", lambda *_args, **_kwargs: profile)
    monkeypatch.setattr(job.requests, "get", lambda *_args, **_kwargs: _FakeResponse(_info(profile)))
    factory = _PopenFactory(metrics=metrics)
    monkeypatch.setattr(job.subprocess, "Popen", factory)

    with pytest.raises(job.LiberoJobError, match="no clean successful evaluation result"):
        job.run_job(args)

    assert factory.runner.returncode == 0
    assert factory.server.terminated


def test_main_preserves_runner_exit_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    error = job.LiberoProcessError("LIBERO runner", 23, tmp_path / "runner.log")
    monkeypatch.setattr(job, "run_job", lambda _args: (_ for _ in ()).throw(error))

    with pytest.raises(SystemExit) as raised:
        job.main(["--checkpoint-path", str(tmp_path / "checkpoint"), "--output-dir", str(tmp_path / "run")])

    assert raised.value.code == 23
