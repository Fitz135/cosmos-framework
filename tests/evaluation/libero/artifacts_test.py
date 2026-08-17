# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmos_framework.evaluation.libero import artifacts


@pytest.fixture
def output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr(artifacts, "OUTPUT_ROOT", root)
    return root


def test_output_path_validation_rejects_root_relative_and_escape(output_root: Path, tmp_path: Path) -> None:
    run_dir = output_root / "cosmos-framework-eval" / "run-1"
    assert artifacts.validate_run_dir(run_dir, create=True) == run_dir

    with pytest.raises(artifacts.OutputPathError):
        artifacts.validate_run_dir(output_root)
    with pytest.raises(artifacts.OutputPathError):
        artifacts.validate_run_dir("relative/run")
    with pytest.raises(artifacts.OutputPathError):
        artifacts.validate_run_dir(tmp_path / "outside")
    with pytest.raises(artifacts.OutputPathError):
        artifacts.validate_output_root(tmp_path)


def test_output_path_validation_rejects_symlink_escape(output_root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    escape = output_root / "escape"
    escape.symlink_to(outside, target_is_directory=True)

    with pytest.raises(artifacts.OutputPathError):
        artifacts.validate_run_dir(escape / "run", create=True)


def test_manifest_is_json_and_atomic(output_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = output_root / "eval" / "run"
    replacements: list[tuple[Path, Path]] = []
    real_replace = artifacts.os.replace

    def tracked_replace(source: str | Path, destination: str | Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", tracked_replace)
    manifest_path = artifacts.write_manifest(run_dir, {"seed": 42, "checkpoint": "edge-10k"})

    assert artifacts.read_manifest(run_dir) == {"seed": 42, "checkpoint": "edge-10k"}
    assert manifest_path.stat().st_mode & 0o777 == 0o644
    assert replacements and replacements[-1][1] == manifest_path
    assert replacements[-1][0].parent == run_dir
    assert not list(run_dir.glob(".manifest.json.*.tmp"))


def test_episode_journal_is_incremental_and_resume_deduplicates(output_root: Path) -> None:
    run_dir = output_root / "eval" / "run"
    first = {"episode_id": "libero_goal:0:0", "task_suite": "libero_goal", "task_id": 0, "success": True}
    second = {"episode_id": "libero_goal:0:1", "task_suite": "libero_goal", "task_id": 0, "success": False}

    assert artifacts.append_episode(run_dir, first)
    assert artifacts.append_episode(run_dir, second)
    assert not artifacts.append_episode(run_dir, first)
    assert artifacts.completed_episode_ids(run_dir) == {first["episode_id"], second["episode_id"]}
    assert artifacts.read_episodes(run_dir) == [first, second]
    assert len((run_dir / artifacts.EPISODES_FILENAME).read_text().splitlines()) == 2
    assert (run_dir / artifacts.EPISODES_FILENAME).stat().st_mode & 0o777 == 0o644

    with pytest.raises(artifacts.DuplicateEpisodeError):
        artifacts.append_episode(run_dir, {**first, "success": False})

    with pytest.raises(artifacts.ArtifactFormatError, match="terminal outcomes"):
        artifacts.append_episode(
            run_dir,
            {"episode_id": "infra", "task_suite": "libero_goal", "task_id": 0, "infra_error": True},
        )


def test_infra_journal_is_separate_retryable_and_keyed_by_attempt(output_root: Path) -> None:
    run_dir = output_root / "eval" / "run"
    record = {
        "attempt_id": "invocation-1:0000",
        "episode_id": "libero_goal:0:0",
        "task_suite": "libero_goal",
        "task_id": 0,
        "infra_error": True,
        "infra_error_type": "policy_server",
    }

    assert artifacts.append_infra_error(run_dir, record)
    assert not artifacts.append_infra_error(run_dir, record)
    assert artifacts.read_infra_errors(run_dir) == [record]
    assert (run_dir / artifacts.INFRA_ERRORS_FILENAME).stat().st_mode & 0o777 == 0o644
    assert artifacts.completed_episode_ids(run_dir) == set()
    assert artifacts.read_episodes(run_dir) == []

    with pytest.raises(artifacts.DuplicateEpisodeError):
        artifacts.append_infra_error(run_dir, {**record, "infra_error_type": "environment"})
    with pytest.raises(artifacts.ArtifactFormatError, match="attempt_id"):
        artifacts.append_infra_error(run_dir, {key: value for key, value in record.items() if key != "attempt_id"})


def test_episode_journal_repairs_only_a_partial_trailing_record(output_root: Path) -> None:
    run_dir = output_root / "eval" / "run"
    first = {"episode_id": "one", "task_suite": "suite", "task_id": 0, "success": True}
    second = {"episode_id": "two", "task_suite": "suite", "task_id": 0, "success": False}
    assert artifacts.append_episode(run_dir, first)
    with (run_dir / artifacts.EPISODES_FILENAME).open("ab") as stream:
        stream.write(b'{"episode_id":"partial"')

    assert artifacts.append_episode(run_dir, second)
    assert artifacts.read_episodes(run_dir) == [first, second]

    with (run_dir / artifacts.EPISODES_FILENAME).open("ab") as stream:
        stream.write(b"not-json\n")
    with pytest.raises(artifacts.ArtifactFormatError):
        artifacts.read_episodes(run_dir)


def test_completion_marker_is_atomic_idempotent_and_seals_journal(output_root: Path) -> None:
    run_dir = output_root / "eval" / "run"
    artifacts.write_manifest(run_dir, {"checkpoint": "edge-10k"})
    artifacts.append_episode(
        run_dir,
        {"episode_id": "one", "task_suite": "libero_10", "task_id": 0, "success": True},
    )

    marker_path = artifacts.mark_complete(
        run_dir,
        metadata={"summary": "summary.json"},
        completed_at="2026-08-18T00:00:00+00:00",
    )
    assert artifacts.is_complete(run_dir)
    assert artifacts.read_completion_marker(run_dir) == {
        "completed_at": "2026-08-18T00:00:00+00:00",
        "episode_count": 1,
        "summary": "summary.json",
    }
    assert artifacts.mark_complete(run_dir, completed_at="different") == marker_path
    assert json.loads(marker_path.read_text())["completed_at"] == "2026-08-18T00:00:00+00:00"
    assert marker_path.stat().st_mode & 0o777 == 0o644

    with pytest.raises(artifacts.RunCompletedError):
        artifacts.append_episode(
            run_dir,
            {"episode_id": "two", "task_suite": "libero_10", "task_id": 0, "success": False},
        )


def test_completion_requires_manifest(output_root: Path) -> None:
    run_dir = output_root / "eval" / "run"
    run_dir.mkdir(parents=True)
    with pytest.raises(artifacts.ArtifactError):
        artifacts.mark_complete(run_dir)
