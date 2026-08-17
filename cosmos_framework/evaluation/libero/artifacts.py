# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Durable, resumable artifacts for LIBERO evaluation runs.

The helpers in this module deliberately accept only per-run directories below
the cluster's canonical experiment-output root.  Episode records are appended
one JSON object per line and keyed by a caller-provided, stable ``episode_id``.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator

OUTPUT_ROOT = Path("/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output")
MANIFEST_FILENAME = "manifest.json"
EPISODES_FILENAME = "episodes.jsonl"
INFRA_ERRORS_FILENAME = "infra_errors.jsonl"
COMPLETION_FILENAME = "_SUCCESS"
_LOCK_FILENAME = ".episodes.lock"


class ArtifactError(RuntimeError):
    """Base exception for evaluation-artifact failures."""


class OutputPathError(ArtifactError, ValueError):
    """Raised when a path escapes the canonical experiment-output root."""


class ArtifactFormatError(ArtifactError, ValueError):
    """Raised when an existing artifact is malformed or internally inconsistent."""


class DuplicateEpisodeError(ArtifactError):
    """Raised when one episode ID is associated with conflicting records."""


class RunCompletedError(ArtifactError):
    """Raised when attempting to mutate an already-completed run."""


def _absolute_normalized(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise OutputPathError(f"Evaluation artifact paths must be absolute, got {candidate!s}")
    return Path(os.path.abspath(candidate))


def validate_output_root(output_root: str | os.PathLike[str] = OUTPUT_ROOT) -> Path:
    """Validate and return the one allowed experiment-output root.

    Both the literal absolute path and its resolved target must match.  This
    rejects aliases through symlinks as well as similarly named sibling roots.
    """

    expected = _absolute_normalized(OUTPUT_ROOT)
    candidate = _absolute_normalized(output_root)
    if candidate != expected:
        raise OutputPathError(f"Output root must be exactly {expected}, got {candidate}")
    if not expected.is_dir():
        raise OutputPathError(f"Canonical output root does not exist or is not a directory: {expected}")
    if candidate.resolve(strict=True) != expected.resolve(strict=True):
        raise OutputPathError(f"Output root resolves outside the canonical location: {candidate}")
    return expected.resolve(strict=True)


def validate_run_dir(run_dir: str | os.PathLike[str], *, create: bool = False) -> Path:
    """Validate a per-run directory below :data:`OUTPUT_ROOT`.

    The output root itself is intentionally rejected: callers must create a
    project/run hierarchy rather than placing artifacts directly at the root.
    Existing symlinks are resolved before containment is checked.
    """

    root_literal = _absolute_normalized(OUTPUT_ROOT)
    root_resolved = validate_output_root(root_literal)
    candidate_literal = _absolute_normalized(run_dir)
    try:
        literal_relative = candidate_literal.relative_to(root_literal)
    except ValueError as exc:
        raise OutputPathError(f"Run directory must be below {root_literal}, got {candidate_literal}") from exc
    if not literal_relative.parts:
        raise OutputPathError("The output root itself cannot be used as a run directory")

    candidate_resolved = candidate_literal.resolve(strict=False)
    try:
        resolved_relative = candidate_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise OutputPathError(f"Run directory resolves outside {root_resolved}: {candidate_literal}") from exc
    if not resolved_relative.parts:
        raise OutputPathError("The output root itself cannot be used as a run directory")

    if create:
        candidate_resolved.mkdir(parents=True, exist_ok=True)
    if candidate_resolved.exists() and not candidate_resolved.is_dir():
        raise OutputPathError(f"Run directory path exists but is not a directory: {candidate_resolved}")
    return candidate_resolved


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _jsonl_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(directory, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.fchmod(descriptor, 0o644)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_manifest(run_dir: str | os.PathLike[str], manifest: Mapping[str, Any]) -> Path:
    """Atomically create or replace ``manifest.json`` for a run."""

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    directory = validate_run_dir(run_dir, create=True)
    path = directory / MANIFEST_FILENAME
    _atomic_write(path, _json_bytes(dict(manifest)))
    return path


def read_manifest(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate a run manifest."""

    directory = validate_run_dir(run_dir)
    path = directory / MANIFEST_FILENAME
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ArtifactFormatError(f"Manifest must contain a JSON object: {path}")
    return manifest


def _episode_id(record: Mapping[str, Any], *, path: Path | None = None, line_number: int | None = None) -> str:
    value = record.get("episode_id")
    if not isinstance(value, str) or not value.strip():
        location = f" in {path}:{line_number}" if path is not None and line_number is not None else ""
        raise ArtifactFormatError(f"Episode record{location} must contain a non-empty string 'episode_id'")
    return value


def _attempt_id(record: Mapping[str, Any], *, path: Path | None = None, line_number: int | None = None) -> str:
    value = record.get("attempt_id")
    if not isinstance(value, str) or not value.strip():
        location = f" in {path}:{line_number}" if path is not None and line_number is not None else ""
        raise ArtifactFormatError(f"Infrastructure record{location} must contain a non-empty string 'attempt_id'")
    return value


def _validate_terminal_episode(record: Mapping[str, Any]) -> None:
    if not isinstance(record.get("success"), bool) or record.get("infra_error", False) is not False:
        raise ArtifactFormatError("Episode records must be terminal outcomes with boolean success and no infra_error")


def _validate_infra_attempt(record: Mapping[str, Any]) -> None:
    _attempt_id(record)
    _episode_id(record)
    if record.get("infra_error") is not True or "success" in record:
        raise ArtifactFormatError("Infrastructure records require infra_error=true and must not contain success")


def _scan_episodes(
    path: Path, *, repair_trailing_record: bool
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not path.exists():
        return [], {}

    mode = "r+b" if repair_trailing_record else "rb"
    records: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    with path.open(mode) as stream:
        line_number = 0
        while True:
            record_offset = stream.tell()
            raw_line = stream.readline()
            if not raw_line:
                break
            line_number += 1
            terminated = raw_line.endswith(b"\n")
            stripped = raw_line.strip()
            if not stripped:
                raise ArtifactFormatError(f"Blank JSONL record in {path}:{line_number}")
            try:
                value = json.loads(stripped.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if repair_trailing_record and not terminated and stream.read(1) == b"":
                    stream.seek(record_offset)
                    stream.truncate()
                    stream.flush()
                    os.fsync(stream.fileno())
                    break
                raise ArtifactFormatError(f"Malformed JSONL record in {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ArtifactFormatError(f"Episode record in {path}:{line_number} must be a JSON object")
            identity = _episode_id(value, path=path, line_number=line_number)
            existing = by_id.get(identity)
            if existing is not None:
                if existing != value:
                    raise DuplicateEpisodeError(f"Conflicting records for episode_id={identity!r} in {path}")
            else:
                by_id[identity] = value
                records.append(value)

            if repair_trailing_record and not terminated:
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
    return records, by_id


def _scan_infra_errors(
    path: Path, *, repair_trailing_record: bool
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not path.exists():
        return [], {}
    mode = "r+b" if repair_trailing_record else "rb"
    records: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    with path.open(mode) as stream:
        line_number = 0
        while True:
            record_offset = stream.tell()
            raw_line = stream.readline()
            if not raw_line:
                break
            line_number += 1
            terminated = raw_line.endswith(b"\n")
            stripped = raw_line.strip()
            if not stripped:
                raise ArtifactFormatError(f"Blank JSONL record in {path}:{line_number}")
            try:
                value = json.loads(stripped.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if repair_trailing_record and not terminated and stream.read(1) == b"":
                    stream.seek(record_offset)
                    stream.truncate()
                    stream.flush()
                    os.fsync(stream.fileno())
                    break
                raise ArtifactFormatError(f"Malformed JSONL record in {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ArtifactFormatError(f"Infrastructure record in {path}:{line_number} must be a JSON object")
            identity = _attempt_id(value, path=path, line_number=line_number)
            existing = by_id.get(identity)
            if existing is not None:
                if existing != value:
                    raise DuplicateEpisodeError(f"Conflicting records for attempt_id={identity!r} in {path}")
            else:
                by_id[identity] = value
                records.append(value)
            if repair_trailing_record and not terminated:
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
    return records, by_id


@contextmanager
def _journal_lock(run_dir: Path, *, exclusive: bool) -> Iterator[BinaryIO]:
    lock_path = run_dir / _LOCK_FILENAME
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield lock_stream
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def append_episode(run_dir: str | os.PathLike[str], record: Mapping[str, Any]) -> bool:
    """Durably append one episode unless its ``episode_id`` is already present.

    Returns ``True`` when a new line is appended and ``False`` for an identical
    replay during resume.  Reusing an ID with different data is an error.
    """

    if not isinstance(record, Mapping):
        raise TypeError("episode record must be a mapping")
    value = dict(record)
    identity = _episode_id(value)
    _validate_terminal_episode(value)
    encoded = _jsonl_bytes(value)
    directory = validate_run_dir(run_dir, create=True)
    journal_path = directory / EPISODES_FILENAME

    with _journal_lock(directory, exclusive=True):
        if (directory / COMPLETION_FILENAME).exists():
            raise RunCompletedError(f"Cannot append episode {identity!r}: run is already complete")
        _, by_id = _scan_episodes(journal_path, repair_trailing_record=True)
        existing = by_id.get(identity)
        if existing is not None:
            if existing != value:
                raise DuplicateEpisodeError(f"Conflicting record for episode_id={identity!r}")
            return False

        descriptor = os.open(journal_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        os.fchmod(descriptor, 0o644)
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError(f"Failed to append episode record to {journal_path}")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(directory)
    return True


def append_infra_error(run_dir: str | os.PathLike[str], record: Mapping[str, Any]) -> bool:
    """Durably append one retryable infrastructure attempt keyed by attempt_id."""
    if not isinstance(record, Mapping):
        raise TypeError("infrastructure record must be a mapping")
    value = dict(record)
    _validate_infra_attempt(value)
    identity = _attempt_id(value)
    encoded = _jsonl_bytes(value)
    directory = validate_run_dir(run_dir, create=True)
    journal_path = directory / INFRA_ERRORS_FILENAME

    with _journal_lock(directory, exclusive=True):
        if (directory / COMPLETION_FILENAME).exists():
            raise RunCompletedError(f"Cannot append infrastructure attempt {identity!r}: run is already complete")
        _, by_id = _scan_infra_errors(journal_path, repair_trailing_record=True)
        existing = by_id.get(identity)
        if existing is not None:
            if existing != value:
                raise DuplicateEpisodeError(f"Conflicting record for attempt_id={identity!r}")
            return False
        descriptor = os.open(journal_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        os.fchmod(descriptor, 0o644)
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError(f"Failed to append infrastructure record to {journal_path}")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(directory)
    return True


def read_episodes(run_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read episode records in append order, collapsing identical duplicates."""

    directory = validate_run_dir(run_dir)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    with _journal_lock(directory, exclusive=False):
        records, _ = _scan_episodes(directory / EPISODES_FILENAME, repair_trailing_record=False)
    for record in records:
        _validate_terminal_episode(record)
    return records


def read_infra_errors(run_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read retryable infrastructure attempts in append order."""
    directory = validate_run_dir(run_dir)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    with _journal_lock(directory, exclusive=False):
        records, _ = _scan_infra_errors(directory / INFRA_ERRORS_FILENAME, repair_trailing_record=False)
    for record in records:
        _validate_infra_attempt(record)
    return records


def completed_episode_ids(run_dir: str | os.PathLike[str]) -> set[str]:
    """Return stable episode IDs already present in the journal."""

    return {_episode_id(record) for record in read_episodes(run_dir)}


def mark_complete(
    run_dir: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
    completed_at: str | None = None,
) -> Path:
    """Atomically write the run completion marker.

    A manifest is required before completion.  Repeated calls are idempotent and
    preserve the original timestamp and metadata.
    """

    directory = validate_run_dir(run_dir)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ArtifactError(f"Cannot complete a run without {MANIFEST_FILENAME}: {directory}")

    marker_path = directory / COMPLETION_FILENAME
    with _journal_lock(directory, exclusive=True):
        if marker_path.exists():
            return marker_path
        episodes, _ = _scan_episodes(directory / EPISODES_FILENAME, repair_trailing_record=True)
        for record in episodes:
            _validate_terminal_episode(record)
        infra_attempts, _ = _scan_infra_errors(directory / INFRA_ERRORS_FILENAME, repair_trailing_record=True)
        for record in infra_attempts:
            _validate_infra_attempt(record)
        terminal_ids = {_episode_id(record) for record in episodes}
        unresolved = {_episode_id(record) for record in infra_attempts} - terminal_ids
        if unresolved:
            raise ArtifactError(f"Cannot complete run with retryable infrastructure episodes: {sorted(unresolved)}")
        marker: dict[str, Any] = {
            "completed_at": completed_at or datetime.now(timezone.utc).isoformat(),
            "episode_count": len(episodes),
        }
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise TypeError("completion metadata must be a mapping")
            reserved = marker.keys() & metadata.keys()
            if reserved:
                raise ValueError(f"Completion metadata cannot replace reserved fields: {sorted(reserved)}")
            marker.update(metadata)
        _atomic_write(marker_path, _json_bytes(marker))
    return marker_path


def is_complete(run_dir: str | os.PathLike[str]) -> bool:
    """Return whether the atomic completion marker exists."""

    directory = validate_run_dir(run_dir)
    return (directory / COMPLETION_FILENAME).is_file()


def read_completion_marker(run_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Read the completion marker, or return ``None`` for an active run."""

    directory = validate_run_dir(run_dir)
    marker_path = directory / COMPLETION_FILENAME
    if not marker_path.exists():
        return None
    with marker_path.open(encoding="utf-8") as stream:
        marker = json.load(stream)
    if not isinstance(marker, dict):
        raise ArtifactFormatError(f"Completion marker must contain a JSON object: {marker_path}")
    return marker
