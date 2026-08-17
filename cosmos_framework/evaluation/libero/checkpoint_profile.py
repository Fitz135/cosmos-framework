# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Resolve strict Cosmos3-Edge checkpoint metadata before LIBERO evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import ValidationError

from cosmos_framework.evaluation.libero.schema import (
    CheckpointFormat,
    CheckpointRole,
    LiberoCheckpointProfile,
    LiberoCheckpointProfileSpec,
    LiberoTargetAdapterSpec,
    WeightsVariant,
)


class CheckpointProfileError(ValueError):
    """Raised when checkpoint metadata is absent, ambiguous, or contradictory."""


_REQUIRED_FIELDS = (
    "checkpoint_role",
    "zero_shot",
    "domain_name",
    "action_chunk_size",
    "conditioning_fps",
    "effective_action_dim",
    "action_space",
    "rotation_space",
    "pose_coordinate_frame",
    "action_normalization",
    "action_stats_path",
    "format_prompt_as_json",
    "cameras",
    "image_size",
    "rotate_images",
    "control_mode",
    "control_frequency",
    "gripper_mode",
    "weights_variant",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointProfileError(f"Could not read JSON metadata at {path}: {error}") from error
    if not isinstance(value, dict):
        raise CheckpointProfileError(f"Expected a JSON object at {path}")
    return value


def _load_profile_spec(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        spec = LiberoCheckpointProfileSpec.model_validate(_read_json(path))
    except ValidationError as error:
        raise CheckpointProfileError(f"Invalid explicit checkpoint profile {path}: {error}") from error
    values = spec.model_dump(mode="json", exclude_none=True)
    expected_hash = values.pop("profile_hash", None)
    values.pop("schema_version", None)
    _resolve_relative_stats_path(values, path.parent)
    return values, expected_hash


def _load_target_adapter(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        adapter = LiberoTargetAdapterSpec.model_validate(_read_json(path))
    except ValidationError as error:
        raise CheckpointProfileError(f"Invalid LIBERO target adapter {path}: {error}") from error
    values = adapter.model_dump(mode="json", exclude_none=True)
    adapter_id = values.pop("adapter_id")
    values.pop("target", None)
    values.pop("schema_version", None)
    values.pop("profile_hash", None)
    _resolve_relative_stats_path(values, path.parent)
    return adapter_id, values


def _resolve_relative_stats_path(values: dict[str, Any], base_dir: Path) -> None:
    raw_path = values.get("action_stats_path")
    if raw_path is None:
        return
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    values["action_stats_path"] = str(path.resolve())


def _detect_checkpoint_format(checkpoint_path: Path) -> CheckpointFormat:
    candidates = (checkpoint_path, checkpoint_path / "model")
    for candidate in candidates:
        if (candidate / ".metadata").is_file() and any(candidate.glob("*.distcp")):
            return CheckpointFormat.DCP
    transformer = checkpoint_path / "transformer"
    if (
        any(checkpoint_path.glob("*.safetensors"))
        or any(checkpoint_path.glob("*.safetensors.index.json"))
        or ((checkpoint_path / "model_index.json").is_file() and any(transformer.glob("*.safetensors*")))
    ):
        return CheckpointFormat.HF
    if any((candidate / ".metadata").is_file() for candidate in candidates):
        raise CheckpointProfileError(f"DCP checkpoint contains no *.distcp weight files: {checkpoint_path}")
    if any((checkpoint_path / name).is_file() for name in ("checkpoint.json", "config.json", "model_index.json")):
        raise CheckpointProfileError(f"HF checkpoint contains no *.safetensors weight files: {checkpoint_path}")
    raise CheckpointProfileError(f"Unknown or incomplete checkpoint format at {checkpoint_path}")


def _discover_resolved_config(checkpoint_path: Path) -> Path | None:
    candidates = [checkpoint_path / "config.yaml"]
    if checkpoint_path.parent.name == "checkpoints":
        candidates.append(checkpoint_path.parent.parent / "config.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _iter_mappings(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _iter_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_mappings(child)


def _resolve_dcp_stats_path(raw_path: str, config_path: Path) -> str:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path)
    package_stats = Path(__file__).resolve().parents[2] / "data" / "generator" / "action" / "normalizer_stats" / path
    candidates = (config_path.parent / path, package_stats)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return str((config_path.parent / path).resolve())


def _merge_fields(
    merged: dict[str, Any],
    field_sources: dict[str, str],
    incoming: Mapping[str, Any],
    source: str,
) -> None:
    for name, value in incoming.items():
        if value is None or name in {"schema_version", "profile_hash"}:
            continue
        if name == "cameras":
            value = tuple(value)
        if hasattr(value, "value"):
            value = value.value
        if name in merged and merged[name] != value:
            raise CheckpointProfileError(
                f"Conflicting value for {name!r}: {merged[name]!r} from {field_sources[name]} "
                f"versus {value!r} from {source}"
            )
        merged[name] = value
        field_sources.setdefault(name, source)


def _dcp_config_fields(config_path: Path) -> dict[str, Any]:
    try:
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    except Exception as error:
        raise CheckpointProfileError(f"Could not read resolved DCP config {config_path}: {error}") from error

    policy_nodes = [
        node
        for node in _iter_mappings(config)
        if "chunk_length" in node and ("action_normalization" in node or "action_space" in node)
    ]
    if not policy_nodes:
        return {}

    merged: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for index, node in enumerate(policy_nodes):
        rotation_space = node.get("rotation_space")
        effective_action_dim = 10 if rotation_space == "6d" else 7 if rotation_space == "3d" else None
        camera_mode = node.get("camera_mode")
        cameras = None
        if camera_mode == "concat_view":
            wrist_camera = node.get("wrist_camera_key")
            if wrist_camera is not None:
                cameras = ("agentview", "wrist")
        elif camera_mode in {"single_view", "external"}:
            cameras = ("agentview",)

        stats_path = node.get("action_stats_path")
        fields = {
            "domain_name": node.get("embodiment_type"),
            "action_chunk_size": node.get("chunk_length"),
            "conditioning_fps": node.get("fps"),
            "effective_action_dim": effective_action_dim,
            "action_space": node.get("action_space"),
            "rotation_space": rotation_space,
            "pose_coordinate_frame": node.get("pose_coordinate_frame"),
            "action_normalization": node.get("action_normalization"),
            "action_stats_path": (
                _resolve_dcp_stats_path(str(stats_path), config_path) if stats_path is not None else None
            ),
            "format_prompt_as_json": node.get("format_prompt_as_json"),
            "cameras": cameras,
            "image_size": node.get("image_size"),
        }
        _merge_fields(merged, sources, fields, f"dcp-config:{config_path}#{index}")
    return merged


def _hf_checkpoint_fields(checkpoint_path: Path) -> dict[str, Any]:
    metadata_path = checkpoint_path / "checkpoint.json"
    if not metadata_path.is_file():
        return {}
    metadata = _read_json(metadata_path)
    policy = metadata.get("policy")
    if policy is None:
        return {}
    if not isinstance(policy, dict):
        raise CheckpointProfileError(f"Expected checkpoint.json.policy to be an object at {metadata_path}")

    fields: dict[str, Any] = {
        "domain_name": policy.get("domain_name"),
        "action_chunk_size": policy.get("action_chunk_size"),
        "conditioning_fps": policy.get("conditioning_fps"),
    }
    for name in _REQUIRED_FIELDS:
        if name in policy:
            fields[name] = policy[name]
    if "action_stats_path" in fields:
        _resolve_relative_stats_path(fields, checkpoint_path)
    use_ema_weights = metadata.get("use_ema_weights")
    if isinstance(use_ema_weights, bool):
        fields["weights_variant"] = WeightsVariant.EMA.value if use_ema_weights else WeightsVariant.REGULAR.value
    return fields


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise CheckpointProfileError(f"Could not hash action statistics at {path}: {error}") from error
    return digest.hexdigest()


def _hash_checkpoint_files(files: Mapping[str, Path]) -> str:
    """Hash labeled checkpoint files with unambiguous path/content framing."""
    digest = hashlib.sha256(b"cosmos-libero-checkpoint-v1\0")
    for label, path in sorted(files.items()):
        encoded_label = label.encode("utf-8")
        try:
            before = path.stat()
            digest.update(len(encoded_label).to_bytes(8, "big"))
            digest.update(encoded_label)
            digest.update(before.st_size.to_bytes(8, "big"))
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            after = path.stat()
        except OSError as error:
            raise CheckpointProfileError(f"Could not fingerprint checkpoint file {path}: {error}") from error
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise CheckpointProfileError(f"Checkpoint file changed while fingerprinting: {path}")
    return digest.hexdigest()


def _checkpoint_fingerprint(
    checkpoint: Path,
    checkpoint_format: CheckpointFormat,
    *,
    resolved_config_path: Path | None,
) -> str:
    """Return a stable fingerprint of weights and load-critical metadata."""
    if checkpoint_format == CheckpointFormat.HF:
        weight_files = sorted(path for path in checkpoint.rglob("*.safetensors") if path.is_file())
        if not weight_files:
            raise CheckpointProfileError(f"HF checkpoint contains no *.safetensors weight files: {checkpoint}")
        metadata_suffixes = {".json", ".yaml", ".yml", ".toml", ".txt", ".jinja", ".model", ".py"}
        excluded_parts = {"cache", "assets", "images", "verify", "verification"}
        metadata_files = {
            path
            for path in checkpoint.rglob("*")
            if path.is_file()
            and path.suffix.lower() in metadata_suffixes
            and not any(part.lower() in excluded_parts for part in path.relative_to(checkpoint).parts[:-1])
            and not path.name.lower().startswith("readme")
        }
        selected = set(weight_files) | metadata_files
        labeled = {path.relative_to(checkpoint).as_posix(): path for path in selected}
        if resolved_config_path is not None:
            labeled[f"__resolved_config__/{resolved_config_path.name}"] = resolved_config_path
        return _hash_checkpoint_files(labeled)

    distcp_files = sorted(path for path in checkpoint.rglob("*.distcp") if path.is_file())
    metadata_files = sorted(path for path in checkpoint.rglob(".metadata") if path.is_file())
    if not distcp_files:
        raise CheckpointProfileError(f"DCP checkpoint contains no *.distcp weight files: {checkpoint}")
    if not metadata_files:
        raise CheckpointProfileError(f"DCP checkpoint contains no .metadata file: {checkpoint}")
    if resolved_config_path is None:
        raise CheckpointProfileError("DCP checkpoint fingerprint requires an explicit or discoverable resolved config")
    labeled = {path.relative_to(checkpoint).as_posix(): path for path in [*metadata_files, *distcp_files]}
    labeled[f"__resolved_config__/{resolved_config_path.name}"] = resolved_config_path
    return _hash_checkpoint_files(labeled)


def _profile_hash(values: Mapping[str, Any]) -> str:
    excluded = {
        "checkpoint_path",
        "checkpoint_format",
        "target_adapter_id",
        "action_stats_path",
        "profile_hash",
        "profile_sources",
    }
    semantic = {name: value for name, value in values.items() if name not in excluded}
    payload = json.dumps(semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def resolve_checkpoint_profile(
    checkpoint_path: str | Path,
    *,
    profile_path: str | Path | None = None,
    target_adapter_path: str | Path | None = None,
    resolved_config_path: str | Path | None = None,
    weights_variant: WeightsVariant | str | None = None,
) -> LiberoCheckpointProfile:
    """Resolve every semantic needed by an Edge LIBERO policy.

    Sources are merged without precedence: duplicate values must agree. Base
    checkpoints are identified by the absence of HF policy metadata or a DCP
    action config and require an explicit target adapter.
    """

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise CheckpointProfileError(f"Checkpoint directory does not exist: {checkpoint}")
    checkpoint_format = _detect_checkpoint_format(checkpoint)

    merged: dict[str, Any] = {}
    field_sources: dict[str, str] = {}
    source_names: list[str] = []
    expected_profile_hash: str | None = None

    native_fields: dict[str, Any]
    resolved_checkpoint_config_path = (
        Path(resolved_config_path).expanduser().resolve() if resolved_config_path else None
    )
    if resolved_checkpoint_config_path is not None and not resolved_checkpoint_config_path.is_file():
        raise CheckpointProfileError(f"Resolved checkpoint config does not exist: {resolved_checkpoint_config_path}")
    if checkpoint_format == CheckpointFormat.HF:
        native_fields = _hf_checkpoint_fields(checkpoint)
        if native_fields:
            source_names.append("hf:checkpoint.json.policy")
    else:
        if resolved_checkpoint_config_path is None:
            resolved_checkpoint_config_path = _discover_resolved_config(checkpoint)
        if resolved_checkpoint_config_path is not None:
            native_fields = _dcp_config_fields(resolved_checkpoint_config_path)
            if native_fields:
                source_names.append(f"dcp:{resolved_checkpoint_config_path}")
        else:
            native_fields = {}

    checkpoint_fingerprint = _checkpoint_fingerprint(
        checkpoint,
        checkpoint_format,
        resolved_config_path=resolved_checkpoint_config_path,
    )

    role = CheckpointRole.FINETUNED if native_fields else CheckpointRole.BASE
    intrinsic = {"checkpoint_role": role.value, "zero_shot": role == CheckpointRole.BASE}
    _merge_fields(merged, field_sources, intrinsic, "checkpoint-classification")
    native_source = source_names[-1] if source_names else "checkpoint"
    _merge_fields(merged, field_sources, native_fields, native_source)

    if profile_path is not None:
        explicit_path = Path(profile_path).expanduser().resolve()
        fields, expected_profile_hash = _load_profile_spec(explicit_path)
        _merge_fields(merged, field_sources, fields, f"profile:{explicit_path}")
        source_names.append(f"profile:{explicit_path}")

    target_adapter_id: str | None = None
    if target_adapter_path is not None:
        adapter_path = Path(target_adapter_path).expanduser().resolve()
        target_adapter_id, fields = _load_target_adapter(adapter_path)
        _merge_fields(merged, field_sources, fields, f"target-adapter:{adapter_path}")
        source_names.append(f"target-adapter:{adapter_path}")
    elif role == CheckpointRole.BASE:
        raise CheckpointProfileError(
            "Base Cosmos3-Edge checkpoints are not LIBERO policies; pass an explicit target_adapter_path "
            "to run a zero-shot diagnostic"
        )

    if weights_variant is not None:
        try:
            variant = WeightsVariant(weights_variant).value
        except ValueError as error:
            raise CheckpointProfileError(f"Invalid weights_variant: {weights_variant!r}") from error
        _merge_fields(merged, field_sources, {"weights_variant": variant}, "runtime:weights_variant")
        source_names.append("runtime:weights_variant")

    missing = [name for name in _REQUIRED_FIELDS if name not in merged]
    if missing:
        raise CheckpointProfileError(
            "Incomplete LIBERO checkpoint profile; missing required fields: " + ", ".join(sorted(missing))
        )

    stats_path = Path(str(merged["action_stats_path"])).expanduser().resolve()
    if not stats_path.is_file():
        raise CheckpointProfileError(f"Action statistics file does not exist: {stats_path}")
    stats_sha256 = _sha256(stats_path)
    configured_stats_sha = merged.get("action_stats_sha256")
    if configured_stats_sha is not None and configured_stats_sha != stats_sha256:
        raise CheckpointProfileError(
            f"Action statistics SHA-256 mismatch for {stats_path}: expected {configured_stats_sha}, got {stats_sha256}"
        )
    merged["action_stats_path"] = str(stats_path)
    merged["action_stats_sha256"] = stats_sha256
    configured_checkpoint_fingerprint = merged.get("checkpoint_fingerprint")
    if configured_checkpoint_fingerprint is not None and configured_checkpoint_fingerprint != checkpoint_fingerprint:
        raise CheckpointProfileError(
            "Checkpoint fingerprint mismatch: "
            f"expected {configured_checkpoint_fingerprint}, resolved {checkpoint_fingerprint}"
        )
    merged["checkpoint_fingerprint"] = checkpoint_fingerprint

    values = {
        "schema_version": 1,
        "checkpoint_path": str(checkpoint),
        "checkpoint_format": checkpoint_format.value,
        "target_adapter_id": target_adapter_id,
        **merged,
        "profile_hash": "0" * 64,
        "profile_sources": tuple(source_names),
    }
    values["profile_hash"] = _profile_hash(values)
    if expected_profile_hash is not None and expected_profile_hash != values["profile_hash"]:
        raise CheckpointProfileError(
            f"Explicit profile hash mismatch: expected {expected_profile_hash}, resolved {values['profile_hash']}"
        )
    try:
        return LiberoCheckpointProfile.model_validate(values)
    except ValidationError as error:
        raise CheckpointProfileError(f"Invalid resolved LIBERO checkpoint profile: {error}") from error
