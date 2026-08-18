# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Pure helpers for 'cosmos_framework.scripts.export_model'.

Kept free of heavy imports (torch, init_script side effects) so they are
unit-testable without a GPU or downloads; registry-touching helpers import
lazily and accept an injectable download callable.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import re
import shutil
import stat
from pathlib import Path
from pydoc import locate
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from cosmos_framework.utils.checkpoint_db import CheckpointDirHf

_EDGE_MODEL_NAME = "nvidia/Cosmos3-Edge-Reasoner"

# HF-standard processor/tokenizer files shipped at the snapshot root. The exported
# config.json is deliberately absent (it must never be overwritten by bundling).
PROCESSOR_FILE_PATTERNS = (
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template*",
    "video_preprocessor*",
)

# Keys the bundled vision_encoder/config.json must carry so inference never needs
# the hub repo's top-level config (see Nemotron3DenseVLTextForCausalLM._ensure_vision_tower).
_BUNDLE_CONFIG_DICT_KEYS = ("vision_config", "projector_config")
_BUNDLE_CONFIG_TOKEN_ID_KEYS = ("image_token_id", "video_token_id", "vision_start_token_id")

_HF_SNAPSHOT_RE = re.compile(r"/snapshots/([0-9a-f]{40})(?:/|$)")


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _dataset_config_value(dataset_config: Any, key: str) -> Any:
    value = _config_value(dataset_config, key)
    if value is not None:
        return value

    target = _config_value(dataset_config, "_target_") or _config_value(dataset_config, "_target")
    if target is None:
        return None
    if isinstance(target, str):
        target = locate(target)
        if target is None:
            return None
    try:
        parameter = inspect.signature(target).parameters.get(key)
    except (TypeError, ValueError):
        return None
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    return parameter.default


def _edge_action_dataset_configs(training_config: Any) -> list[Any]:
    """Return action dataset configs from legacy and packing dataloaders."""
    dataloader_train = _config_value(training_config, "dataloader_train")

    # Legacy action recipes wrap datasets in the joint-dataloader hierarchy.
    dataloaders = _config_value(dataloader_train, "dataloaders")
    action_data = _config_value(dataloaders, "action_data")
    action_dataloader = _config_value(action_data, "dataloader")
    wrapper_config = _config_value(action_dataloader, "dataset")
    legacy_entries = _config_value(wrapper_config, "list_of_datasets")
    if legacy_entries:
        configs = [_config_value(entry, "dataset") for entry in legacy_entries]
        if all(config is not None for config in configs):
            return configs

    # PackingDataLoader recipes place named dataset entries directly under the
    # rank-partitioned dataloader.
    packed_dataloader = _config_value(dataloader_train, "dataloader")
    packed_entries = _config_value(packed_dataloader, "datasets")
    if packed_entries:
        entries = packed_entries.values() if hasattr(packed_entries, "values") else packed_entries
        configs = [_config_value(entry, "dataset") for entry in entries]
        if configs and all(config is not None for config in configs):
            return configs

    raise ValueError(
        "Cosmos3 Edge export requires action dataset configs at either "
        "dataloader_train.dataloaders.action_data.dataloader.dataset.list_of_datasets "
        "or dataloader_train.dataloader.datasets."
    )


def build_edge_policy_metadata(training_config: Any) -> dict[str, Any]:
    """Resolve policy manifest fields from an action experiment config."""
    metadata_by_dataset: list[dict[str, Any]] = []
    for action_dataset_config in _edge_action_dataset_configs(training_config):
        target = _config_value(action_dataset_config, "_target_") or _config_value(action_dataset_config, "_target")
        action_chunk_size = _dataset_config_value(action_dataset_config, "chunk_length")
        conditioning_fps = _dataset_config_value(action_dataset_config, "fps")
        domain_name = _dataset_config_value(action_dataset_config, "embodiment_type")
        if domain_name is None and target is not None:
            domain_name = getattr(target, "EMBODIMENT_TYPE", None)

        if not isinstance(action_chunk_size, int) or isinstance(action_chunk_size, bool) or action_chunk_size <= 0:
            raise ValueError(
                "Cosmos3 Edge action dataset config must define a positive integer `chunk_length`, "
                f"got {action_chunk_size!r}."
            )
        if (
            not isinstance(conditioning_fps, (int, float))
            or isinstance(conditioning_fps, bool)
            or conditioning_fps <= 0
        ):
            raise ValueError(
                f"Cosmos3 Edge action dataset config must define a positive numeric `fps`, got {conditioning_fps!r}."
            )
        if not isinstance(domain_name, str) or not domain_name.strip():
            target_name = getattr(target, "__name__", repr(target))
            raise ValueError(
                "Cosmos3 Edge action dataset config must define `embodiment_type` or expose "
                f"`EMBODIMENT_TYPE`; dataset target is {target_name}."
            )

        metadata_by_dataset.append(
            {
                "action_chunk_size": action_chunk_size,
                "conditioning_fps": float(conditioning_fps),
                "domain_name": domain_name.strip(),
            }
        )

    metadata = metadata_by_dataset[0]
    for field in metadata:
        if any(dataset_metadata[field] != metadata[field] for dataset_metadata in metadata_by_dataset[1:]):
            raise ValueError(
                "Cosmos3 Edge checkpoint.json can represent only one action policy metadata value per field; "
                f"the configured datasets disagree on `{field}`."
            )
    return metadata


def canonicalize_edge_local_processor(model_dict: dict[str, Any]) -> bool:
    """Replace an export-host-local Edge processor path with its public family name."""
    tokenizer_cfg = ((model_dict.get("config") or {}).get("vlm_config") or {}).get("tokenizer")
    if not isinstance(tokenizer_cfg, dict) or tokenizer_cfg.get("repository"):
        return False
    tokenizer_type = tokenizer_cfg.get("tokenizer_type")
    if not isinstance(tokenizer_type, str) or not Path(tokenizer_type).is_dir():
        return False
    tokenizer_cfg["repository"] = None
    tokenizer_cfg["revision"] = None
    tokenizer_cfg["subdir"] = ""
    tokenizer_cfg["tokenizer_type"] = _EDGE_MODEL_NAME
    return True


def is_edge_model(model_dict: dict[str, Any]) -> bool:
    """Return True when the model config describes a Cosmos3-Edge reasoner backbone."""
    vlm_config = (model_dict.get("config") or {}).get("vlm_config") or {}
    if vlm_config.get("model_name") == _EDGE_MODEL_NAME:
        return True
    model_instance = vlm_config.get("model_instance") or {}
    if "Nemotron3DenseVLTextForCausalLM" in str(model_instance.get("_target_", "")):
        return True
    backbone_path = (vlm_config.get("pretrained_weights") or {}).get("backbone_path") or ""
    return "Cosmos3-Edge" in backbone_path


def resolve_edge_export_policy_metadata(
    training_config: Any,
    model_dict: dict[str, Any],
    *,
    base_checkpoint: bool,
    use_ema_weights: bool,
) -> dict[str, Any] | None:
    """Resolve Edge policy metadata or validate an explicit base export.

    Base Cosmos3-Edge DCPs are action-capable but are not task-finetuned
    policies, so their exports intentionally omit ``checkpoint.json.policy``.
    The explicit flag keeps that omission from hiding a malformed fine-tuning
    config on the default export path.
    """
    is_edge = is_edge_model(model_dict)
    action_gen = (model_dict.get("config") or {}).get("action_gen")

    if base_checkpoint:
        if not is_edge:
            raise ValueError("--base-checkpoint requires a Cosmos3-Edge model config.")
        if action_gen is not True:
            raise ValueError("--base-checkpoint requires model.config.action_gen=true.")
        if use_ema_weights:
            raise ValueError("--base-checkpoint requires regular weights; pass --no-use-ema-weights.")
        return None

    if is_edge and action_gen:
        return build_edge_policy_metadata(training_config)
    return None


def reasoner_vision_capable(model_dict: dict[str, Any]) -> bool:
    """Whether the checkpoint being exported can serve reasoner image/video prompts.

    Dict-shaped mirror of 'inference.args._reasoner_vision_capable' (keep the
    two in sync). Edge is always capable (lazy tower); Qwen-family (Nano/Super)
    needs a truthy 'include_visual'. Unknown families are treated as capable
    (never block).
    """
    vlm_config = (model_dict.get("config") or {}).get("vlm_config") or {}
    model_instance = vlm_config.get("model_instance") or {}
    target = str(model_instance.get("_target_", "") if isinstance(model_instance, dict) else "")
    model_name = vlm_config.get("model_name") or ""
    if "Nemotron3DenseVLTextForCausalLM" in target or "Cosmos3-Edge" in model_name:
        return True
    if "Qwen" not in target and not model_name.startswith("Qwen/"):
        return True
    config = model_instance.get("config") if isinstance(model_instance, dict) else None
    include_visual = config.get("include_visual") if isinstance(config, dict) else None
    return bool(include_visual)


def set_include_visual(model_dict: dict[str, Any], value: bool) -> bool:
    """Set 'include_visual' on the 'create_vlm_config' node; return True when applied.

    The node's kwargs are applied to the MoT config wrapper via setattr at load
    time ('configs.base.defaults.reasoner.create_vlm_config'), and the wrapper's
    'include_visual' gates visual-tower construction
    ('_MoTConfigBase.vision_config'), so this is what makes a '--no-vit' export
    loadable without vision weights.
    """
    vlm_config = (model_dict.get("config") or {}).get("vlm_config") or {}
    model_instance = vlm_config.get("model_instance")
    if not isinstance(model_instance, dict):
        return False
    config = model_instance.get("config")
    if not isinstance(config, dict):
        return False
    config["include_visual"] = value
    return True


def resolve_vision_bundle_dirs(vlm_checkpoint_path: Path) -> tuple[Path, Path | None]:
    """Locate the 'vision_encoder/' weights dir and the snapshot root.

    Accepts either an nvidia/Cosmos3-Edge snapshot root (contains
    'vision_encoder/model.safetensors') or a 'vision_encoder/' directory passed
    directly via '--vit-checkpoint-path'. The returned root is None when the
    weights directory was passed directly and its parent has no 'config.json'.
    """
    if (vlm_checkpoint_path / "vision_encoder" / "model.safetensors").is_file():
        return vlm_checkpoint_path / "vision_encoder", vlm_checkpoint_path
    if (vlm_checkpoint_path / "model.safetensors").is_file():
        root = vlm_checkpoint_path.parent
        return vlm_checkpoint_path, root if (root / "config.json").is_file() else None
    raise FileNotFoundError(
        f"No 'vision_encoder/model.safetensors' under '{vlm_checkpoint_path}' (expected an "
        "nvidia/Cosmos3-Edge snapshot root or a vision_encoder/ directory)."
    )


def build_vision_encoder_bundle_config(vision_encoder_dir: Path, snapshot_root: Path | None) -> dict[str, Any]:
    """Build the merged, self-describing 'vision_encoder/config.json' for the bundle.

    Configs and multimodal token ids resolve standalone-first (matching
    '_ensure_vision_tower'), falling back to the snapshot's top-level
    'config.json', so inference never needs the hub repo's top-level config.
    """
    standalone_file = vision_encoder_dir / "config.json"
    standalone: dict[str, Any] | None = None
    if standalone_file.is_file():
        standalone = json.loads(standalone_file.read_text())
    top: dict[str, Any] | None = None
    if snapshot_root is not None and (snapshot_root / "config.json").is_file():
        top = json.loads((snapshot_root / "config.json").read_text())
    if standalone is None and top is None:
        raise FileNotFoundError(
            f"No 'config.json' found in '{vision_encoder_dir}' or its snapshot root; cannot "
            "build the vision_encoder bundle config. Point --vit-checkpoint-path at a full "
            "nvidia/Cosmos3-Edge snapshot directory."
        )

    merged: dict[str, Any] = {}
    for key in _BUNDLE_CONFIG_DICT_KEYS:
        value = (standalone or {}).get(key) or (top or {}).get(key)
        if not isinstance(value, dict):
            raise ValueError(
                f"Missing '{key}' in both the standalone vision_encoder config and the snapshot's "
                "top-level config.json; cannot build the vision_encoder bundle config."
            )
        merged[key] = dict(value)
    for key in _BUNDLE_CONFIG_TOKEN_ID_KEYS:
        value = (standalone or {}).get(key)
        if value is None:
            value = (top or {}).get(key)
        if not isinstance(value, int):
            raise ValueError(
                f"Cannot resolve '{key}' for the vision_encoder bundle config (absent from both the "
                "standalone vision_encoder config and the snapshot's top-level config.json). Point "
                "--vit-checkpoint-path at a full nvidia/Cosmos3-Edge snapshot directory."
            )
        merged[key] = value
    return merged


def bundle_vision_encoder(vision_encoder_dir: Path, snapshot_root: Path | None, output_dir: Path) -> None:
    """Copy 'vision_encoder/model.safetensors' verbatim and write the merged config."""
    bundle_config = build_vision_encoder_bundle_config(vision_encoder_dir, snapshot_root)
    output_vision_dir = output_dir / "vision_encoder"
    output_vision_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vision_encoder_dir / "model.safetensors", output_vision_dir / "model.safetensors")
    (output_vision_dir / "config.json").write_text(json.dumps(bundle_config, indent=2, sort_keys=True) + "\n")


def bundle_processor_files(snapshot_root: Path, output_dir: Path) -> list[str]:
    """Copy HF-standard processor/tokenizer files from the snapshot root; return copied names.

    All-or-nothing: on a mid-copy failure the already-copied files are removed
    (best-effort) before the exception propagates — inference prefers local
    processor files over the hub, so a partial set must never be left behind.
    """
    copied: list[str] = []
    try:
        for pattern in PROCESSOR_FILE_PATTERNS:
            for source in sorted(snapshot_root.glob(pattern)):
                if not source.is_file():
                    continue
                assert source.name != "config.json", "processor bundling must never overwrite the exported config.json"
                shutil.copyfile(source, output_dir / source.name)
                copied.append(source.name)
    except Exception:
        for name in copied:
            with contextlib.suppress(OSError):
                (output_dir / name).unlink()
        raise
    return copied


# Hugging Face repo ids are 'org/name'; anything else in a tokenizer node
# (local directory, URI) has no hub snapshot to bundle from.
_HF_REPO_ID_RE = re.compile(r"[\w.\-]+/[\w.\-]+")

# Registered tokenizer snapshots live under this sanitized S3 prefix in the
# checkpoint registry ('inference.common.checkpoints.register_checkpoints');
# it is the prefix the training/inference download paths consult for plain
# repo ids ('configs.base.defaults.reasoner.download_tokenizer_files',
# 'maybe_download_hf_model_from_s3').
_TOKENIZER_REGISTRY_URI_PREFIX = "s3://bucket/cosmos3/pretrained/huggingface"

# Files fetched when downloading a processor snapshot for bundling; mirrors
# '_PROCESSOR_HF_INCLUDE' in 'cosmos_framework.inference.inference'.
PROCESSOR_HF_INCLUDE = ("*.json", "*.jinja", "merges.txt", "vocab.json")

# Files fetched when export resolves the Edge ViT bundle from the hub: the
# 'vision_encoder/' artifact plus the HF-standard config/processor files
# (~1 GB) — never the generation shards ('transformer/', 'vae/') or demo
# assets the bundle doesn't need (tens of GB on a cold cache). Applied at the
# export call site only: the registry entry itself must stay unfiltered
# because training's backbone seeding downloads the full repo through it
# ('load_language_model' resolves the root model.safetensors.index.json,
# whose weight_map points into 'transformer/*.safetensors').
EDGE_VIT_BUNDLE_HF_INCLUDE = ("vision_encoder/*", *PROCESSOR_HF_INCLUDE)


def processor_source_from_tokenizer_node(tokenizer_node: Any) -> dict[str, Any] | None:
    """Extract the VLM processor origin from a 'vlm_config.tokenizer' node.

    Accepts 'repository' (+'revision'/'subdir') nodes, or a plain HF repo id in
    'tokenizer_type' / 'pretrained_model_name' (revision None, pinned later via
    the checkpoint registry). Returns {'repository', 'revision', 'subdir'}, or
    None when the node names no HF repository (missing node, local dir, URI).
    """
    if not isinstance(tokenizer_node, dict):
        return None
    repository = tokenizer_node.get("repository")
    if isinstance(repository, str) and _HF_REPO_ID_RE.fullmatch(repository):
        return {
            "repository": repository,
            "revision": tokenizer_node.get("revision"),
            "subdir": tokenizer_node.get("subdir") or "",
        }
    repo_id = tokenizer_node.get("tokenizer_type") or tokenizer_node.get("pretrained_model_name")
    if isinstance(repo_id, str) and _HF_REPO_ID_RE.fullmatch(repo_id):
        return {"repository": repo_id, "revision": None, "subdir": ""}
    return None


def resolve_processor_download(source: dict[str, Any]) -> CheckpointDirHf:
    """Build the 'CheckpointDirHf' download spec for a processor source.

    An explicit revision is used as-is; revision None is pinned via the
    checkpoint registry, with unregistered repos falling back to 'main'. Only
    HF-standard processor/config files are included in the download.
    """
    from cosmos_framework.utils.checkpoint_db import CheckpointConfig, CheckpointDirHf, sanitize_uri

    repository: str = source["repository"]
    revision: str | None = source.get("revision")
    subdirectory: str = source.get("subdir") or ""
    if revision is None:
        registered = CheckpointConfig.maybe_from_uri(sanitize_uri(f"{_TOKENIZER_REGISTRY_URI_PREFIX}/{repository}"))
        if registered is not None and isinstance(registered.hf, CheckpointDirHf):
            repository = registered.hf.repository
            revision = registered.hf.revision
            subdirectory = registered.hf.subdirectory
        else:
            revision = "main"
    return CheckpointDirHf(
        repository=repository,
        revision=revision,
        subdirectory=subdirectory,
        include=PROCESSOR_HF_INCLUDE,
    )


def bundle_processor_from_tokenizer_node(
    tokenizer_node: Any,
    output_dir: Path,
    *,
    download: Callable[[CheckpointDirHf], str] | None = None,
) -> dict[str, Any] | None:
    """Best-effort processor bundling from the same origin inference uses.

    Resolves the 'vlm_config.tokenizer' node, copies the HF-standard processor
    files into the export root, and returns the manifest 'processor_source'
    entry. Any failure logs a warning and returns None — processor bundling
    must never fail an export. 'download' is injectable for tests.
    """
    from cosmos_framework.utils import log

    source = processor_source_from_tokenizer_node(tokenizer_node)
    if source is None:
        log.warning(
            "Skipping processor bundling: the model config's 'vlm_config.tokenizer' node names no "
            "Hugging Face repository. The exported checkpoint will fetch its processor at inference time."
        )
        return None
    try:
        checkpoint_dir = resolve_processor_download(source)
        snapshot_path = checkpoint_dir.download() if download is None else download(checkpoint_dir)
        copied = bundle_processor_files(Path(snapshot_path), output_dir)
    except Exception as e:
        log.warning(
            f"Skipping processor bundling: could not resolve the '{source['repository']}' processor "
            f"snapshot ({e}). The export continues; the checkpoint will fetch its processor at inference time."
        )
        return None
    if not copied:
        log.warning(f"Skipping processor bundling: no processor/tokenizer files found in '{snapshot_path}'.")
        return None
    log.info(f"Bundled processor files from '{checkpoint_dir.repository}': {', '.join(copied)}")
    return build_artifact_source(repo=checkpoint_dir.repository, resolved_path=snapshot_path, bundled=True)


def hf_revision_from_snapshot_path(path: str | Path) -> str | None:
    """Extract the commit hash from an HF cache snapshot path, else None."""
    match = _HF_SNAPSHOT_RE.search(Path(path).as_posix())
    return match.group(1) if match else None


def build_artifact_source(*, repo: str, resolved_path: str | Path | None, bundled: bool) -> dict[str, Any]:
    """Build a provenance entry for 'export_manifest.json'."""
    return {
        "repo": repo,
        "revision": hf_revision_from_snapshot_path(resolved_path) if resolved_path is not None else None,
        "bundled": bundled,
    }


def sanitize_export_args(export_args: dict[str, Any]) -> dict[str, Any]:
    """Redact local-path values from the manifest's 'export_args' entries.

    Mirrors '--student-only-checkpoint-metadata': '*_path' / '*_dir' keys and
    any string containing a path separator become None; other values pass
    through. Hub repo@revision provenance lives in '*_source' and is unaffected.
    """
    return {
        key: None
        if key.endswith(("_path", "_dir")) or (isinstance(value, str) and ("/" in value or "\\" in value))
        else value
        for key, value in export_args.items()
    }


def clean_stale_export_artifacts(output_dir: Path) -> list[str]:
    """Remove artifacts a previous export may have left in 'output_dir'; return removed names.

    Inference prefers bundled files local-first, so a re-export must not leave
    a stale 'vision_encoder/' or processor set behind. Only the paths this tool
    manages are removed; user files and the model shards are never touched.
    """
    removed: list[str] = []
    vision_encoder_dir = output_dir / "vision_encoder"
    if vision_encoder_dir.is_dir():
        shutil.rmtree(vision_encoder_dir)
        removed.append("vision_encoder/")
    for pattern in (*PROCESSOR_FILE_PATTERNS, "export_manifest.json"):
        for path in sorted(output_dir.glob(pattern)):
            if path.is_file():
                path.unlink()
                removed.append(path.name)
    return removed


def normalize_export_permissions(output_dir: Path) -> None:
    """Make a completed export readable and traversable by other users.

    Regular files are set to ``0644`` and directories to ``0755``. Symbolic
    links are never followed or changed, including links to directories outside
    the export root. A symlink passed as the export root is rejected because
    normalizing it would otherwise operate on its target.

    Permission errors propagate. If normalization fails after the exporter has
    written ``checkpoint.json``, that completion marker is removed so an
    incomplete export cannot be mistaken for a published checkpoint.
    """
    root = Path(output_dir)
    root_stat = root.lstat()
    if stat.S_ISLNK(root_stat.st_mode):
        raise ValueError(f"Refusing to normalize permissions through symlink export root '{root}'.")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise NotADirectoryError(f"Export root '{root}' is not a directory.")

    def set_mode(path: Path, mode: int, *, directory: bool) -> None:
        # Linux does not implement os.chmod(..., follow_symlinks=False).
        # Open with O_NOFOLLOW and chmod the descriptor instead: this both
        # avoids changing a symlink target and closes the lstat/chmod race.
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        flags |= os.O_DIRECTORY if directory else os.O_NONBLOCK
        descriptor = os.open(path, flags)
        try:
            opened_stat = os.fstat(descriptor)
            expected_type = stat.S_ISDIR(opened_stat.st_mode) if directory else stat.S_ISREG(opened_stat.st_mode)
            if not expected_type:
                raise OSError(f"Export path '{path}' changed type while normalizing permissions.")
            os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)

    try:
        set_mode(root, 0o755, directory=True)

        def raise_walk_error(error: OSError) -> None:
            raise error

        for current_root, directory_names, file_names in os.walk(root, topdown=True, onerror=raise_walk_error):
            current_dir = Path(current_root)
            traversable_directories: list[str] = []
            for name in directory_names:
                directory = current_dir / name
                entry_stat = directory.lstat()
                if stat.S_ISLNK(entry_stat.st_mode):
                    continue
                if stat.S_ISDIR(entry_stat.st_mode):
                    set_mode(directory, 0o755, directory=True)
                    traversable_directories.append(name)
            # Explicitly prune symlink (and raced non-directory) entries so os.walk
            # cannot descend through them even if platform defaults change.
            directory_names[:] = traversable_directories

            for name in file_names:
                file_path = current_dir / name
                entry_stat = file_path.lstat()
                if stat.S_ISREG(entry_stat.st_mode):
                    set_mode(file_path, 0o644, directory=False)
    except BaseException:
        # checkpoint.json is the export completion marker. Removing the link
        # itself is safe even if a corrupt export replaced it with a symlink.
        (root / "checkpoint.json").unlink(missing_ok=True)
        raise


# A NaN-latent decode yields a uniform frame (JPEG rounding keeps its pixel std
# well under 1), while any real generation lands orders of magnitude above.
_CONSTANT_IMAGE_STD_THRESHOLD = 1.0


def constant_image_failure(pixels: np.ndarray) -> str | None:
    """Heuristic '--verify' check for NaN/undecodable latents; None when the image passes.

    NaN latents decode into a valid, uniform (typically all-black) JPEG with
    status 'success', invisible to existence/size checks; a (near-)zero pixel
    std flags that class. Not a general image-quality gate.
    """
    array = np.asarray(pixels, dtype=np.float32)
    if array.size == 0:
        return "decoded to an empty pixel array"
    std = float(array.std())
    if std < _CONSTANT_IMAGE_STD_THRESHOLD:
        return (
            f"is (nearly) constant (pixel std {std:.4f} < {_CONSTANT_IMAGE_STD_THRESHOLD}); "
            "this is how NaN/undecodable latents decode (uniform/black frame)"
        )
    return None


def build_export_manifest(
    *,
    vision_tower_source: dict[str, Any] | None,
    processor_source: dict[str, Any] | None,
    export_args: dict[str, Any],
    framework_commit: str | None,
) -> dict[str, Any]:
    """Build the 'export_manifest.json' provenance sidecar."""
    return {
        "vision_tower_source": vision_tower_source,
        "processor_source": processor_source,
        "export_args": export_args,
        "framework_commit": framework_commit,
    }


def read_framework_commit(start: Path | None = None) -> str | None:
    """Resolve the framework git commit by reading '.git' files (no subprocess).

    Returns None outside a git worktree (e.g. a pip-installed package).
    """
    current = (start or Path(__file__)).resolve()
    for parent in [current, *current.parents]:
        git_dir = parent / ".git"
        if git_dir.is_dir():
            return _read_git_head(git_dir)
    return None


def _read_git_head(git_dir: Path) -> str | None:
    try:
        head = (git_dir / "HEAD").read_text().strip()
    except OSError:
        return None
    if not head.startswith("ref:"):
        return head or None
    ref = head.removeprefix("ref:").strip()
    ref_file = git_dir / ref
    if ref_file.is_file():
        return ref_file.read_text().strip() or None
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text().splitlines():
            if line.endswith(f" {ref}"):
                return line.split(" ", 1)[0]
    return None
