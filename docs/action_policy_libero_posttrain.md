# Cosmos3 LIBERO action-policy SFT

Full SFT of a Cosmos3 base into a LIBERO action policy: vision + language in,
action chunks out.

Three presets are provided (all use lr 5e-5, warmup 500, cycle 16000, and
global batch 2048):

- **(A) Nano libero_10-only** — trains on `libero_10` alone; peaks by ~iter 1500
  (max_iter 2000). Fast.
  `action_policy_libero_nano` + `action_policy_libero_10_nano.toml` +
  `launch_sft_action_policy_libero_10_nano.sh`.
- **(B) Nano libero-all** — equal mix of all 4 LIBERO suites; needs longer training
  (max_iter 5000).
  `action_policy_libero_all_nano` + `action_policy_libero_all_nano.toml` +
  `launch_sft_action_policy_libero_all_nano.sh`.
- **(C) Edge merged libero-all at 10 FPS** — trains all 40 tasks from one
  merged LeRobot v3 root, retains the pretrained Cosmos3-Edge action heads, and
  uses an 8-action chunk (max_iter 5000).
  `action_policy_libero_all_edge_10fps` +
  `action_policy_libero_all_edge_10fps.toml` +
  `launch_sft_action_policy_libero_all_edge_10fps.sh`.

| Piece            | Path                                                                                                 |
| ---------------- | ---------------------------------------------------------------------------------------------------- |
| Dataset          | `cosmos_framework/data/generator/action/datasets/libero_lerobot_dataset.py` (`LIBEROLeRobotDataset`) |
| SFT wrapper      | `get_action_libero_sft_dataset` in `.../datasets/action_sft_dataset.py`                              |
| Nano norm stats  | `.../normalizer_stats/libero_native_frame_wise_relative_rot6d.json`                                  |
| Edge norm stats  | `.../normalizer_stats/libero_10fps_pm_one_native_frame_wise_relative_rot6d.json`                     |
| Edge experiment  | `.../posttrain_config/action_policy_libero_all_edge_10fps.py`                                        |
| Edge run TOML    | `examples/toml/sft_config/action_policy_libero_all_edge_10fps.toml`                                  |
| Edge launch      | `examples/launch_sft_action_policy_libero_all_edge_10fps.sh`                                         |
| Inference server | `cosmos_framework/scripts/action_policy_server_libero.py`                                            |
| Edge preflight   | `cosmos_framework/evaluation/libero/preflight.py`                                                   |
| Edge orchestrator| `cosmos_framework/evaluation/libero/job.py`                                                         |
| Edge runner      | `cosmos_framework/evaluation/libero/runner.py`                                                      |
| Edge adapter     | `cosmos_framework/evaluation/libero/profiles/edge_libero_target_adapter.json`                       |
| Nano legacy eval | `cosmos_framework/simulation/libero/closed_loop_eval.py`                                             |

## 1. Data

`LIBEROLeRobotDataset` reads a local LeRobot directory. The two Nano presets use
the 20 FPS
[`nvidia/LIBERO_LeRobot_v3`](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3),
which the bundled `quantile_rot` stats and the 20 Hz eval assume.

**Preset A (libero_10-only)** — `LIBERO_ROOT` points at the `libero_10` suite dir:

```bash
hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset \
  --include 'libero_10/**' --local-dir <nfs>/LIBERO_LeRobot_v3
export LIBERO_ROOT=<nfs>/LIBERO_LeRobot_v3/libero_10
```

**Preset B (libero-all)** — download all 4 suites; `LIBERO_ROOT` is the **parent** dir:

```bash
hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --local-dir <nfs>/LIBERO_LeRobot_v3
export LIBERO_ROOT=<nfs>/LIBERO_LeRobot_v3          # parent of libero_spatial/object/goal/10
```

**Preset C (Edge merged 10 FPS)** — `LIBERO_ROOT` points at one LeRobot v3 root:

```bash
export LIBERO_ROOT=/path/to/merged-libero
test -f "$LIBERO_ROOT/meta/info.json"
```

Its metadata must declare 10 FPS and include
`observation.images.image` (third-person) plus
`observation.images.image2` (wrist). It uses all episodes (`split=full`), 9
observation frames and 8 actions per sample. The committed 10D rot6d statistics
match this dataset's `[-1, 1]` gripper convention. This recipe explicitly uses
LeRobot's `pyav` video backend, avoiding a runtime dependency on TorchCodec's
system FFmpeg shared libraries.

Actions are `frame_wise_relative` rot6d (10D = pos 3 + rot6d 6 + gripper 1),
`concat_view` (third-person + wrist, each 256×256 → 256×512), `quantile_rot`
normalized. The pipeline snaps the 256×512 concat to a 192×320 model canvas; the
eval server reproduces the same snap (§4).

## 2. Train

Convert the selected base checkpoint to DCP once (registered catalog name;
downloads from the HF Hub — see [docs/training.md](./training.md) Step 2):

```bash
# Nano presets:
python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o examples/checkpoints/Cosmos3-Nano \
  --checkpoint-path Cosmos3-Nano

# Edge preset:
python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o examples/checkpoints/Cosmos3-Edge \
  --checkpoint-path Cosmos3-Edge
```

When converting from a local Edge Hugging Face snapshot in an offline
environment, also pass
`--config-file cosmos_framework/inference/configs/model/Cosmos3-Edge.yaml`
and `--vae-path /path/to/Wan2.2_VAE.pth`. The converter will use the
snapshot-bundled Edge processor files instead of contacting the Hub.

Common env, then pick a preset launcher:

```bash
export LD_LIBRARY_PATH=''                      # NGC container: avoid torch._C import error
export BASE_CHECKPOINT_PATH=examples/checkpoints/Cosmos3-Nano   # the DCP dir from the convert step
export WAN_VAE_PATH=<Wan2.2_VAE.pth>
export IMAGINAIRE_OUTPUT_ROOT=/path/to/output_root

# Preset A — libero_10-only (LIBERO_ROOT = the libero_10 suite dir):
export LIBERO_ROOT=<nfs>/LIBERO_LeRobot_v3/libero_10
bash examples/launch_sft_action_policy_libero_10_nano.sh        # HSDP 2x8; set NNODES/NODE_RANK/MASTER_ADDR per node

# Preset B — libero-all 4-suite (LIBERO_ROOT = the LIBERO_LeRobot_v3 parent dir):
export LIBERO_ROOT=<nfs>/LIBERO_LeRobot_v3
bash examples/launch_sft_action_policy_libero_all_nano.sh    # HSDP 2x8; needs ~4500 iters to converge

# Preset C — Edge merged 10 FPS:
export BASE_CHECKPOINT_PATH=examples/checkpoints/Cosmos3-Edge
export LIBERO_ROOT=/path/to/merged-libero
# Optional for offline startup from an already-downloaded Edge HF snapshot:
export COSMOS3_EDGE_PROCESSOR_PATH=/path/to/Cosmos3-Edge-hf
bash examples/launch_sft_action_policy_libero_all_edge_10fps.sh  # FSDP8, one node
```

The Edge launcher requires Flash Attention 3 and deliberately provides no
attention fallback. It overwrites `I4_ATTN_BACKENDS` with the `flash3`
allow-list, then executes a small bfloat16 FA3 forward/backward kernel before
starting `torchrun`. The launch stops before model loading unless all of the
following hold:

- the worker uses an SM90 Hopper GPU (H100 or H200);
- `flash-attn-3-nv>=1.0.3` matches the active Torch/CUDA ABI;
- Cosmos resolves `flash3` as the only allowed and compatible backend; and
- the test kernel produces finite outputs and gradients.

The repository's `cu128-train` and `cu130-train` dependency groups provide
Torch 2.10-compatible FA3 builds. Do not use the `cu130-torch213` group for this
recipe because it does not currently declare a Torch 2.13 FA3 wheel. A manual
launch that bypasses the wrapper must reproduce its enforcement before
`torchrun`:

```bash
export I4_ATTN_BACKENDS=flash3
python -m cosmos_framework.scripts.check_flash_attention_3
```

The Nano recipes set lr 5e-5, warmup 500, cycle 16000, `save_iter=500`, HSDP 2x8 (global
batch 2048 = `max_samples_per_batch` 128 × 16 ranks × grad_accum 1). They differ only in
`max_iter`: **2000** for libero_10-only (peaks ~iter 1500), **5000** for libero-all
(the 4-suite mix takes longer to converge on libero_10, ~iter 4500).

The Edge recipe uses one-node FSDP8 and global batch 2048 =
`max_samples_per_batch` 128 × 8 ranks × grad accum 2. If 128 samples per rank
OOMs, preserve global batch with `64/4`, then `32/8`, using the launcher's
`EXTRA_TAIL_OVERRIDES`. W&B runs in offline mode. Periodic checkpoints are
written every 2000 optimizer steps. For the default 5000-step run this produces
`iter_000002000` and `iter_000004000`; the trainer's final-save guard also
writes `iter_000005000` because 5000 is not divisible by 2000.

The Edge recipe also generates one qualitative EMA WAM rollout after every
periodic checkpoint (every 2000 optimizer steps). The callback jointly samples
vision and action, then saves the video portion at 10 FPS. Its three rows are
the predicted rollout, the clean-latent VAE reconstruction ceiling, and ground
truth. Only data-parallel rank 0 writes the local video; offline W&B records
first, middle, and last preview frames. The final-save guard does not invoke an
`EveryN` callback, so generate a step-5000 visualization post hoc if required.

Videos are stored under:

```text
<IMAGINAIRE_OUTPUT_ROOT>/cosmos3_action_libero/action_sft/
  action_policy_libero_all_edge_10fps/EveryNDrawSample/Iter000002000/
```

The default uses one sample, guidance 1.0, and 8 denoising steps to bound the
training pause. Override the cadence or quality through the launcher, for
example:

```bash
EXTRA_TAIL_OVERRIDES="trainer.callbacks.libero_rollout.every_n=4000 trainer.callbacks.libero_rollout.num_sampling_step=30" \
  bash examples/launch_sft_action_policy_libero_all_edge_10fps.sh
```


## 3. Strict Cosmos3-Edge closed-loop eval

Cosmos3-Edge evaluation uses a strict pipeline:

1. the simulator preflight locks the LIBERO packages, assets, EGL renderer,
   camera streams, controller, and control frequency;
2. the checkpoint resolver merges load metadata with the explicit target
   adapter and rejects missing or conflicting policy semantics;
3. the single-checkpoint job starts the policy server, validates its versioned
   `/info` handshake and immutable checkpoint fingerprint, invokes the runner
   in a separate LIBERO environment, and cleans up both process groups;
4. the runner writes resumable episode records and seals a run only after every
   requested episode has a terminal non-infrastructure result.

Do not use the legacy Nano client for Edge checkpoints. It does not implement
the strict profile, fingerprint, handshake, or canonical artifact contract.

### 3.1 Formal three-checkpoint matrix

All three comparison targets are HF checkpoints and use the committed adapter:

```bash
TARGET_ADAPTER="$PWD/cosmos_framework/evaluation/libero/profiles/edge_libero_target_adapter.json"
```

| Checkpoint | Required job flags | Interpretation |
| ---------- | ------------------ | -------------- |
| Base Cosmos3-Edge action HF regular | `--checkpoint-path <BASE_EDGE_ACTION_HF> --target-adapter-path "$TARGET_ADAPTER" --weights-variant regular` | Explicit zero-shot LIBERO diagnostic. Report `checkpoint_role=base, zero_shot=true`; never describe it as a fine-tuned policy. |
| 5k fine-tuned Edge HF EMA | `--checkpoint-path <EDGE_5K_EMA_HF> --config-file <EDGE_5K_LOAD_CONFIG> --target-adapter-path "$TARGET_ADAPTER" --weights-variant ema` | Fine-tuned EMA checkpoint at iteration 5000. Use the immutable relocated config when the export embeds an unavailable VAE path. |
| 10k fine-tuned Edge HF EMA | `--checkpoint-path <EDGE_10K_EMA_HF> --target-adapter-path "$TARGET_ADAPTER" --weights-variant ema` | Fine-tuned EMA checkpoint at iteration 10000. |

A self-contained HF directory normally needs no external `--config-file`.
If a deployment requires an explicit Edge load config, pass the exact pinned
file; it becomes part of checkpoint identity and must remain unchanged across
resume. The fine-tuned exports still need the target adapter when their
`checkpoint.json.policy` contains only the training-native chunk/FPS/domain
fields. `--policy-profile-path` is reserved for an explicit complete profile;
duplicate sources must agree exactly.

The public `Cosmos3-Edge-hf` snapshot is the Transformers reasoner/vision
source, not a directly loadable Cosmos3 Omni action-policy export. Build the
formal Base target from the converted action DCP with an explicitly relocated
config and the public snapshot as the processor/vision source:

```bash
python -m cosmos_framework.scripts.export_model \
  --checkpoint-path <BASE_EDGE_DCP>/model \
  --config-file <RELOCATED_BASE_EDGE_CONFIG_JSON> \
  --no-use-ema-weights \
  --base-checkpoint \
  --vit-checkpoint-path <PUBLIC_BASE_EDGE_HF> \
  -o <BASE_EDGE_ACTION_HF>
```

`--base-checkpoint` is deliberately explicit: it accepts only an Edge action
model with regular weights and omits fine-tuning policy metadata, so the
LIBERO resolver classifies the export as Base and requires the target adapter.
The relocated config must change only unavailable processor/VAE paths and be
retained with source and derived hashes. Do not pass the public snapshot itself
to the action-policy job.

Completed exports publish regular files as `0644` and directories as `0755`
so a root container can hand the artifact to a non-root evaluator. Symlinks are
not followed; a permission-normalization failure removes `checkpoint.json`
rather than leaving a false completion marker.

The same rule applies to older fine-tuned exports: do not edit an immutable HF
checkpoint in place. If its load config names a storage path unavailable to the
evaluation worker, create a byte-preserving external config with only that path
relocated, retain a source/derived hash manifest, and pass it explicitly with
`--config-file`. The external config content becomes part of the checkpoint
fingerprint and must be identical on resume.

DCP is a source/debug format, not one of the formal three comparison targets.
A fine-tuned DCP requires its matching resolved `--config-file`, the adapter,
and `--weights-variant regular`. Direct EMA DCP loading through YAML/JSON is
unsupported; export EMA to HF.

The resolved adapter locks chunk 8 at 10 FPS, effective action dimension 10,
frame-wise relative translation, rot6d rotation, native pose frame,
`quantile_rot` statistics and SHA-256, JSON prompts, `agentview+wrist`
256-pixel inputs with 180-degree correction, OSC_POSE at 10 Hz, and `pm_one`
gripper semantics. The job deliberately exposes no runtime overrides for these
profile fields.

### 3.2 Locked simulator environment and preflight

Keep the CUDA model server and simulator in separate environments. Create the
simulator environment from the committed lockfile instead of cloning LIBERO and
manually mixing versions:

```bash
cd <REPO_ROOT>
UV_PROJECT_ENVIRONMENT=.venv-libero uv sync --frozen --group libero

SERVER_PYTHON="$PWD/.venv/bin/python"
RUNNER_PYTHON="$PWD/.venv-libero/bin/python"
```

The PyPI LIBERO wheel does not contain the simulator asset payload. Download
the official `jadechoghari/libero-assets` snapshot once into a persistent data
directory, then point a non-interactive site config at that directory. Use
absolute paths for both placeholders below. This initializer intentionally
refuses to overwrite an existing non-empty site config:

```bash
export LIBERO_CONFIG_PATH=<ABSOLUTE_LIBERO_CONFIG_DIR>
export LIBERO_ASSET_DIR=<ABSOLUTE_PERSISTENT_LIBERO_ASSET_DIR>
mkdir -p "$LIBERO_CONFIG_PATH" "$LIBERO_ASSET_DIR"
if [ -s "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  echo "Refusing to overwrite existing LIBERO config" >&2
  exit 1
fi
touch "$LIBERO_CONFIG_PATH/config.yaml"

"$RUNNER_PYTHON" -c \
  'import os; from libero.libero.utils.download_utils import download_assets_from_huggingface; download_assets_from_huggingface(os.environ["LIBERO_ASSET_DIR"])'

"$RUNNER_PYTHON" - <<'PY'
import os
from pathlib import Path

import yaml
from libero.libero import get_default_path_dict

asset_dir = Path(os.environ["LIBERO_ASSET_DIR"]).expanduser().resolve()
required = (
    "articulated_objects",
    "stable_scanned_objects",
    "turbosquid_objects",
    "stable_hope_objects",
    "scenes",
)
missing = [name for name in required if not (asset_dir / name).is_dir()]
if missing:
    raise SystemExit(f"incomplete LIBERO assets: {missing}")

config = get_default_path_dict()
config["assets"] = str(asset_dir)
config_path = Path(os.environ["LIBERO_CONFIG_PATH"]).expanduser().resolve() / "config.yaml"
config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
PY
```

On an offline worker, pre-stage the same official snapshot at
`LIBERO_ASSET_DIR` and run only the verification/configuration block. For an
existing pinned config, keep it unchanged and verify that its `assets` entry
resolves to a complete asset directory before preflight.

Every artifact path must be an absolute per-run directory below the output root
compiled into `cosmos_framework.evaluation.libero.artifacts.OUTPUT_ROOT`.
Discover it instead of copying a developer path into scripts:

```bash
CANONICAL_OUTPUT_ROOT="$("$SERVER_PYTHON" -c \
  'from cosmos_framework.evaluation.libero.artifacts import OUTPUT_ROOT; print(OUTPUT_ROOT)')"
PREFLIGHT_DIR="$CANONICAL_OUTPUT_ROOT/cosmos-framework-eval/libero/preflight/<RUN_ID>"

export LD_LIBRARY_PATH=''
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export EGL_DEVICE_ID=0

"$RUNNER_PYTHON" -m cosmos_framework.evaluation.libero.preflight \
  --output-dir "$PREFLIGHT_DIR" \
  --seed 0 \
  --render-gpu-device-id 0
```

Proceed only when `preflight.json` says `"status": "passed"`. It records the
installed LIBERO, robosuite, and MuJoCo versions; BDDL, init-state, and asset
directories; EGL details; adapter and action-stat hashes; and a task-0 render
smoke for `libero_spatial`, `libero_object`, `libero_goal`, and
`libero_10`. It does not certify `libero_90`.

### 3.3 Run one checkpoint

The job uses its own Python for the CUDA policy server and
`--runner-python` for the simulator. This 10k EMA smoke command makes every
sampling and rollout choice explicit:

Pass the virtual environment's launcher path itself (for example
`.venv-libero/bin/python`). The job canonicalizes its location without
dereferencing the launcher symlink, so Python retains that environment's
`sys.prefix` and site-packages.

```bash
CHECKPOINT=<EDGE_10K_EMA_HF>
RUN_DIR="$CANONICAL_OUTPUT_ROOT/cosmos-framework-eval/libero/runs/<CHECKPOINT_ID>/smoke/libero_spatial"

export COSMOS_EVAL_IMAGE=<EXACT_IMAGE_REFERENCE>
export COSMOS_EVAL_JOB_ID=<SCHEDULER_JOB_ID>

"$SERVER_PYTHON" -m cosmos_framework.evaluation.libero.job \
  --checkpoint-path "$CHECKPOINT" \
  --target-adapter-path "$TARGET_ADAPTER" \
  --weights-variant ema \
  --runner-python "$RUNNER_PYTHON" \
  --server-port 8000 \
  --num-steps 8 \
  --guidance 1.0 \
  --task-suite libero_spatial \
  --task-ids 0 \
  --trials 1 \
  --num-envs 1 \
  --action-horizon 8 \
  --seed 0 \
  --max-steps 20 \
  --warmup-steps 10 \
  --mujoco-gl egl \
  --render-gpu-device-id 0 \
  --request-timeout 120 \
  --output-dir "$RUN_DIR"
```

For the base and 5k exports, replace only the checkpoint-specific matrix flags.
The job normalizes checkpoint/config/profile/adapter paths before switching
subprocess working directories. It rejects occupied ports, malformed or stale
handshakes, profile/fingerprint mismatches, invalid action payloads, a
zero-episode run, unresolved infrastructure failures, and non-zero child exits.

The action-policy server disables the framework's generic text/video
guardrails. Those guardrails protect generative-media endpoints and would add
an unrelated checkpoint download before a fixed simulator policy can start.
For local HF checkpoints, configuration files already inside the checkpoint
are fingerprinted exactly once even when the loader discovers them
automatically. A configuration supplied from outside the checkpoint remains an
explicit part of the immutable fingerprint.

The `--seed` value is a non-negative signed-63-bit base seed. Protocol
`cosmos-libero-eval-v2` derives each logical policy seed from canonical JSON of
the complete suite/task/trial/decision identity and SHA-256. It deliberately
does not truncate, take modulo, or fold that logical identity to uint32. The
server advertises, and the runner requires, the exact
`sampling_seed_contract` value
`sha256-canonical-json-first64-mask63-v1+mt19937-uint32-identity-or-le-u32-pair-v1`.
At the NumPy MT19937 boundary, a logical seed through `2**32 - 1` remains the
same scalar integer, preserving its historical `RandomState` stream; a larger
seed becomes the lossless low-word-first pair
`[seed & 0xffffffff, seed >> 32]`. Simulator episode seeds use a separately
domain-separated uint32 derivation and are recorded as `episode_seed` in each
episode and infrastructure-attempt record. The schema-v2 manifest stores both
`base_seed` and the handshake contract, so a seed-contract or protocol change
requires a new run directory.

Use this promotion ladder; each checkpoint, stage, and suite gets a new run
directory:

| Stage | Suites and tasks | Trials | Parallel envs | Max steps | Gate |
| ----- | ---------------- | ------ | ------------- | --------- | ---- |
| Smoke | one primary suite, `--task-ids 0` | 1 | 1 | 20 | Server loads, handshake matches, one terminal episode, finite `[8,10]` actions, no infrastructure error. |
| Pilot | one primary suite, `--task-ids 0,1` | 3 | 2 | 0 (suite default) | Deterministic seeds, reset/step stability, sensible latency and memory, complete artifacts. |
| Full | each of the four primary suites, empty `--task-ids` | 50 per task | Start at 8 | 0 (suite default) | Separate sealed run for every checkpoint × suite; aggregate only identical profile/fingerprint and sampling settings. |

For full evaluation, invoke the same job command once per suite:

```bash
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  RUN_DIR="$CANONICAL_OUTPUT_ROOT/cosmos-framework-eval/libero/runs/<CHECKPOINT_ID>/full/$SUITE"
  # Re-run the command above with:
  #   --task-suite "$SUITE" --task-ids "" --trials 50
  #   --num-envs 8 --max-steps 0 --output-dir "$RUN_DIR"
done
```

A run directory contains `manifest.json`, `episodes.jsonl`,
`infra_errors.jsonl`, `metrics.json`, `_SUCCESS`, `job.log`,
`server.log`, `runner.log`, and `server_runtime/`. Re-running the exact
same request resumes stable episode IDs. A changed checkpoint, config, profile,
sampling request, or rollout plan requires a new run directory. Successfully
retried historical infrastructure attempts remain auditable, while terminal
`overall.infra_errors` must be zero.

### 3.4 Scheduled execution

The server and runner communicate over loopback, so use one replica and one GPU
per job. Pin the exact image and workspace mount and record both in provenance.
Verify the installed `rlaunch` or `rjob` help before submission.

```bash
rlaunch \
  --gpu=1 --cpu=16 --memory=131072 \
  --charged-group=<QUOTA_GROUP> \
  --positive-tags=<GPU_TAG> \
  --image=<EVAL_IMAGE> \
  --mount=<WORKSPACE_MOUNT_URI>:<WORKSPACE_MOUNT_POINT> \
  -- bash
```

```bash
rjob submit \
  --name <JOB_NAME> \
  --group <RESOURCE_GROUP> \
  --charged-group <QUOTA_GROUP> \
  --private-machine group \
  --preemptible no \
  --image <EVAL_IMAGE> \
  --replica 1 --gpu 1 --cpu 16 --memory 131072 \
  --positive-tags <STORAGE_COMPATIBILITY_TAG> \
  --custom-resources <SHARED_STORAGE_RESOURCE>=1 \
  --mount <WORKSPACE_MOUNT_URI>:<WORKSPACE_MOUNT_POINT> \
  --share-host-shm true \
  --restart-policy never \
  --backoff_limit 1 \
  --auto-delete-duration 168h \
  -- bash -lc '<source environment; export EGL, LIBERO, COSMOS_EVAL_IMAGE, and COSMOS_EVAL_JOB_ID; run preflight or job>'
```

Treat `--positive-tags` as a hard selector and use only labels verified by a
successful compatible job; an inferred accelerator-name tag can exclude the
very workers it was meant to select. Keep guaranteed jobs non-preemptible with
the installed CLI's `--preemptible no` spelling. After submission, inspect the
job plus its concrete replica with `rjob get <JOB_NAME>`,
`rjob events <REPLICA_NAME> --replica`,
`rjob logs replica <REPLICA_NAME> -n 200`, or
`rjob logs job <JOB_NAME> -n 200`. Do not launch multiple replicas against one
run directory. Size CPU and memory from the pilot when increasing `--num-envs`.

### 3.5 Nano legacy client

`cosmos_framework/simulation/libero/closed_loop_eval.py` remains a Nano
legacy path for previously trained Nano checkpoints. Its 20 FPS assumptions,
manual policy overrides, response protocol, and relative output layout are not
the Cosmos3-Edge benchmark contract. Use it only to reproduce legacy Nano
results; use `preflight` + `job` + `runner` for every Edge result.

## 4. Heads-up

- **Lower-memory training GPUs** — reduce the per-rank training batch with
  `--opts dataloader_train.max_samples_per_batch=64` and scale replication or
  accumulation to preserve global batch 2048.
- **Base Edge is zero-shot** — the target adapter makes the action space
  interpretable but does not make the base model LIBERO-trained. Keep its
  metrics separate from fine-tuned checkpoints.
- **DCP EMA** — direct EMA evaluation through a YAML/JSON DCP config is
  unsupported. Export EMA to HF.
- **Profile parity is fail-fast** — do not repair camera order, image rotation,
  gripper sign, control frequency, normalization, prompt format, or action
  dimension with ad hoc runtime flags. Fix the checkpoint metadata or explicit
  adapter/profile.
- **Primary-suite guarantee** — preflight certifies the four 10-task suites
  listed above, not `libero_90`.
- **Canonical output only** — arbitrary relative or external output directories
  are rejected. Use a distinct absolute subdirectory for each immutable
  checkpoint × suite × stage request.
