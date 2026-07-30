# Cosmos3-Edge LIBERO Fine-Tuning Development Record

## Status

- Branch: `feature/cosmos3-edge-libero`
- Base branch: `main`
- Current phase: DEV-0004 complete; DEV-0005 Edge recipe next
- Full training: not started

## Objective

Fine-tune Cosmos3-Edge as an action policy on the merged LIBERO LeRobot
dataset available on the PJLAB H cluster. The implementation remains separate
from the existing Cosmos3-Nano LIBERO recipes and uses the Edge model's
pretrained action heads.

## Development boundaries

- Generic, reusable user instructions remain in the repository's existing
  `docs/` and `examples/` locations.
- Cluster paths, rjob commands, implementation decisions, experiment output,
  failures, and measurements are maintained in this document.
- Dataset, model weights, W&B artifacts, and rjob logs are not committed.

## Dataset

- Root:
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/data/lerobot-libero`
- LeRobot version: v3.0
- Episodes: 1693
- Frames: 273465
- Tasks: 40
- Native FPS: 10
- External camera: `observation.images.image`
- Wrist camera: `observation.images.image2`
- Stored action: 7D `[dpos(3), axisangle(3), gripper(1)]`
- Gripper convention: approximately `[-1, 1]` (`pm_one`)

The two video features were visually checked from matching frames. `image`
contains the third-person view and `image2` contains the wrist-mounted view.

## Action chunk decision

The official 20 FPS LIBERO recipe uses 16 actions, covering 0.8 seconds. This
dataset is 10 FPS, so the Edge recipe uses 8 actions and 9 observation frames
to preserve the same 0.8-second physical prediction horizon. Eight also
satisfies the loader requirement that chunk length be divisible by four.

## Action normalization

The existing bundled LIBERO statistics were generated for a 20 FPS conversion
with a different gripper convention and must not be reused. DEV-0004 adds a
deterministic statistics CLI that:

1. reads every action parquet in sorted order;
2. converts axis-angle rotation to the same rot6d representation used by the
   training dataset;
3. computes mean, standard deviation, min, max, q01, and q99 over the resulting
   10D actions; and
4. fingerprints the action parquet and small metadata files with SHA-256.

The generated statistics will be committed as a small reproducibility asset;
the source dataset remains outside Git.
The 273465-frame output fingerprint is
`4a64eacab21f27ed949dce5a024d028fe38101b2a69ee2a6ae25eac11a4e84a1`.

## Planned Edge recipe

- Experiment: `action_policy_libero_all_edge_10fps`
- Model base: `EDGE_MODEL_CONFIG`
- Action chunk: 8
- Native FPS: 10
- Camera layout: external view left, wrist view right
- Parallelism: single-node FSDP, shard degree 8
- Target global batch: 2048
- Optimizer LR: `5e-5`, with 5x action-module multipliers
- Training length: 5000 iterations
- Checkpoint interval: 500 iterations
- Tracking: local logs plus offline W&B

The recipe will load the renewed Cosmos3-Edge action-head weights. It will
skip only `net_ema` during base initialization and fail if the expected action
head keys are absent.

## Cluster execution

Every cluster session must source the shared `.bashrc` and enable `proxy_on`.
GPU validation and training require an rjob allocation; no GPU workload runs
on the development pod.

Planned persistent resources:

- Edge DCP:
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/model/Cosmos3-Edge-dcp`
- Wan VAE:
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/model/wan22_vae/Wan2.2_VAE.pth`
- Training root:
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/model/cosmos3-edge-libero-all-10fps`

The rjob command, job IDs, selected image/SKU, effective batch fallback, loss
summary, checkpoint paths, and export validation will be added after execution.

## Implementation log

### DEV-0004

- Parameterize the wrist camera feature while retaining the NVIDIA default.
- Add deterministic 7D axis-angle to 10D rot6d statistics generation.
- Add unit tests for camera resolution, the 9-frame/8-action window, and
  statistics determinism.
- Generate and validate the dataset-specific statistics file.
- Validation: six focused tests passed; Ruff and format checks passed.
- Real-data validation loaded all 1693 episodes and produced a finite sample
  with video shape `[3, 9, 256, 512]` and action shape `[8, 10]`.
- The GPU-less development pod's default TorchCodec path cannot load
  `libnppicc.so.12`; the CPU integration check therefore explicitly selected
  LeRobot's PyAV backend. The rjob smoke test must verify the default TorchCodec
  path in the CUDA container before full training.

### DEV-0005

Pending: add the Edge experiment, TOML, launcher, registration, user-facing
documentation, and configuration tests.

### DEV-0006

Pending: record rjob smoke training, resume validation, full training, export,
and real-sample forward validation.
