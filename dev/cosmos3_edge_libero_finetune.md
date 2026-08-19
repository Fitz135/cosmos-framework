# Cosmos3-Edge LIBERO Fine-Tuning Development Record

## Status

- Branch: `feature_libero_finetune`
- Base branch: `feature/cosmos3-edge-libero`
- Current phase: DEV-0021 sealed three-checkpoint LIBERO evaluation and
  cross-suite reporting.
- Original 5k training: `cosmos3-edge-libero-full-b128-2361150` succeeded.
- 10k continuation output: complete `iter_000010000/model` DCP under the
  canonical output root.
- Formal checkpoints: base Edge HF regular, 5k fine-tuned HF EMA, and 10k
  fine-tuned HF EMA.
- Locked simulator preflight: passed for the four primary suites.
- Policy rollout status: the three `v8` smokes and three-checkpoint pilot
  passed their artifact/infra gates. The formal `full-v1-45ebff5` matrix is
  complete: all 12 RJobs succeeded, all 12 suite runs are sealed, and all
  6000 terminal episodes have zero unresolved infrastructure errors.
- Previous real-sample action check: finite `[1, 8, 10]` output from the 5k
  export.

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

## Edge recipe

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

The recipe loads the renewed Cosmos3-Edge action-head weights. It skips only
`net_ema` during base initialization and fails if the expected action-head keys
are absent.

The experiment keeps the public `EDGE_MODEL_CONFIG` immutable by deep-copying
it before recipe-specific changes. It disables secondary diffusion-expert
weight initialization because the DCP is the single source of model weights,
retains full activation checkpointing and the 45056-token limit, and trains
`k_norm_und_for_gen` alongside the generation and action modules. It does not
change the Nano recipes, the shared Edge baseline, the inference server, or
closed-loop evaluation.

The paired TOML selects single-node FSDP8, bf16, batch 128 per rank,
gradient accumulation 2, LR `5e-5`, warmup 500, 5000 iterations, checkpoint
interval 500, and W&B offline mode. The launcher validates the merged dataset's
10 FPS metadata, `image2` camera feature, action statistics, base DCP, and Wan
VAE before starting. OOM fallbacks preserve global batch 2048 with `64/4` and
then `32/8`.

## Cluster execution

Every cluster session must source the shared `.bashrc` and enable `proxy_on`.
GPU validation and training require an rjob allocation; no GPU workload runs
on the development pod.

Persistent resources:

- Edge DCP:
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/model/Cosmos3-Edge-dcp`
- Wan VAE:
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/model/wan22_vae/Wan2.2_VAE.pth`
- Training root:
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/model/cosmos3-edge-libero-all-10fps`

Preparation used the image
`registry.h.pjlab.org.cn/ailab-llmrazor/xtuner_tmp:pt28_20260303_f2adb47`,
the `llmagent_gpu` charged group, a private group machine, one H200, 32 CPU
cores, 131072 MiB memory, `brainpp.cn/fuse=1`, and the
`gpfs://gpfs2/intern-pretrain-shared02` mount. The reusable command shape was:

```bash
source /etc/profile.d/ssh-init.sh
rjob submit \
  --name cosmos3-edge-libero-prepare \
  --charged-group llmagent_gpu \
  --private-machine group \
  --image registry.h.pjlab.org.cn/ailab-llmrazor/xtuner_tmp:pt28_20260303_f2adb47 \
  --cpu 32 --gpu 1 --memory 131072 \
  --custom-resources brainpp.cn/fuse=1 \
  --mount gpfs://gpfs2/intern-pretrain-shared02:/mnt/shared-storage-gpfs2/intern-pretrain-shared02 \
  --set-env COSMOS_LIBERO_MODE=prepare -- \
  bash /mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/.cache/cosmos3_edge_libero_job.sh
```

Preparation job history:

- `cosmos3-edge-libero-prepare-56070503`: failed because the allocated H200
  node could not download the VAE through the proxy.
- `cosmos3-edge-libero-prepare-r2-1726033`: failed because a local HF
  checkpoint without the registered Edge YAML selected the wrong root config.
- `cosmos3-edge-libero-prepare-r3-27977290`: failed because the registered
  config still tried to download the Edge processor.
- `cosmos3-edge-libero-prepare-r4-77213900`: stopped after the local processor
  fix exposed an attempted Wan2.2 VAE Hub download.
- `cosmos3-edge-libero-prepare-r5-96028663`: verified the local VAE path was
  injected, then failed because the inherited bucket name still selected the
  object-store backend and required `credentials/gcp_training.secret`.
- `cosmos3-edge-libero-prepare-r6-630658`: succeeded. It ran on an H200 with
  PyTorch `2.10.0+cu128`, loaded both processor and VAE from GPFS, wrote the
  DCP checkpoint, and completed the structured-TOML training dry-run.
- `cosmos3-edge-libero-dryrun-r7-95807184`: rjob returned success, but the log
  exposed an OmegaConf `ConfigKeyError` because the first local-processor
  implementation removed structured tokenizer fields. The result was rejected
  as validation evidence and the recipe was corrected to preserve those fields
  with `None` values.
- `cosmos3-edge-libero-dryrun-r8-63642809`: succeeded without serialization
  errors. The resolved model and dataset tokenizer configs both used the local
  Edge HF snapshot with `repository=None`.

The converter now supports `--vae-path` and redirects compatible
Cosmos3-Edge snapshots to their bundled processor. Supplying a local VAE also
clears the object-store bucket and credential fields, so conversion requires no
Hub or internal object-store access after the artifacts are staged.
The training recipe accepts `COSMOS3_EDGE_PROCESSOR_PATH`; PJLAB jobs set it to
the staged Edge HF snapshot so model and dataset processor construction also
remain offline.

The DEV-0005/DEV-0006 effective dry-run confirmed FSDP shard degree 8, max batch 128, gradient
accumulation 2, LR `5e-5`, warmup 500, max iteration 5000, checkpoint interval
500, W&B offline, full dataset split, 10 FPS, action chunk 8, `image2`, and the
dataset-specific normalizer. DEV-0009 changes future periodic saves to every
2000 steps. The trainer's final-save guard means a default 5000-step run writes
checkpoints at 2000, 4000, and 5000.
Checkpoint-aligned qualitative visualization is part of the Edge recipe. The
trainer writes the DCP before invoking `on_training_step_end`, where an
`EveryNDrawSample` callback uses EMA weights to generate one joint WAM rollout
every 2000 optimizer steps. It saves a 10 FPS, three-row local video containing
the prediction, clean-latent VAE reconstruction, and ground truth; offline W&B
receives first/middle/last preview frames. The default uses guidance 1.0 and 8
denoising steps, saves only rank 0, and does not write to object storage. This
keeps the output aligned with periodic checkpoints while bounding the pause.
The final checkpoint at step 5000 is written after the training loop and does
not fire the `EveryN` callback; its visualization remains an explicit post hoc
operation.

The completed `cosmos3-edge-libero-full-b128-2361150` process was launched
before this callback was added and could not hot-load it. It was therefore not
interrupted only to enable visualization. The callback applies to future
resumes or new runs; complete checkpoints can also be visualized post hoc
without changing their weights.

Generic behavior and override syntax are mirrored in
`docs/action_policy_libero_posttrain.md`; PJLAB execution evidence and output
paths remain here.


Training jobs use one private 8×H200 node, 120 CPU cores, 1800000 MiB host
memory, host networking/shared memory, gang start, eight shared RDMA devices,
one Mellanox RDMA device, `brainpp.cn/fuse=1`, and the same GPFS mount:

```bash
source /etc/profile.d/ssh-init.sh
rjob submit \
  --name cosmos3-edge-libero-<mode>-b128 \
  --charged-group llmagent_gpu \
  --private-machine group \
  --image registry.h.pjlab.org.cn/ailab-llmrazor/xtuner_tmp:pt28_20260303_f2adb47 \
  --cpu 120 --gpu 8 --memory 1800000 \
  --share-host-shm=True --host-network=true --gang-start=true \
  --custom-resources \
    rdma/mlnx_shared=8 mellanox.com/mlnx_rdma=1 brainpp.cn/fuse=1 \
  --mount gpfs://gpfs2/intern-pretrain-shared02:/mnt/shared-storage-gpfs2/intern-pretrain-shared02 \
  --set-env \
    COSMOS_LIBERO_MODE=<smoke|resume|full> \
    RESUME_CHECKPOINT=<required-for-resume-and-full> \
    PER_RANK_BATCH=128 GRAD_ACCUM=2 -- \
  bash /mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/.cache/cosmos3_edge_libero_job.sh
```

## Strict Cosmos3-Edge LIBERO evaluation (DEV-0015)

### Runtime contract

The strict Edge path is
`cosmos_framework.evaluation.libero.preflight` →
`cosmos_framework.evaluation.libero.job` →
`cosmos_framework.evaluation.libero.runner`. The job resolves checkpoint
identity locally, starts the CUDA policy server, requires the server's
versioned `/info` profile and immutable checkpoint fingerprint to match,
starts the runner with the separate LIBERO Python, and terminates complete
server and runner process groups on success, error, KeyboardInterrupt, or
SIGTERM.

The committed adapter at
`cosmos_framework/evaluation/libero/profiles/edge_libero_target_adapter.json`
locks 10 FPS, chunk 8, 10D frame-wise-relative rot6d actions, native pose
frame, `quantile_rot` stats SHA, JSON prompts, agentview+wrist at 256 pixels
with rotation correction, OSC_POSE at 10 Hz, and `pm_one` gripper semantics.
Sampling is separate and explicitly recorded with
`--num-steps 8 --guidance 1.0`.

Protocol `cosmos-libero-eval-v2` also locks sampling-seed semantics. The runner
derives a logical policy seed from the complete
`(base_seed, task_suite, task_id, trial_id, decision_index)` identity by
canonical JSON plus SHA-256, takes the first eight digest bytes as big-endian,
and clears the sign bit. This signed-63-bit identity is not truncated or folded
to fit NumPy. At the RNG boundary, values through `2**32 - 1` remain scalar so
their existing `numpy.random.RandomState` streams stay byte-identical; larger
values are passed losslessly as the low 32-bit word followed by the high 32-bit
word. The exact advertised contract is
`sha256-canonical-json-first64-mask63-v1+mt19937-uint32-identity-or-le-u32-pair-v1`.
The runner requires an exact `/info.sampling_seed_contract` match and copies it
to the immutable schema-v2 manifest. Simulator randomness remains separately
domain-separated: each episode and infrastructure-attempt record includes its
slot-independent uint32 `episode_seed`.

Protocol `cosmos-libero-eval-v3` additionally locks the gripper boundary
adapter as `pm-one-finite-clamp-v1`. `pm_one` describes the raw command
convention; it does not guarantee that an unconstrained diffusion sample lies
inside the actuator range. The committed `quantile_rot` statistics use
`global_raw.q01[-1] = -1` and `global_raw.q99[-1] = 1`, so the gripper channel's
inverse affine transform has offset 0 and scale 1. A model-space excursion is
therefore unchanged by server denormalization. The explicit adapter requires
finite values, preserves in-range values, and clamps only the gripper channel
to `[-1, 1]` before `env.step`, matching the legacy `pm_one` behavior. Shape,
finiteness, pose conversion, and every non-gripper channel remain fail-fast.
Each episode and infrastructure-attempt record stores
`gripper_adapter_contract` plus a `gripper_adapter_telemetry` object containing
`raw_min`, `raw_max`, `generated_value_count`,
`clipped_generated_value_count`, `clipped_generated_value_rate`, and
`max_abs_overshoot`. `raw_min` and `raw_max` are measured after server-side
denormalization and pose conversion but before the simulator-boundary clamp.
The counts cover every action in each complete generated chunk processed
before termination, including the unexecuted tail when `action_horizon` is
shorter than the chunk. Metrics schema 2 places the terminal-episode aggregate
under `gripper_adapter`; infrastructure attempts remain auditable but are
excluded. The aggregate rate is
`sum(clipped_count) / sum(generated_count)`, never the mean of per-episode
rates. The exact contract is part of the strict handshake and immutable
schema-3 manifest.

### Formal checkpoint matrix

| Target | Current PJLab path | Required flags and interpretation |
| ------ | ------------------- | --------------------------------- |
| Base Edge action HF regular | `/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework-eval/checkpoints/cosmos3-edge-base-regular-hf` | Exported from the Base action DCP with `--base-checkpoint`; adapter plus `--weights-variant regular`; explicit zero-shot diagnostic only. |
| 5k fine-tune HF EMA | `/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/model/cosmos3-edge-libero-all-10fps/export-final-iter5000` | External config `.../output/cosmos-framework-eval/checkpoints/cosmos3-edge-libero-5k-ema-runtime/config.json`, adapter, and `--weights-variant ema`; formal 5k fine-tuned target. |
| 10k fine-tune HF EMA | `/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework-eval/checkpoints/cosmos3-edge-libero-10k-ema` | Adapter plus `--weights-variant ema`; formal 10k fine-tuned target. |

The public `/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/model/Cosmos3-Edge-hf`
snapshot is used only as the Base export's processor and vision source. It is a
native Transformers `Cosmos3EdgeForConditionalGeneration` snapshot, not a
Cosmos3 Omni action-policy checkpoint, and must not be passed directly to the
LIBERO action server. The Base action source is
`/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/model/Cosmos3-Edge-dcp/model`;
its relocated config and provenance manifest live in
`/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework-eval/checkpoints/cosmos3-edge-base-regular/`.
The relocated config source/derived SHA-256 values are
`ce5c5ee6f18262b84d98033a1a754dfe5bef9d9dab535f8ffc8bd0ad4fdff993`
and `199490774b7bf22a2ce64414153a42179d93804f92ee27a4d94d55ed4d765a05`.
Rjob `c3-libero-base-export-0819-v1` produced the self-contained HF export
from commit `1c6382b`; `c3-libero-base-perms-0819-v1` performed the one-time
permission repair before resolver acceptance.

Both fine-tuned exports contain only their selected EMA weights. Their
`checkpoint.json.policy` records training-native policy fields; the target
adapter supplies the remaining evaluation semantics. HF fingerprints include
safetensors, load-critical configuration and processor/tokenizer assets, plus
an explicit external config when one is supplied.

The 5k export's embedded config retained one unavailable GPFS2 VAE path. The
original checkpoint remains unchanged; its external runtime config relocates
only that path to the existing GPFS1 VAE. The source and derived config hashes
are `6f96c4b50598516df70502d5378f296da89aab8bd205d10a8e4b00c26804eede`
and `cbabe664997adfc7b9326276a2cb7bd1b94d611692ecf5de63322db7d868ebe0`;
the exact replacement is recorded in the sibling `source_manifest.json`.

DCP remains a source/debug format rather than a formal comparison target. The
10k source DCP is
`/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework-training/cosmos3-edge-libero-10k-b128-fa3/cosmos3_action_libero/action_sft/action_policy_libero_all_edge_10fps_10k/checkpoints/iter_000010000/model`;
its resolved config is the sibling run's `config.yaml`. DCP evaluation
requires that exact config, the adapter, and regular weights. Direct EMA DCP
evaluation through YAML/JSON is rejected, which is why the formal 5k and 10k
EMA targets are HF exports.

### Resolved identity evidence

The strict resolver resolved the final three comparison targets after the Base
export and 5k runtime-config relocation on 2026-08-19:

| Target | Checkpoint fingerprint | Policy profile hash |
| ------ | ---------------------- | ------------------- |
| Base Edge action HF regular | `99725010794b9248cdc23f004c74793ac1c7dcaf135e2eae153e423d6b9b5907` | `56a0a43c2775baf6c20aac89a96ab1b305edce1ec685c0447b596243f43a354d` |
| 5k fine-tune HF EMA plus external config | `7b09f1edfbdc7f50a15bb86dda3cdac7b351b4595f2ad693193aeaa82bea5459` | `fb000cae6eee2fddb7c15b374f920bae50a5dd6fc6478688f4430cf19328cda9` |
| 10k fine-tune HF EMA | `ca83c3545605e368c30a8a4a89188294d6b6f7ff6d85e73b11cfd8c1dc4a2921` | `b3284d55e4ec80e2ff4c4126b55be5cc75c66aa571509d1c5f1ee90d53c32783` |

These checkpoint fingerprints and profile hashes also remain the policy
identities for v3. DEV-0020 adds the gripper adapter as a separate exact
handshake and manifest contract rather than changing checkpoint metadata or
target semantics. The protocol/manifest change still forbids resuming a v2 run
directory under v3.

The existing 10k export and the new Base export required one-time root-worker
permission repair for top-level safetensor shards created as `0600`.
DEV-0018 makes future exports deterministic and cross-user readable: regular
files are `0644`, directories are `0755`, symlinks are not followed, and a
failed normalization removes the `checkpoint.json` completion marker.

### Locked simulator and assets

The repository-local simulator environment is
`/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/project/cosmos-framework/.venv-libero`,
resolved from `uv.lock` with the `libero` dependency group. Confirmed
versions are LIBERO 0.1.1, robosuite 1.4.0, and MuJoCo 3.3.2. The
425 MiB asset payload is stored separately at
`/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/data/libero-assets` and
exposed to LIBERO through the symlink
`/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/project/cosmos-framework/.venv-libero/lib/python3.13/site-packages/libero/libero/assets`.
That link targets the external data directory; the wheel does not bundle this
asset payload. The BDDL files and init states remain in the installed LIBERO
package. The
non-interactive config is
`/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework-eval/runtime/libero-config/config.yaml`
and points its asset entry at the package path above, which resolves through
the symlink to the external asset directory.

The confirmed manifest is
`/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework-eval/libero/preflight/20260818-cpu-egl-diagnostic-v7/preflight.json`.
It records `status=passed`, `MUJOCO_GL=egl`, loaded `libEGL.so.1`, adapter
and stats hashes, and successful task-0 rendering after 10 warmup steps for
`libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`.
`libero_90` is outside this preflight guarantee.

### Canonical output and promotion plan

All evaluation artifacts remain below
`/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output`:

```text
output/cosmos-framework-eval/
  checkpoints/cosmos3-edge-libero-10k-ema/
  runtime/libero-config/config.yaml
  libero/preflight/<run-id>/preflight.json
  libero/runs/<checkpoint-id>/<smoke|pilot|full>/<suite>/
  libero/reports/<run-id>/summary.json
```

Each immutable run directory holds `manifest.json`, `episodes.jsonl`,
`infra_errors.jsonl`, `metrics.json`, `_SUCCESS`, `job.log`,
`server.log`, `runner.log`, and `server_runtime/`. Exact reruns resume by
stable episode ID. Historical infrastructure attempts remain auditable after a
successful retry; success requires evaluated terminal episodes and
`overall.infra_errors=0`.

All promotion stages are complete:

| Stage | Executed scope | Result |
| ----- | -------------- | ------ |
| Smoke | each checkpoint, `libero_spatial` task 0, 1 trial, 1 env, 20 steps | Three `smoke-v8-45ebff5` runs sealed with matching identity, finite `[8,10]` actions, schema-2 metrics, `_SUCCESS`, and zero infra errors. The deliberately short horizon produced no task success and was not used as a policy-quality estimate. |
| Pilot | each checkpoint, `libero_spatial` tasks 0 and 1, 3 trials, 2 envs, canonical max steps | Three `pilot-v1-45ebff5` runs sealed: Base 0/6, 5k EMA 5/6, and 10k EMA 6/6, all with zero infra errors. |
| Full | three HF checkpoints × four primary suites × 10 tasks × 50 trials, 8 envs | All 12 `full-v1-45ebff5` RJobs succeeded and sealed 500 episodes each: 6000/6000 terminal episodes, zero infra errors. |

Base metrics will be labeled zero-shot and never mixed with fine-tuned
results. The 5k and 10k EMA results remain separate checkpoint fingerprints.

### First scheduled H200 smoke and startup corrections

The initial 2026-08-19 retry used three independent one-GPU jobs. An explicit
`h200` positive tag excluded two machines that the scheduler identified as
H200s, so the still-empty `v2` jobs were stopped and replaced with the proven
private-pool contract: `group=evoagi_gpu`, `charged-group=evoagi_gpu`,
`private-machine=group`, `preemptible=no`, `feature/gpfs=yes`, the GPFS1 mount,
and `brainpp.cn/fuse=1`. All three `v3` jobs then reached H200 workers and their
locked GPU preflights passed all four primary suites.

The `v3` policy servers failed before model inference for one common reason:
the generic inference defaults enabled media guardrails and attempted to
download `nvidia/Cosmos-Guardrail1`, which timed out. The run also exposed a
second fail-fast issue before handshake: the loader-discovered HF
`config.json` was already part of the checkpoint metadata set but was hashed a
second time as an external resolved config, producing a different server
profile hash from the job resolver. DEV-0016 disables media guardrails for the
robot-policy endpoint and de-duplicates checkpoint-internal resolved configs;
truly external configs remain fingerprinted. The failed `v3` directories are
diagnostic evidence only, and the corrected evaluation must use new run IDs.

The `v4` retry proved that disabling guardrails was effective: the 10k server
loaded the local model without a guardrail download. It also exposed two
remaining identity errors before any rollout. First, the server fingerprinted
the framework's implicit `base/config.py` before the HF loader replaced it with
the checkpoint-local config, so its handshake differed from the job resolver.
Second, the supposed Base policy path was actually the public Transformers
reasoner/vision snapshot and failed because it has no Cosmos3 Omni
`vlm_config.tokenizer` node. DEV-0017 excludes only the implicit default config
from policy identity, preserves every explicit external config, adds an
explicit regular-only Base export mode, and replaces the invalid Base target
with a self-contained action HF export derived from the Base DCP. The `v4`
directories remain diagnostic evidence and are not promotable.

The `v5` retry validated the remaining checkpoint/startup contracts. All three
jobs acquired H200s and passed the four-suite GPU preflight. Base and 10k loaded
their models and matched the strict server handshake, including the new Base
identity. Their runners then failed before environment creation because the job
resolved `.venv-libero/bin/python` through its symlink to the bare uv base
interpreter, which lacked the virtualenv site-packages. The 5k server separately
failed on its embedded unavailable GPFS2 VAE path. DEV-0018 preserves the
virtualenv launcher symlink while still making relative paths absolute, and
uses the immutable external 5k config described above. The failed `v5`
directories are diagnostic evidence only; the corrected smoke uses a new ID.

The `v6` retry from commit `241aa98` proved the corrected runner and checkpoint
contracts end to end up to the first model call. Jobs
`c3-libero-base-smoke-0819-v6`, `c3-libero-5k-smoke-0819-v6`, and
`c3-libero-10k-smoke-0819-v6` each acquired an H200, passed the locked
four-suite GPU preflight, loaded the intended policy, matched the exact
profile/fingerprint handshake in the identity table above, retained the
LIBERO virtualenv, and created its manifest and infrastructure journal. Their
first `/predict_batch` request used the same deterministic logical seed
`7221137112376841976`, which the pre-fix model forwarded directly to
`numpy.random.RandomState`; NumPy rejected it with `ValueError: Seed must be
between 0 and 2**32 - 1`, and the runner recorded an HTTP-400 policy-server
infrastructure error. None of these directories contains promotable metrics or
`_SUCCESS`:

```text
libero/runs/base-edge-action-hf-regular/smoke-v6-241aa98/libero_spatial/
libero/runs/edge-5k-hf-ema/smoke-v6-241aa98/libero_spatial/
libero/runs/edge-10k-hf-ema/smoke-v6-241aa98/libero_spatial/
```

DEV-0019 retains the collision-resistant signed-63-bit logical identity and
adds the versioned, lossless MT19937 key adaptation described in the runtime
contract. Because the protocol and manifest identity changed from v1 to v2,
the `v6` directories must never be resumed. The later `v7` Base, 5k, and 10k
jobs therefore used new job and run IDs; their outcome is recorded below.

The `v7` retry from commit `b0a6cf3` proved that seed adaptation is effective.
Jobs `c3-libero-base-smoke-0819-v7`, `c3-libero-5k-smoke-0819-v7`, and
`c3-libero-10k-smoke-0819-v7` acquired H200 workers, passed the four-suite GPU
preflight, loaded and handshook their intended v2 profiles, and completed the
first batch-size-one UniPC inference with eight denoising steps. The measured
server inference times were 23.644 s for Base, 22.707 s for 5k, and 23.691 s
for 10k; none reproduced the v6 NumPy seed error.

All three returned action chunks, then failed in the runner's
`policy_action_adapter` before the first environment step because at least one
finite gripper value lay materially outside `[-1, 1]`. Each infrastructure
record has `decisions=1`, `steps=0`, and `episode_seed=2218099160`; none has
`metrics.json` or `_SUCCESS`. The v7 directories are diagnostic and
non-promotable:

```text
libero/runs/base-edge-action-hf-regular/smoke-v7-b0a6cf3/libero_spatial/
libero/runs/edge-5k-hf-ema/smoke-v7-b0a6cf3/libero_spatial/
libero/runs/edge-10k-hf-ema/smoke-v7-b0a6cf3/libero_spatial/
```

The failure is an adapter-contract defect rather than a normalization mismatch.
For the gripper dimension, the selected `global_raw` q01/q99 values are exactly
`-1/+1`, making denormalization the identity. The Nano legacy evaluator already
defines `pm_one` as a finite pass-through with `[-1, 1]` clamping. DEV-0020
makes that behavior explicit, versioned, and auditable instead of silently
loosening the strict adapter. Protocol v3/schema 3 requires fresh `v8` job and
run IDs. The resulting `v8` smoke, pilot, and full formal matrix are complete
and recorded below.

### Completed v8 smoke and pilot

Commit `45ebff509696730429d028e7b8cdc2a7bcf75584` was clean in every
schema-3 manifest. The three `smoke-v8-45ebff5` RJobs all succeeded, each run
sealed one 20-step terminal episode with schema-2 metrics, `_SUCCESS`, and no
infrastructure error. Each policy completed three real 8-step predictions.
The task outcome was false because the deliberately short smoke horizon is a
startup gate, not a policy-quality measurement.

| Checkpoint | Smoke RJob | Gripper clipped/generated | Raw range | Max overshoot |
| ---------- | ---------- | ------------------------- | --------- | ------------- |
| Base Edge regular, zero-shot | `c3-libero-base-smoke-0819-v8` | 7/24 (29.1667%) | `[-1.706365, 1.351510]` | 0.706365 |
| 5k Edge EMA | `c3-libero-5k-smoke-0819-v8` | 6/24 (25.0000%) | `[-1.005497, -0.985256]` | 0.005497 |
| 10k Edge EMA | `c3-libero-10k-smoke-0819-v8` | 21/24 (87.5000%) | `[-1.063464, -0.971878]` | 0.063464 |

The `pilot-v1-45ebff5` promotion then evaluated `libero_spatial` tasks 0 and
1 with three trials per task, two parallel environments, canonical 220-step
limits, eight-step chunks, and zero infrastructure errors:

| Checkpoint | Pilot success | Wilson 95% CI | Gripper clipped/generated | Raw range | Max overshoot |
| ---------- | ------------- | ------------- | ------------------------- | --------- | ------------- |
| Base Edge regular, zero-shot | 0/6 (0.0000%) | [0.0000%, 39.0334%] | 410/1344 (30.5060%) | `[-3.163910, 3.252380]` | 2.252380 |
| 5k Edge EMA | 5/6 (83.3333%) | [43.6497%, 96.9947%] | 378/744 (50.8065%) | `[-1.042301, 1.030949]` | 0.042301 |
| 10k Edge EMA | 6/6 (100.0000%) | [60.9666%, 100.0000%] | 458/576 (79.5139%) | `[-1.092764, 1.081303]` | 0.092764 |

### Formal full-v1 results

The formal matrix used run ID `full-v1-45ebff5`, all 10 tasks in each of
`libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`, 50 trials
per task, eight parallel environments, seed 0, action horizon 8, 10 warmup
steps, canonical suite limits, eight UniPC steps, guidance 1.0, and a
120-second request timeout. Twelve independent one-replica/one-H200 RJobs each
wrote one checkpoint/suite directory. All 12 scheduler jobs report
`Succeeded`; every directory contains exactly 500 terminal episode records,
schema-2 `metrics.json`, and `_SUCCESS`. There are 6000 terminal episodes and
zero infrastructure attempts or unresolved infrastructure errors.

Counts below are exact values from the sealed suite artifacts; percentages and
intervals are rounded for display. The checkpoint aggregate is micro
episode-weighted across four equally sized suites; checkpoints remain
independent identities.

| Checkpoint | `libero_spatial` | `libero_object` | `libero_goal` | `libero_10` | Overall (Wilson 95% CI) |
| ---------- | ---------------- | --------------- | ------------- | ----------- | ----------------------- |
| Base Edge regular, zero-shot | 0/500 (0.0000%) | 0/500 (0.0000%) | 0/500 (0.0000%) | 0/500 (0.0000%) | 0/2000 (0.0000%), [0.0000%, 0.1917%] |
| 5k Edge EMA | 388/500 (77.6000%) | 476/500 (95.2000%) | 397/500 (79.4000%) | 410/500 (82.0000%) | 1671/2000 (83.5500%), [81.8612%, 85.1102%] |
| 10k Edge EMA | 384/500 (76.8000%) | 479/500 (95.8000%) | 371/500 (74.2000%) | 425/500 (85.0000%) | 1659/2000 (82.9500%), [81.2390%, 84.5346%] |

The following clipping rates are count-weighted over complete generated
chunks, including unexecuted tails, and exclude infrastructure attempts:

| Checkpoint | `libero_spatial` | `libero_object` | `libero_goal` | `libero_10` | Overall |
| ---------- | ---------------- | --------------- | ------------- | ----------- | ------- |
| Base Edge regular, zero-shot | 35259/112000 (31.4813%) | 44792/140000 (31.9943%) | 48239/152000 (31.7362%) | 82915/260000 (31.8904%) | 211205/664000 (31.8080%) |
| 5k Edge EMA | 28381/60264 (47.0945%) | 24511/54664 (44.8394%) | 29367/65496 (44.8379%) | 55495/121168 (45.8000%) | 137754/301592 (45.6756%) |
| 10k Edge EMA | 48530/61704 (78.6497%) | 43294/53440 (81.0142%) | 54489/69960 (77.8859%) | 92822/113272 (81.9461%) | 239135/298376 (80.1455%) |

Across all four suites, the Base pre-clamp range was
`[-4.657578, 4.700876]` with maximum overshoot 3.700876; the 5k range was
`[-1.079304, 1.071179]` with overshoot 0.079304; and the 10k range was
`[-1.133901, 1.115089]` with overshoot 0.133901. Clipping telemetry is an
adapter-boundary audit metric, not task success and not a reason to merge the
two fine-tuned checkpoint identities.

Formal provenance is uniform except for the declared checkpoint identity and
per-job scheduler ID: protocol `cosmos-libero-eval-v3`, manifest schema 3,
metrics schema 2, clean Git commit `45ebff509696730429d028e7b8cdc2a7bcf75584`,
image
`registry.h.pjlab.org.cn/ailab-llmrazor/xtuner_tmp:pt28_20260303_f2adb47`,
H200 GPU, Torch `2.13.0+cu130`, LIBERO 0.1.1, robosuite 1.4.0, MuJoCo 3.3.2,
and uv-lock SHA-256
`f5ec25154f3dc63f49680c91c0dee807b02ea6b846833d6f204bc5659f67e555`.
The RJob names are
`c3-libero-{base,5k,10k}-{spatial,object,goal,10}-full-0819-v1`; each exact
name is also immutable in its suite manifest.

Rollout-efficiency totals are descriptive outcomes under the same canonical
suite limits, not normalized scores:

| Checkpoint | Steps total / mean | Decisions total / mean |
| ---------- | ------------------ | ---------------------- |
| Base Edge regular, zero-shot | 660000 / 330.0000 | 83000 / 41.5000 |
| 5k Edge EMA | 295284 / 147.6420 | 37699 / 18.8495 |
| 10k Edge EMA | 292048 / 146.0240 | 37297 / 18.6485 |

The sealed report was reproduced and atomically published with the committed
CLI:

```bash
.venv/bin/python -m cosmos_framework.evaluation.libero.aggregate \
  --runs-root /mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework-eval/libero/runs \
  --run-id full-v1-45ebff5 \
  --checkpoint-id base-edge-action-hf-regular \
  --checkpoint-id edge-5k-hf-ema \
  --checkpoint-id edge-10k-hf-ema \
  --suites libero_spatial,libero_object,libero_goal,libero_10 \
  --task-ids 0,1,2,3,4,5,6,7,8,9 \
  --trials-per-task 50 \
  --require-canonical-max-steps \
  --require-zero-infra-attempts \
  --output /mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework-eval/libero/reports/full-v1-45ebff5/summary.json
```

The 139089-byte output has report schema 1, kind
`libero-checkpoint-matrix`, protocol `cosmos-libero-eval-v3`, 12 sealed
sources, 6000 terminal episodes, and SHA-256
`0e5a5fa4090cae396a227516175bd1e5ef36675035f4593a8b5298fbddad76d2`.
It retains separate checkpoint objects, their exact identity/profile and
per-source artifact hashes, task/suite/overall Wilson metrics, steps and
decisions, and terminal-only count-weighted gripper telemetry. Re-running the
same CLI accepted the byte-identical report; a conflicting output is never
overwritten.

### PJLab launch skeletons

The current mount URI is
`gpfs://gpfs1/evoagi-share/VTLA:/mnt/shared-storage-user/evoagi-share/VTLA`.
One loopback policy-server/runner pair uses one replica and one GPU.

```bash
rlaunch \
  --gpu=1 --cpu=16 --memory=131072 \
  --charged-group=evoagi_gpu \
  --positive-tags=feature/gpfs=yes \
  --image=<EVAL_IMAGE> \
  --custom-resources=brainpp.cn/fuse=1 \
  --mount=gpfs://gpfs1/evoagi-share/VTLA:/mnt/shared-storage-user/evoagi-share/VTLA \
  -- bash
```

```bash
rjob submit \
  --name <JOB_NAME> \
  --group evoagi_gpu \
  --charged-group evoagi_gpu \
  --private-machine group \
  --preemptible no \
  --image <EVAL_IMAGE> \
  --replica 1 --gpu 1 --cpu 16 --memory 131072 \
  --positive-tags feature/gpfs=yes \
  --custom-resources brainpp.cn/fuse=1 \
  --mount gpfs://gpfs1/evoagi-share/VTLA:/mnt/shared-storage-user/evoagi-share/VTLA \
  --share-host-shm true \
  --restart-policy never \
  --backoff_limit 1 \
  --auto-delete-duration 168h \
  -- bash -lc '<source shared environment; export EGL and LIBERO variables; run preflight or job>'
```

`--positive-tags` is a hard scheduling selector. The proven private-pool
contract uses only `feature/gpfs=yes`; adding an inferred `h200` tag excluded
otherwise compatible H200 workers. It intentionally uses
`--preemptible no` rather than an unsupported `--gpu-qos` spelling. Inspect the
job and its concrete replica without changing cluster state:

```bash
rjob get <JOB_NAME>
rjob events <REPLICA_NAME> --replica
rjob logs replica <REPLICA_NAME> -n 200
rjob logs job <JOB_NAME> -n 200
```

The 16-CPU/128-GiB request supported all formal eight-environment jobs without
OOM or infrastructure failure; it is a validated configuration, not a measured
minimum. The final image reference, scheduler ID, mount, package versions,
resolved profile, checkpoint fingerprint, and sampling settings are persisted
as provenance.

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
  LeRobot's PyAV backend. DEV-0006 made that backend an explicit, portable
  recipe setting after the CUDA image also proved unable to load TorchCodec's
  required FFmpeg libraries.

### DEV-0005

- Added and registered `action_policy_libero_all_edge_10fps`.
- Added its paired structured TOML and launcher.
- Added configuration tests covering Edge model semantics, pretrained action
  head loading, 10 FPS/8-action/full-dataset behavior, `image2`, statistics,
  FSDP8, global batch 2048, offline W&B, and save cadence.
- Updated `docs/action_policy_libero_posttrain.md`, `docs/training.md`, and
  `examples/README.md` with only reusable behavior and launch instructions.
  Cluster-specific paths and execution evidence remain in this document.
- Extended HF-to-DCP conversion with an optional local VAE path and
  checkpoint-bundled Edge processor selection; added five focused regression
  tests for the offline redirects.
- Added optional local Edge processor selection to the recipe and launcher so
  actual model construction does not require Hub access.
- Validation: Ruff, format, shell syntax, diff, focused recipe/converter tests,
  all example TOML schema tests, launcher mock validation against the real
  dataset, and the corrected H200 dry-run passed. The full TOML loader suite
  has two pre-existing failures on the GPU-less development pod because
  `transformer_engine` requires CUDA runtime discovery; the shipped-example
  schema subset passed.

### DEV-0006

- `cosmos3-edge-libero-smoke-b128-57317639`: failed before model loading
  because the wrapper called the image's system `torchrun`, whose Python
  environment lacked `omegaconf`. The PJLAB job wrapper now puts the repository
  `.venv/bin` first on `PATH`.
- `cosmos3-edge-libero-smoke-b128-r2-39257879`: reached distributed
  initialization and dataset construction, then failed because TorchCodec
  could not load any supported FFmpeg `libavutil.so`. The dataset now exposes a
  backward-compatible `video_backend`, and the Edge recipe selects `pyav`.
- `cosmos3-edge-libero-smoke-b128-r3-9912282`: succeeded on 8×H200 with
  batch/accumulation `128/2`. It loaded all 549 base-model keys while
  intentionally warm-starting `net_ema`, ran 20 steps without OOM, and wrote
  `iter_000000020`. The first compiled step took 88.65 s; steady-state steps
  took about 19.6 s. Rank losses were about `15.12–15.69` at step 1 and
  `14.35–15.14` at step 20.
- `cosmos3-edge-libero-resume-b128-76675783`: succeeded. It restored the model,
  all 2898 optimizer state items, scheduler, trainer, RNG state, and iteration
  20; then ran steps 21–22 and wrote `iter_000000022`. Step-22 rank losses were
  `14.48–15.52`.
- `cosmos3-edge-libero-full-b128-2361150`: started from `iter_000000022` with
  the validated `128/2` setting and succeeded at iteration 5000. The final
  rank-0 loss was `0.5654`; `iter_000005000` saved in 2.98 s.
  The trainer logged `Done with training` at 2026-07-31 14:56:37 +08:00.
  The complete DCP occupies 30 GiB and contains eight model shards of about
  2.53 GB plus eight optimizer shards of about 1.42 GB, together with trainer,
  scheduler, RNG, and metadata state. The checkpoint is at
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/model/cosmos3-edge-libero-all-10fps/cosmos3_action_libero/action_sft/action_policy_libero_all_edge_10fps_full/checkpoints/iter_000005000`.
  The offline W&B run is
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/model/cosmos3-edge-libero-all-10fps/cosmos3_action_libero/action_sft/action_policy_libero_all_edge_10fps_full/wandb/offline-run-20260730_112114-pk7bdv6q`.

- Added `PackingDataLoader` support to Edge export policy metadata extraction
  while retaining the legacy joint-dataloader layout. Resolved training YAML
  can now be inspected without a legacy root `_type`, and the LIBERO wrapper
  exposes its default `embodiment_type` for export.
- `cosmos3-edge-libero-export-check-21245457`: exposed the missing root `_type`
  compatibility issue before reading weights.
- `cosmos3-edge-libero-export-check-r2-21420917`: passed raw-config loading and
  exposed that the resolved wrapper config did not contain a policy domain.
- `cosmos3-edge-libero-export-check-r3-80137598`: succeeded on one H200. Its
  generated `checkpoint.json` contains `action_chunk_size=8`,
  `conditioning_fps=10.0`, and `domain_name=libero`.
- Five focused export-metadata tests and the focused Edge recipe/export suite
  passed. The new resolved-config unit test cannot collect on the development
  pod because its existing `transformer_engine` import requires `libcudart`;
  the real H200 export precheck validates that path end to end.
- `cosmos3-edge-libero-export-smoke-53586371`: completed a provisional
  safetensors export from the resume checkpoint. The 7.3 GiB result contains
  two model shards, the 979 MB Edge vision encoder, processor/tokenizer files,
  export manifest, and correct policy metadata.
- `cosmos3-edge-libero-forward-smoke-32878448`: the real sample loaded, but the
  action service's default guardrails imported OpenCV/RetinaFace and failed on
  the container's missing `libxcb.so.1`. The isolated forward checker now
  disables unrelated guardrails.
- `cosmos3-edge-libero-forward-smoke-r2-12363635`: the exported model loaded,
  then the checker exposed an incorrect normalizer-statistics path. It also
  showed that the export's bundled processor was bypassed because the public
  config retained the PJLAB-local processor path.
- `cosmos3-edge-libero-export-smoke-r2-37086228`: an initial attempt to scrub
  the local processor path before model construction selected the registered
  object-store processor and failed for missing credentials. Canonicalization
  was moved to the final public-config rewrite, after local model construction
  and processor bundling.
- `cosmos3-edge-libero-export-smoke-r3-61518455`: succeeded. The public config
  contains `nvidia/Cosmos3-Edge-Reasoner` rather than a PJLAB path, while the
  export remains self-contained through its bundled processor.
- `cosmos3-edge-libero-forward-smoke-r3-10700008`: succeeded on one H200 with
  real merged-LIBERO sample 0. The prompt was “put the white mug on the left
  plate and put the yellow and white mug on the right plate”; input video shape
  was `[3, 9, 256, 512]`; the offline exported model loaded in 10.98 s and a
  two-denoising-step forward completed in 22.57 s; the finite denormalized
  action output had shape `[1, 8, 10]` and range `[-2.3340, 1.4602]`.
- Added checkpoint-aligned qualitative visualization to the Edge training
  recipe. Every save interval now triggers one EMA WAM sample and stores a
  local 10 FPS prediction/VAE-reconstruction/ground-truth video plus offline
  W&B preview frames. Static configuration tests lock its cadence, rank count,
  sampling settings, and local-only output behavior.
- The first visualization dry-run submission was rejected before job creation
  because the one-GPU request was incompatible with host networking.
  `cosmos3-edge-libero-viz-dryrun-r2-66499044` then succeeded and resolved
  exactly one `libero_rollout` callback with one EMA sample, 8 denoising steps,
  10 FPS, local-only output, and cadence 500.
- `cosmos3-edge-libero-viz-smoke-37737720`,
  `cosmos3-edge-libero-viz-smoke-r2-55564857`, and
  `cosmos3-edge-libero-viz-smoke-fsdp8-67119292` failed in the CUDA preflight
  without loading a checkpoint. A development-pod `uv run` had unexpectedly
  synchronized the shared `.venv` from the verified `torch 2.10.0+cu128` to
  `torch 2.13.0+cu130`; the already-running full job was unaffected because
  its process had loaded the old libraries before that change. New validation
  jobs therefore build an isolated node-local CUDA 12.8 environment without
  modifying or restarting the full job.
- `cosmos3-edge-libero-viz-smoke-fsdp8-r3-83728679` and
  `cosmos3-edge-libero-viz-smoke-fsdp8-r4-75714959` failed before CUDA
  preflight because the H200 node's PyPI proxy timed out while fetching NVRTC
  and CMake respectively. The replacement flow copies non-CUDA packages from
  the existing environment, restores the locked CUDA 12.8 packages from the
  shared uv cache, and installs a staged NVRTC wheel only after verifying its
  SHA256
  `a7756528852ef889772a84c6cd89d41dfa74667e24cca16bb31f8f061e3e9994`.
- `cosmos3-edge-libero-viz-smoke-fsdp8-r5-22977600` passed the CUDA preflight
  and reached 8-rank `torchrun`, but an unanchored temporary rsync exclusion
  also removed `wandb/integration/torch`; it failed before model loading and
  produced no training artifact. Anchoring exclusions to the top-level
  `site-packages` CUDA packages fixed this environment-only issue.
- `cosmos3-edge-libero-viz-smoke-fsdp8-r6-84101056` succeeded end to end on
  8×H200. The preflight reported `torch 2.10.0+cu128`, CUDA 12.8 available,
  and all eight H200s. It restored `iter_000000022`, ran iteration 23 with
  rank-0 loss `15.1887`, saved `iter_000000023` in 6.93 s, switched to EMA,
  and completed the 8-step UniPC sample in 8.30 s.
- The validated local artifact is
  `cosmos3_action_libero/action_sft/action_policy_libero_all_edge_10fps_viz_smoke_fsdp8/EveryNDrawSample/Iter000000023/ema_ReplicateID0000_Sample_Iter000000023.mp4`.
  FFprobe reports H.264, `320×576`, 10 FPS, 9 frames, and 0.9 s duration. Its
  three vertically stacked rows are prediction, VAE reconstruction, and ground
  truth. The reconstruction and ground truth are coherent, while the iteration
  23 prediction is still noise-like, as expected for this deliberately early
  checkpoint; this confirms that the visualization exposes qualitative
  convergence rather than masking it.
- Final training, export, and forward validation are complete.
- `cosmos3-edge-libero-final-export-94756306` used the final run config,
  `iter_000005000`, and the GPFS-local Edge HF vision bundle and succeeded on
  one H200. The output is at
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/model/cosmos3-edge-libero-all-10fps/export-final-iter5000`.
  The 7.3 GiB export has two model shards of 5,000,054,640 and 1,739,338,512
  bytes, a 978,739,880-byte bundled vision encoder, and bundled processor and
  tokenizer files. `checkpoint.json` records action chunk 8, 10 FPS, domain
  `libero`, and EMA weights. The public model and processor configuration
  replaced the export-host-local processor with
  `nvidia/Cosmos3-Edge-Reasoner`; no PJLAB processor path remains in those
  public fields.
- `cosmos3-edge-libero-final-forward-81163290-6-170ed` was stopped without
  starting because its mistyped GPFS source (`gpfs2-intern-...`) caused the
  storage admission webhook to receive an empty site. The corrected submission
  uses `gpfs://gpfs2/intern-pretrain-shared02:...`.
- `cosmos3-edge-libero-final-forward-r2-4620973-e0067` succeeded on one H200
  with real merged-LIBERO sample 0. The prompt was “put the white mug on the
  left plate and put the yellow and white mug on the right plate”; input video
  shape was `[3, 9, 256, 512]`; model load took 9.903 s and a two-step forward
  took 16.196 s. The denormalized action output was finite with shape
  `[1, 8, 10]` and range `[-1.0180979, 1.0006449]`. The machine-readable report
  is at
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/model/cosmos3-edge-libero-all-10fps/forward-final-iter5000.json`.
- `cosmos3-edge-libero-final-validate-62811407-d389a` used the copied
  `pytest` console script, whose absolute shebang still selected the polluted
  shared `.venv`; collection therefore failed on its torch/torchvision version
  mismatch. The corrected validation invokes the node-local Python with
  `-m pytest` so the verified CUDA 12.8 environment is authoritative.
- `cosmos3-edge-libero-final-validate-r2-314760-71180` succeeded on one H200
  with `torch 2.10.0+cu128`: 15 focused recipe, dataset, and export-policy tests
  passed in 3.25 s; the resolved-YAML config test passed with four unrelated
  tests deselected in 5.54 s; and direct TOML parsing passed. Development-pod
  Ruff, format, shell syntax, and diff checks also passed.

### DEV-0009

- Changed both the Python experiment baseline and structured TOML to save
  periodic DCP checkpoints every 2000 steps. The EMA visualization callback now
  uses the same cadence. Trainer final-save behavior preserves step 5000, so a
  default run saves steps 2000, 4000, and 5000 while generating periodic
  rollouts at steps 2000 and 4000.
- Audited Flash Attention without changing its configuration. H200/SM90 orders
  the Cosmos attention backends as `flash3`, `cudnn`, `natten`, then `flash2`,
  and the CUDA 12.8 dependency group declares `flash-attn-3-nv` and
  `flash-attn`. However, the completed-run evidence recorded Torch 2.10/CUDA
  12.8 but not the backend actually selected, so historical Flash Attention 3
  use is not proven.
- The migrated persistent `.venv` currently uses Torch `2.13.0+cu130` and has
  neither `flash-attn-3-nv` nor `flash-attn` installed. A new run that directly
  uses this environment will therefore skip the external Flash Attention 3/2
  backends and select the next compatible backend. Before claiming FA3 for a
  future H200 run, use the CUDA 12.8 dependency group and record both a
  successful `flash_attn_3_nv` import and the selected backend in the job log;
  setting `I4_ATTN_BACKENDS=flash3` provides a fail-fast enforcement check.

### DEV-0010

- The Edge LIBERO launcher now overwrites `I4_ATTN_BACKENDS` with the exact
  allow-list `flash3`; an inherited value cannot restore cuDNN, NATTEN, or FA2
  fallback.
- Added a recipe prelaunch hook and a dedicated CUDA check that verifies SM90,
  imports the ABI-matched `flash-attn-3-nv`, confirms that Cosmos filters the
  backend list to only `flash3`, selects that backend for a representative
  bfloat16 training shape, and executes finite forward and backward kernels.
  Any failure exits before `torchrun`, model loading, or checkpoint I/O.
- The persistent development `.venv` remains Torch 2.13/CUDA 13.0 and is not a
  valid FA3 training environment. GPU validation therefore uses an isolated
  node-local Torch 2.10 environment resolved from the repository's locked CUDA
  dependency group; it does not create a second repository `.venv`.
- H-cluster migration validation established that scheduled jobs must use the
  `evoagi_gpu` charged group and explicitly mount the new GPFS workspace. The
  first two submissions were rejected before job creation by obsolete/default
  charged groups. Preflight `r3` then exposed the missing workspace mount;
  `r4` exposed compute-node package-index timeouts; `r5` showed that offline
  resolution could not discover the cached Torch wheel; `r6` exposed a missing
  NVSHMEM runtime; and `r7` exposed the image's older NCCL symbol set. These
  were environment-only diagnostics and did not start training.
- `cosmos3-edge-libero-fa3-preflight-r8` succeeded on one NVIDIA H200
  (`gpu-lg-cmc-h-h200-0120`) using the locked Torch 2.10.0+cu128 and
  `flash-attn-3-nv` 1.0.3 packages plus their CUDA 12.8 runtime libraries. The
  check constrained Cosmos to `backend=flash3`, selected it for bfloat16
  training tensors, and executed finite forward and backward kernels on SM90.
  The exact terminal evidence is: `Flash Attention 3 preflight passed:
  backend=flash3, package=1.0.3, torch=2.10.0+cu128, cuda=12.8, device=NVIDIA
  H200, sm=90.`
- The persistent evidence log is
  `/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework/flash3-preflight/DEV-0010/preflight.log`.
  The validated wheel SHA-256 values are
  `7b4bd23ed63de97456fcc81c26fea9f02ee02ce1112111c4dac0d8cfe574b23e`
  for Torch and
  `ed7b3cf08ffacdeadfaa44ee674a3a5e67e7011829e0e757eb6f3463fb8d443e`
  for Flash Attention 3. The node-local validation environment was ephemeral;
  no second repository `.venv` was retained.

### DEV-0011

- Scheduled a continuation from the migrated 30 GiB `iter_000005000` DCP to a
  total `trainer.max_iter=10000`, retaining batch/accumulation `128/2`, global
  batch 2048, LR `5e-5`, warmup 500, FSDP8, W&B offline, and forced FA3. The
  resume override sets `checkpoint.load_training_state=true`, so model, EMA,
  optimizer, scheduler, trainer, and RNG state continue from iteration 5000.
- All inputs are now available under the migrated GPFS1 workspace: the merged
  LIBERO dataset, Edge DCP/HF processor, Wan VAE, and final iteration-5000
  checkpoint. The new run writes only below
  `/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework-training/cosmos3-edge-libero-10k-b128-fa3`.
  With `save_iter=2000`, expected checkpoints and EMA rollout videos are at
  iterations 6000, 8000, and 10000.
- The persistent repository `.venv` remains Torch 2.13/CUDA 13.0. A direct
  full sync to CUDA 12.8 was stopped before installation because unpacking the
  CUDA libraries into GPFS was unacceptably slow. The job instead creates one
  ephemeral node-local runtime from the repository's complete locked uv cache,
  while keeping only one persistent repository `.venv`. The runtime uses an
  immutable code export from commit `b97f5840afd9b19c762b5548537e0c30e0f395c0`.
- The launch wrapper is an experiment artifact at
  `/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework-training/jobs/cosmos3-edge-libero-10k-b128-fa3/launch.sh`
  with SHA-256
  `3aac9cbdf298a1cbb703d1fe4f1ccf425199c8cde82ed78e8f9dfa49014fde14`.
  It validates migrated inputs, constructs the node-local environment, forces
  `I4_ATTN_BACKENDS=flash3`, runs the real FA3 preflight, and only then starts
  eight-rank `torch.distributed.run`.
- `cosmos3-edge-libero-10k-fa3-preflight` failed before environment setup
  because the container lacks `git-lfs`; no training or CUDA kernel ran. The
  wrapper now disables the LFS filter when exporting the pinned code commit.
  `cosmos3-edge-libero-10k-fa3-preflight-r2` then succeeded on H200 node 0119:
  Torch `2.10.0+cu128`, CUDA 12.8, torchvision `0.25.0+cu128`, and
  `flash-attn-3-nv` `1.0.3+cu128.torch210`; the forced FA3 forward/backward
  kernel passed on SM90.
- The first formal submission, `cosmos3-edge-libero-10k-b128-fa3`, requested
  120 CPUs and was stopped while still pending because all 15 matching nodes
  reported insufficient free CPU. `cosmos3-edge-libero-10k-b128-fa3-r2` keeps
  8 H200s, 1,800,000 MiB memory, host networking/shared memory, gang start, and
  RDMA resources while reducing CPU to 96. It was submitted at 2026-08-04
  14:21 +08:00 and remained queued for a matching node at handoff. The durable
  combined bootstrap/training log is `logs/full.log` under the run root above.

### DEV-0012

- `cosmos3-edge-libero-10k-b128-fa3-r2` reached H200 node
  `gpu-lg-cmc-h-h200-0905.host.h.pjlab.org.cn`. Its formal-run preflight passed
  with Torch `2.10.0+cu128`, CUDA 12.8, `flash-attn-3-nv`
  `1.0.3+cu128.torch210`, H200 SM90, forced `backend=flash3`, and finite
  forward/backward kernels.
- All eight ranks then entered `distributed.init()`, but ranks failed at
  `os.sched_setaffinity(0, device.get_cpu_affinity())` with
  `OSError: [Errno 22] Invalid argument`. The failure occurred before process
  group initialization, checkpoint loading, model/data loading, or any
  training iteration and produced no new checkpoint.
- NVML reports CPUs using the host topology, while the rjob container may
  restrict the process to a different Kubernetes cpuset. The old code passed
  the host CPU IDs directly to `sched_setaffinity` and caught only NVML
  exceptions. The fix intersects the NVML set with `os.sched_getaffinity(0)`,
  skips manual binding when the intersection is empty, and treats an OS-level
  rejection as a non-fatal warning.
- Focused regression tests cover a partially overlapping cpuset, disjoint CPU
  sets, and the original `OSError` path. The generic container failure mode and
  behavior are also documented in `docs/faq.md`.

### DEV-0013

- Updated the external launch wrapper to export immutable code commit
  `fd013090dcef551b41a1b131434d1f4cf961a1aa`, which contains the
  container-aware CPU-affinity fix. The updated wrapper SHA-256 is
  `b9aea0daa90f1825d2c919e3733f3e5bf9005e720291b94d4a480f68e2cca35a`;
  `bash -n` passed and its forced `I4_ATTN_BACKENDS=flash3`, max iteration
  10000, and full training-state resume overrides were rechecked.
- Submitted `cosmos3-edge-libero-10k-b128-fa3-r3` at 2026-08-04 16:11
  +08:00 with the validated configuration: one private node, 8 H200s, 96 CPU
  cores, 1,800,000 MiB memory, host networking/shared memory, gang start,
  eight shared RDMA devices, one Mellanox RDMA device, `brainpp.cn/fuse=1`,
  the `evoagi_gpu` charged group, and the explicit GPFS1 VTLA mount.
- Submission-time `rjob get` reported replica
  `cosmos3-edge-libero-10k-b128-fa3-r3-cfcd2` as `STARTING` while the job was
  in queue. Training has not started yet; the job will append environment,
  FA3 preflight, resume, iteration, loss, checkpoint, and visualization
  evidence to the existing durable `logs/full.log` under the run root.

### DEV-0014

- `cosmos3-edge-libero-10k-b128-fa3-r3` started at 2026-08-04 20:39:56
  +08:00 on H200 node 0905. Forced-FA3 preflight passed, all eight NCCL ranks
  initialized through the new empty-cpuset fallback, the iteration-5000 model,
  optimizer, scheduler, trainer, and RNG state loaded, and training reached
  iteration 5016 with rank-0 loss `0.8696` and approximately 21.5 seconds per
  iteration.
- At 20:47:40 +08:00, both the RJob and replica specs acquired the exact
  `stopTimestamp=2026-08-04T12:47:40Z`; the pod then received SIGTERM and the
  job entered `Stopped`. This rules out an application exception, GPU OOM,
  preemption, and scheduler eviction: a client or API with namespace authority
  explicitly invoked the stop path. The RJob events, status, annotations, and
  CRD fields do not identify that caller, and the user confirmed they did not
  issue the stop.
- The termination callback observed SIGTERM at iteration 5016, but no
  `iter_000005016` directory was created. The retry therefore resumes again
  from the last complete `iter_000005000` checkpoint.
- Submitted `cosmos3-edge-libero-10k-b128-fa3-r4` at 20:52 +08:00 with the
  unchanged immutable code, wrapper, H200 resources, forced-FA3 environment,
  and full-state resume configuration. It scheduled immediately on node 0905;
  FA3 preflight passed again and all eight ranks initialized with NCCL by
  20:53:14. No code or training configuration change was required.
- The recurring monitor `monitor-cosmos3-edge-libero-10k-r4` now treats `r4`
  as the active job and must only inspect state and evidence. It must not stop,
  delete, patch, or resubmit cluster resources without explicit user
  authorization.

### DEV-0015

- Added a strict, resumable Cosmos3-Edge LIBERO evaluation pipeline with
  machine-readable EGL/simulator preflight, immutable checkpoint/profile
  identity, versioned HTTP handshake, deterministic batched runner, canonical
  artifacts, infrastructure-attempt retry, and a single-checkpoint job that
  isolates and cleans up server/runner process groups.
- Locked Edge target semantics in one adapter. The formal comparison is base
  Edge HF regular versus the 5k and 10k fine-tuned HF EMA exports; DCP is
  retained only as a regular-weight source/debug path.
- Confirmed the locked LIBERO 0.1.1 / robosuite 1.4.0 / MuJoCo 3.3.2
  environment and a four-primary-suite EGL preflight. At the DEV-0015 commit
  boundary real policy rollouts had not started; the later smoke, pilot, and
  full results are recorded in the formal-results section above.
- Updated the public guide and this operations record with the checkpoint
  matrix, 10k export, environment/assets, canonical output and resume rules,
  promotion gates, and PJLab launch/mount skeletons.
- Combined focused validation passed 127 tests. Ruff lint and format checks
  passed, and Pyrefly reported 0 errors. The strict resolver successfully
  resolved all three real checkpoint profiles and fingerprints, and the locked
  EGL preflight passed all four primary LIBERO suites.

### DEV-0023

- The requested model-only warm start is
  `/mnt/shared-storage-user/evoagi-share/VTLA/chenyitong/outputs/cosmos3-edge/cosmos3_action/wam_action_mixed_dense_epoch/mixed-edge-dense-epoch64-noac-resume10000-20260726-125729/checkpoints/iter_000090000`.
  Its model DCP has 1,098 metadata entries: 549 regular `net.*` tensors and
  549 `net_ema.*` tensors. Comparing the regular tensors with
  `/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/model/Cosmos3-Edge-dcp`
  found 549 shared keys, zero keys unique to either side, and zero shape
  mismatches. The 64-way source layout has eight 2,528,971,546-byte data
  shards and 56 zero-byte replica placeholders, consistent with FSDP8 across
  eight replicated groups.
- The LIBERO run deliberately uses the regular source weights:
  `checkpoint.load_path` is the requested iteration-90000 directory,
  `checkpoint.load_training_state=false`,
  `checkpoint.keys_to_skip_loading=["net_ema."]`, and
  `checkpoint.load_ema_to_reg=false`. Training therefore starts at iteration
  0 with fresh optimizer, scheduler, trainer, RNG, and EMA state; it does not
  resume the WAM step 90000 or its training state.
- The established Edge recipe remains otherwise unchanged: one 8-rank FSDP
  node, 128 samples/rank, gradient accumulation 2, global batch 2048, LR
  `5e-5`, warmup 500, 10,000 iterations, W&B offline, 10 FPS merged
  two-camera LIBERO data, and forced Flash Attention 3. DCP checkpoints and EMA
  rollout visualizations are expected every 2000 steps at 2000, 4000, 6000,
  8000, and 10000.
- The formal output root is
  `/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework-training/cosmos3-edge-libero-wam90k-10k`.
  The immutable launcher uses code commit
  `9048081deb43e179bc6e782d64d9496479efecd7` and is stored at
  `/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework-training/jobs/cosmos3-edge-libero-wam90k-10k/launch.sh`
  with SHA-256
  `c48c67f6163f36ef1352e0887c671804a1dc63735526ceb58eadc74d75709e66`.
  It reconstructs the validated node-local Torch 2.10/CUDA 12.8 runtime,
  requires a finite Flash Attention 3 forward/backward preflight, and writes a
  durable combined log to `logs/full.log` under the formal output root.
- A structured TOML dry-run passed with the requested source, model-only load,
  FSDP8, batch/accumulation `128/2`, and local Edge processor; its emitted
  config is retained under the smoke output root's `dryrun/config-run`.
  The one-step job `cosmos3-edge-libero-wam90k-smoke` was submitted at
  17:00 +08:00, remained queued without a worker or log, and was intentionally
  stopped at 17:12 before CUDA or training ran. Its launcher SHA-256 is
  `3da8be5d757cfa6e3ed0d2b487ba5137e54931b7e932d178d79b62e904441c91`.
- The formal job `cosmos3-edge-libero-wam90k-10k` was submitted at 17:13
  +08:00 with one private non-preemptible replica, 8 H200s, 96 CPUs,
  1,800,000 MiB host memory, host networking/shared memory, gang start, RDMA,
  `brainpp.cn/fuse=1`, the `evoagi_gpu` group/quota, and the explicit GPFS1
  VTLA mount. At submission its replica
  `cosmos3-edge-libero-wam90k-10k-cfcd2` was `STARTING` in the scheduler
  queue; no worker-side preflight or training evidence existed yet.
