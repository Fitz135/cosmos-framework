# Development Changelog

This file records completed development changes to Cosmos-Framework. Every Git
commit maps to exactly one changelog entry through a shared, monotonically
increasing `DEV-NNNN` identifier. Keep entries in reverse commit order, with the
newest entry first, and never reuse or renumber an identifier.

Each entry must contain:

- **Heading**: change ID, completion timestamp, and short title.
- **Area**: affected component or workflow.
- **Summary**: concise description of the completed change.
- **Documentation**: documentation added or updated with the change.
- **Validation**: checks run to verify the change.

Start the matching Git commit subject with the same identifier, for example
`DEV-0002: add training configuration`. Do not record a commit's own hash in its
entry because changing the entry would produce a different hash.

Use this template:

```markdown
## DEV-NNNN — YYYY-MM-DD HH:MM +08:00 — Short title

- **Area**: ...
- **Summary**: ...
- **Documentation**: ...
- **Validation**: ...
```

## DEV-0022 — 2026-08-19 15:45 +08:00 — Add Edge LIBERO evaluation runbook

- **Area**: Cosmos3-Edge LIBERO operator documentation.
- **Summary**: Added a concise Chinese runbook covering the strict evaluation
  entry points, environment preflight, four primary suites and canonical
  limits, local and validated RJob launch patterns, artifact inspection, and
  the completed three-checkpoint results.
- **Documentation**: Added `dev/edge_libero_eval.md` as the practical operator
  guide for launching, monitoring, and reading Edge LIBERO evaluations.
- **Validation**: Extracted every Bash code block and passed `bash -n`; checked
  all referenced code, adapter, checkpoint, and report paths; matched the
  formal report SHA-256; scanned for credentials and sensitive token names;
  and passed `git diff --check`.

## DEV-0021 — 2026-08-19 07:32 +08:00 — Seal and report the LIBERO checkpoint matrix

- **Area**: Strict LIBERO sealed-run aggregation, formal three-checkpoint
  evaluation reporting, and H200 RJob provenance.
- **Summary**: Added a fail-closed matrix aggregator that reloads and validates
  every immutable manifest, episode journal, metrics file, and completion
  marker; re-derives suite/task/checkpoint metrics and count-weighted gripper
  telemetry; refuses incompatible policy or rollout contracts; preserves each
  checkpoint identity; and atomically publishes a deterministic report. The
  formal Base regular, 5k EMA, and 10k EMA evaluation is complete across all
  four primary suites: 12 sealed suite runs, 6000 terminal episodes, and zero
  infrastructure errors.
- **Documentation**: Replaced the pending promotion state with the completed
  v8 smoke, pilot, and full results; recorded exact suite/checkpoint success
  rates, Wilson intervals, gripper telemetry, checkpoint identities, the
  successful single-replica RJob contract, and the reproducible sealed-report
  command and artifact path.
- **Validation**: All 154 LIBERO tests and the 42 focused aggregation/profile
  tests passed; Ruff passed and targeted Pyrefly reported 0 errors. The formal
  CLI validated all 12 immutable suite artifacts and atomically published a
  139089-byte report for 6000 episodes with zero infra attempts; separate
  scheduler inspection confirmed all 12 RJobs succeeded. An identical rerun
  proved idempotency, and the report SHA-256 is
  `0e5a5fa4090cae396a227516175bd1e5ef36675035f4593a8b5298fbddad76d2`.
  `git diff --check` passed.

## DEV-0020 — 2026-08-19 04:53 +08:00 — Version LIBERO gripper adaptation

- **Area**: LIBERO gripper semantics, strict policy/server handshake,
  episode-level action-adapter provenance, and scheduled H200 smoke execution.
- **Summary**: Made the finite `pm_one` actuator mapping an explicit immutable
  contract instead of treating every model-space excursion beyond `[-1, 1]` as
  an infrastructure failure. The adapter preserves the legacy pass-through
  semantics inside the range, clamps only the gripper channel at the LIBERO
  environment boundary, leaves all other action validation strict, and records
  complete-generated-chunk clipping telemetry per episode. Bumped the strict
  protocol to `cosmos-libero-eval-v3`, the run manifest to schema 3, and metrics
  to schema 2 with count-weighted clipping aggregation rather than an average
  of episode rates.
- **Documentation**: Defined the explicit
  `pm-one-finite-clamp-v1` contract, explained why the committed
  quantile statistics make gripper denormalization an identity transform,
  recorded the three non-promotable `v7` adapter failures, and reserved fresh
  `v8` job/run identities for post-fix validation.
- **Validation**: The focused action/metrics/runner/server suite passed 102
  tests, the full LIBERO suite passed 142 tests, and the final runner suite
  passed 28 tests. Ruff lint and format checks passed; targeted Pyrefly reported
  0 errors; and `git diff --check` passed. Full Pyrefly still reported 21
  pre-existing or environment dependency errors involving packages such as
  Apex, Lance, torchaudio, Flash Attention, NATTEN, Diffusers, and OpenPI, but
  none was in the nine modified source/test files. All three `v7` H200 jobs
  proved the v2 seed repair by completing one 8-step policy inference, then
  failed before the first environment step at the pre-fix strict `pm_one` range
  check. At the DEV-0020 commit boundary the `v8` H200 retry had not yet been
  submitted; its successful promotion evidence is recorded in DEV-0021.

## DEV-0019 — 2026-08-19 04:26 +08:00 — Version LIBERO policy sampling seeds

- **Area**: LIBERO deterministic policy sampling, server/runner protocol
  identity, durable episode provenance, and scheduled H200 smoke execution.
- **Summary**: Preserved canonical SHA-256-derived signed-63-bit logical policy
  seeds instead of truncating them to uint32, and adapted the RNG boundary with
  a lossless MT19937 key contract: existing uint32 seeds remain scalar while
  larger seeds become low-word-first uint32 pairs. Bumped the strict protocol to
  `cosmos-libero-eval-v2`, made the exact
  `sha256-canonical-json-first64-mask63-v1+mt19937-uint32-identity-or-le-u32-pair-v1`
  contract part of the `/info` handshake and schema-v2 run manifest, and
  recorded the slot-independent simulator `episode_seed` in every episode and
  infrastructure-attempt record.
- **Documentation**: Defined the v2 logical-seed and MT19937-key semantics,
  synchronized the proven single-replica private-pool `rjob` flags and
  monitoring commands, and recorded the three non-promotable `v6` failures and
  the fresh-directory requirement for the pending `v7` retry.
- **Validation**: The full LIBERO-related suite passed 145 tests; Ruff lint and
  format checks passed; focused Pyrefly reported 0 errors (9 suppressed);
  `git diff --check` passed; and a non-documentation/development grep found no
  remaining v1 protocol reference. The three `v6` H200 jobs passed the locked
  four-suite EGL preflight, loaded their intended models, and matched their
  strict profile/fingerprint handshakes before consistently exposing the
  pre-fix NumPy seed-range failure. At the DEV-0019 commit boundary the `v7`
  H200 retry had not yet been submitted; its diagnostic outcome and the later
  successful `v8` promotion are recorded in DEV-0020 and DEV-0021.

## DEV-0018 — 2026-08-19 03:55 +08:00 — Make LIBERO runtime artifacts portable

- **Area**: LIBERO runner environment selection, Edge HF export publication,
  and 5k checkpoint runtime provenance.
- **Summary**: Preserved virtualenv Python launcher symlinks so the simulator
  runner keeps its environment site-packages; normalized completed HF exports
  to cross-user `0644` files and `0755` directories without following symlinks,
  removing the completion marker on failure; and pinned a hashed external 5k
  config that relocates its sole unavailable VAE path without modifying the
  checkpoint.
- **Documentation**: Recorded the final Base and 5k identities, Base export and
  permission jobs, the explicit 5k load-config contract, and the two runtime
  defects exposed by the non-promotable `v5` H200 smoke.
- **Validation**: The combined LIBERO evaluation, action-server profile, and
  export suites passed 204 tests; Ruff lint and format checks passed; focused
  Pyrefly reported 0 errors; `git diff --check` passed. The real Base export
  resolved as HF/base/regular after cross-user permission repair, and the
  derived 5k config resolved as HF/finetuned/EMA with its new fingerprint.

## DEV-0017 — 2026-08-19 03:12 +08:00 — Correct LIBERO Base policy identity

- **Area**: Cosmos3-Edge LIBERO checkpoint loading, Base action-policy export,
  and strict server/job provenance.
- **Summary**: Excluded the framework's implicit default config from the
  server-side profile identity while retaining explicit external configs, and
  added an explicit regular-only Base Edge export mode that omits fine-tuning
  policy metadata. Corrected the formal Base target from the native public
  reasoner/vision snapshot to a self-contained Cosmos3 Omni action HF export
  derived from the Base DCP.
- **Documentation**: Added the Base action export command and clarified the
  roles of the Base DCP, relocated config, public processor/vision snapshot,
  and final HF policy artifact. Recorded the two deterministic defects exposed
  by the non-promotable `v4` H200 smoke.
- **Validation**: The combined LIBERO evaluation, action-server profile, and
  export suites passed 197 tests; Ruff lint and format checks passed; focused
  Pyrefly reported 0 errors (4 suppressed); `git diff --check` passed.

## DEV-0016 — 2026-08-19 02:42 +08:00 — Harden LIBERO policy startup identity

- **Area**: Cosmos3-Edge LIBERO policy-server startup and HF checkpoint
  identity.
- **Summary**: Disabled the unrelated generative-media guardrails for the
  fixed robot-policy endpoint so model startup no longer downloads
  `Cosmos-Guardrail1`, and made HF fingerprinting count an automatically
  discovered config inside the checkpoint exactly once while preserving
  explicit external-config identity.
- **Documentation**: Documented the offline action-server contract and the
  first scheduled H200 smoke evidence, including the corrected private-GPU
  scheduling selector and the two runtime defects exposed by the failed
  pre-fix jobs.
- **Validation**: The two focused regression files passed 26 tests; the full
  LIBERO evaluation and action-server profile suite passed 129 tests; Ruff
  lint and format checks passed; focused Pyrefly reported 0 errors (4
  suppressed). All three pre-fix H200 jobs passed the locked four-suite EGL
  preflight before consistently exposing the guarded startup failure.

## DEV-0015 — 2026-08-18 05:27 +08:00 — Add strict Cosmos3-Edge LIBERO evaluation

- **Area**: Cosmos3-Edge checkpoint identity, policy serving, LIBERO simulator
  preflight, closed-loop evaluation, and durable artifacts.
- **Summary**: Added fail-fast Edge policy profiles and checkpoint
  fingerprints, a versioned server protocol, locked four-suite EGL preflight,
  deterministic batched/resumable runner, and a single-checkpoint orchestrator
  that validates handshakes, records sampling/provenance, and cleans up complete
  process groups. The formal matrix separates base HF regular zero-shot from
  the 5k and 10k fine-tuned HF EMA checkpoints.
- **Documentation**: Replaced the Nano-oriented public eval instructions with
  the strict preflight/job/runner workflow, retained the old client as Nano
  legacy, and expanded the Edge development record with the three-checkpoint
  matrix, 10k EMA export, locked assets/environment, canonical output,
  promotion ladder, and PJLab mount examples.
- **Validation**: Successfully resolved the three real checkpoint targets with
  distinct immutable fingerprint/profile hashes; combined focused validation
  passed 127 tests; Ruff lint and format checks passed; Pyrefly reported 0
  errors; and the locked LIBERO/robosuite/MuJoCo EGL preflight passed all four
  primary suites.

## DEV-0014 — 2026-08-04 20:54 +08:00 — Recover from explicit rjob stop

- **Area**: Cosmos3-Edge LIBERO H200 training operations.
- **Summary**: Established that the healthy `r3` run was explicitly stopped
  through the RJob control plane at iteration 5016, verified that no new
  checkpoint survived, and resubmitted the unchanged full-state continuation
  as `cosmos3-edge-libero-10k-b128-fa3-r4`.
- **Documentation**: Recorded forced-FA3, distributed initialization, resume,
  throughput/loss, the exact stop timestamp and evidence boundary, missing
  termination checkpoint, new job/replica, and monitoring constraints in the
  complete LIBERO development record.
- **Validation**: Inspected RJob and replica events plus raw CRD spec/status;
  checked the durable log and output tree; `r4` scheduled on H200 node 0905,
  passed forced-FA3 preflight, and initialized all eight NCCL ranks;
  `git diff --check` passed.

## DEV-0013 — 2026-08-04 16:12 +08:00 — Resubmit 10k Edge LIBERO training

- **Area**: Cosmos3-Edge LIBERO H200 training orchestration.
- **Summary**: Pinned the training wrapper to the container-aware affinity
  commit and resubmitted the 8×H200 continuation from iteration 5000 to 10000
  with forced Flash Attention 3 and the validated 96-CPU resource profile.
- **Documentation**: Recorded the immutable code commit, launch-wrapper hash,
  exact resource allocation, new rjob and replica IDs, submission state, and
  durable evidence location in the complete LIBERO development record.
- **Validation**: The external launch wrapper passed `bash -n`; its immutable
  commit, forced-FA3 environment, max iteration, and full-state resume options
  were checked; `rjob get cosmos3-edge-libero-10k-b128-fa3-r3` confirmed the
  submitted replica in `STARTING`/in-queue state; `git diff --check` passed.

## DEV-0012 — 2026-08-04 16:03 +08:00 — Make CPU affinity container-aware

- **Area**: Distributed initialization and Cosmos3-Edge LIBERO cluster
  training.
- **Summary**: Intersected NVML's GPU-local host CPU set with the current
  process's allowed container cpuset, made empty intersections and OS affinity
  errors non-fatal, and added regressions for the failure that stopped the
  first formal 10k continuation before process-group initialization.
- **Documentation**: Added generic container-affinity troubleshooting to the
  FAQ and recorded the failed rjob, successful forced-FA3 preflight, root
  cause, absence of new checkpoints, and retry requirement in the complete
  LIBERO development record.
- **Validation**: Ruff and format checks passed; all 11 tests in
  `cosmos_framework/utils/distributed_test.py` passed; `git diff --check`
  passed.

## DEV-0011 — 2026-08-04 14:22 +08:00 — Schedule 10k Edge LIBERO continuation

- **Area**: Cosmos3-Edge LIBERO cluster training and migrated runtime setup.
- **Summary**: Scheduled an 8×H200 continuation from iteration 5000 to 10000
  with forced Flash Attention 3, batch/accumulation 128/2, 2000-step
  checkpoints, and all new outputs under the required GPFS1 output root.
- **Documentation**: Recorded migrated input paths, immutable code/runtime
  staging, preflight evidence, resource configuration, job history, output
  paths, and queued status in the full development record.
- **Validation**: Checked the launch wrapper with `bash -n`; H200 preflight job
  `cosmos3-edge-libero-10k-fa3-preflight-r2` proved Torch 2.10/CUDA 12.8 and
  finite forced-FA3 forward/backward kernels before the formal submission.

## DEV-0010 — 2026-08-04 01:37 +08:00 — Force Flash Attention 3 for Edge LIBERO

- **Area**: Cosmos3-Edge LIBERO training launcher and attention-backend
  validation.
- **Summary**: Forced the Edge recipe to use only Flash Attention 3 and added a
  fail-fast GPU preflight that proves SM90 support, package compatibility,
  backend selection, and finite FA3 forward/backward kernels before training.
- **Documentation**: Updated the public LIBERO training guide with the strict
  FA3 requirements and recorded the complete H200 job and environment evidence
  in the development document.
- **Validation**: Ran Ruff, format, focused pytest, shell syntax, and diff
  checks; `cosmos3-edge-libero-fa3-preflight-r8` also passed the real kernel
  check on an NVIDIA H200 with Torch 2.10.0+cu128 and CUDA 12.8.

## DEV-0009 — 2026-08-03 23:41 +08:00 — Reduce Edge checkpoint frequency

- **Area**: Cosmos3-Edge LIBERO checkpointing, qualitative visualization, and
  attention-backend audit.
- **Summary**: Changed periodic checkpoint and EMA rollout cadence from 500 to
  2000 steps while preserving the trainer's final step-5000 checkpoint; audited
  Flash Attention availability and recorded that the migrated environment does
  not currently provide the FA3/FA2 packages.
- **Documentation**: Updated the public LIBERO post-training guide and the full
  development record with the new cadence, final-save behavior, and
  evidence-bounded attention conclusion.
- **Validation**: Ran Ruff, format, TOML parsing, launcher syntax, diff checks,
  the focused Edge recipe tests, and static checks for trainer final-save and
  H200 backend-priority behavior.

## DEV-0008 — 2026-08-03 23:24 +08:00 — Centralize experiment outputs

- **Area**: Remote experiment storage policy.
- **Summary**: Required every generated training, inference, checkpoint, log,
  offline W&B, export, visualization, and evaluation artifact to use
  `/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output` as its single
  output root, with per-project and per-run subdirectories.
- **Documentation**: Updated `AGENTS.md` and recorded the experiment-output
  location policy in this changelog.
- **Validation**: Ran `git diff --check`, verified the exact output root in
  `AGENTS.md`, and confirmed that the focused commit contains only the root
  instructions and this changelog entry.

## DEV-0007 — 2026-08-03 22:46 +08:00 — Make workspace path portable

- **Area**: Remote development documentation.
- **Summary**: Replaced the host-specific development-documentation path with
  the repository-relative `dev/` path so the instructions remain portable when
  the workspace moves.
- **Documentation**: Updated `AGENTS.md` and recorded the relative-path policy
  in this changelog.
- **Validation**: Ran `git diff --check` and verified that `AGENTS.md` contains
  no absolute shared-storage path.

## DEV-0006 — 2026-08-03 20:07 +08:00 — Complete Edge LIBERO training and export

- **Area**: Cosmos3-Edge LIBERO training, qualitative visualization, model
  export, and real-sample action-policy validation.
- **Summary**: Added portable PyAV decoding, checkpoint-aligned EMA rollout
  visualization, resolved-config export compatibility, action-policy metadata,
  bundled Edge assets, and final 5000-step training and export evidence.
- **Documentation**: Updated the generic LIBERO guide for PyAV and qualitative
  rollout behavior, and synchronized complete PJLAB job IDs, checkpoints,
  failures, metrics, export contents, and forward evidence in the dev record.
- **Validation**: Ruff, format, shell syntax, TOML parsing, and diff checks
  passed; 16 focused tests passed in the isolated CUDA 12.8 environment; the
  8×H200 run reached iteration 5000; the final 7.3 GiB export loaded with its
  bundled Edge assets; and a real merged-LIBERO sample produced finite
  `[1, 8, 10]` actions.

## DEV-0005 — 2026-07-30 18:53 +08:00 — Add Cosmos3-Edge LIBERO recipe

- **Area**: Cosmos3-Edge action-policy post-training and offline checkpoint
  preparation.
- **Summary**: Added the merged 10 FPS LIBERO Edge experiment, FSDP8 TOML,
  launcher, registration, configuration tests, and offline-capable local
  Edge/VAE HF-to-DCP conversion and training processor selection.
- **Documentation**: Updated the generic LIBERO and training guides plus the
  examples index, and synchronized the full PJLAB design, paths, rjob history,
  failures, and validation evidence in `dev/cosmos3_edge_libero_finetune.md`.
- **Validation**: Ruff, format, shell syntax, diff, focused tests, all shipped
  example TOML schema tests, a real-dataset launcher mock, and H200 config
  dry-run passed; the DCP was built entirely from GPFS-local artifacts.

## DEV-0004 — 2026-07-30 16:27 +08:00 — Support merged 10 FPS LIBERO data

- **Area**: LIBERO action-policy dataset loading and action normalization.
- **Summary**: Added an explicit wrist-camera feature key, deterministic 7D
  axis-angle to 10D rot6d statistics generation, the dataset-specific
  normalizer, and focused tests for the 8-action/9-frame window.
- **Documentation**: Created `dev/cosmos3_edge_libero_finetune.md` with the
  design, dataset schema, source fingerprint, cluster paths, and validation.
- **Validation**: Six focused tests passed; Ruff, format, and diff checks
  passed; a real PyAV-decoded sample had shape `[3,9,256,512]` with finite
  `[8,10]` actions.

## DEV-0003 — 2026-07-30 14:31 +08:00 — Centralize development documentation

- **Area**: Repository documentation organization.
- **Summary**: Required development plans, design notes, implementation records,
  experiment notes, review notes, and changelogs to live under `dev/`.
- **Documentation**: Updated the repository workflow rules in `AGENTS.md` and
  recorded the policy in this changelog.
- **Validation**: Checked the Markdown structure, sequential change IDs, the
  absolute `dev/` path, and the scoped Git diff.

## DEV-0002 — 2026-07-30 14:22 +08:00 — Align changelog with Git commits

- **Area**: Repository governance and development history.
- **Summary**: Established a one-to-one mapping between changelog entries and
  Git commits using shared, monotonically increasing `DEV-NNNN` identifiers and
  timestamped entries.
- **Documentation**: Updated `AGENTS.md` and the changelog format and template.
- **Validation**: Checked Markdown structure, sequential IDs, timestamps, and the
  staged Git diff.

## DEV-0001 — 2026-07-30 14:16 +08:00 — Establish repository development workflow

- **Area**: Repository governance and development process.
- **Summary**: Required Git-based development, local-only GitHub credential
  handling, focused commits and pushes after completed updates, and synchronized
  documentation maintenance.
- **Documentation**: Updated `AGENTS.md` and created this development changelog.
- **Validation**: Reviewed the Markdown structure and verified the scoped Git
  diff. This historical entry maps to commit
  `001d3082e4668f740b85481ccdbaf10276f28a5c`, created before the shared-ID rule.
