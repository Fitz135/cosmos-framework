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
