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
