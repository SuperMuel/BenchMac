# Reproducibility Playbook

Deterministic environments are central to BenchMAC. Both agent runs and harness evaluations rely on the same container images so that a migration tested today will behave identically in the future.

## Docker Images

Each instance supplies a Dockerfile that:

1. Pins the Node.js base image by SHA256 digest (`FROM node:18-bookworm-slim@sha256:…`).
2. Configures Debian APT mirrors via `snapshot.debian.org` and disables validity checks so package versions never drift.
3. Installs only the minimal system dependencies required to run Angular CLI workflows.
4. Records `node --version` and `npm --version` for auditability.
5. Creates `/app/project`, downloads a history-free tarball of `base_commit`, and initializes a single-commit Git repo tagged `baseline`.

## Baseline Green Principle

Before an instance ships, the validation suite confirms that:

- `install` and `build` commands succeed on the untouched baseline
- no Git history leaked into the image (only the baseline commit is present)

This guarantees that evaluation failures stem from the migration diff, not from environmental drift or pre-existing breakage.

## Shared Between Agents and Harness

The same image powers both stages:

- Agents spawn containers to explore, edit, and generate patches.
- The harness starts a fresh container per submission to apply the diff and run the canonical commands.

Because the image is immutable and pinned, agents cannot rely on mutable global state, and results are directly comparable across different systems and over time.
