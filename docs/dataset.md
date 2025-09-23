# BenchMAC Dataset

BenchMAC curates repository-level Angular upgrade tasks called **instances**. Each instance defines everything needed to reproduce a real migration: the source repository, a baseline commit, target version, canonical commands, and a Dockerfile describing the environment.

## Selection Principles

- **Reproducibility first:** Only commit snapshots that install and build cleanly inside the pinned container make the cut.
- **Open-source and licensed:** Instances use permissively licensed repositories so the benchmark can be shared and extended.
- **Application-centric:** Tasks target runnable Angular applications, not minimal demos or libraries, to exercise templates, configuration, and tests.
- **Sandbox friendly:** Projects that rely on external secrets or services are excluded; everything needed must ship in the container.

## BenchMAC v1.0

The inaugural release packages nine consecutive upgrades of `gothinkster/angular-realworld-example-app`, spanning Angular 11→12 through 19→20. Reusing a single well-maintained repo keeps validation costs reasonable while still surfacing framework churn across versions.

Instance metadata lives in [`data/instances.jsonl`](../data/instances.jsonl). Each line specifies:

- `instance_id`
- `repo`
- `base_commit`
- source and target Angular versions
- canonical `install`, `build`, optional `lint`, and optional `test` commands

Matching Dockerfiles reside in [`data/dockerfiles/`](../data/dockerfiles/); they are referenced by the harness at runtime.

## Silver Patches

For every instance we extract the human-authored migration diff from the repository’s history and store it as a **silver patch**. Applying the silver patch inside the harness must succeed end-to-end. If it fails, the issue is with the instance definition or environment, not with competing SUTs. Silvers therefore double as regression tests for benchmark maintenance.

## Growing the Dataset

Want to contribute a new instance? Follow the workflow in [`docs/add-new-instance.md`](add-new-instance.md):

1. identify candidate repository and commits
2. craft a pinned Dockerfile using `curl | tar` (no Git history)
3. append an entry to `data/instances.jsonl`
4. validate via `uv run pytest -m instance_validation`

Future roadmap items include broadening to additional repositories, covering earlier Angular majors, and introducing cross-project variety in testing depth and configuration complexity.
