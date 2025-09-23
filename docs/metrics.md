# BenchMAC Metrics

BenchMAC focuses on execution-based signals so migrations are judged by how well the upgraded repository behaves, not by how similar the diff looks to a reference.

## Primary Metrics

| Metric | Definition | Success Criteria |
| :-- | :-- | :-- |
| Patch application | Git can apply the submitted diff without conflicts | `git apply` exits 0 and leaves the repo in a clean state |
| Target version attainment | Declared framework dependencies match the target major | `package.json`, `angular.json`, and lockfiles reflect the requested Angular release |
| Build success | Canonical `build` command exits 0 | The project compiles using the pinned toolchain |

Each instance specifies the command sequence in `data/instances.jsonl`. Lint may be omitted when a project lacks lint tooling; in that case the metric is marked as _not evaluated_.

## Supporting Signals

Beyond the primary scores, the harness records:

- stdout/stderr logs for every command
- exit codes and timestamps
- the diff that was applied and the resulting Git status
- Docker image metadata (Node.js, npm versions, snapshot date)

These artifacts provide the evidence needed to categorize failures (e.g., dependency conflicts, outdated template syntax, brittle unit tests) and to debug harness regressions.

## Scoring Philosophy

- **Execution over heuristics:** The harness runs the real toolchain rather than inferring success from static analysis.
- **Deterministic baselines:** Every instance starts from a “green” commit that installs and builds successfully; regressions are therefore attributable to the migration.
- **Comparable results:** Because evaluation happens in identical containers, scores from different SUTs are directly comparable and reproducible.

Future versions may layer in cost or runtime tracking, but functional success remains the north star.
