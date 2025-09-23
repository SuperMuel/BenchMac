# BenchMAC Architecture

BenchMAC evaluates Angular migrations in two isolated stages so systems under test (SUTs) can experiment freely while the harness stays reproducible and fair.

## Stage 1 – Patch Generation

SUTs run inside an instance-specific Docker container that provides the baseline repository, pinned toolchain, and developer utilities. Within that sandbox an SUT may:

- inspect and edit files in `/app/project`
- execute the canonical commands (install, build, lint, test) for feedback
- run additional tools such as `ng update`, package managers, or custom scripts

The only requirement is to emit a **unified diff** (`.patch`) describing the proposed migration. This diff is the contract between the generation and evaluation stages; it allows:

- complete freedom for agent design, prompts, or rule-based tooling
- controlled evaluation without leaking an SUT's internal workflow
- simple archival and versioning of attempts for later analysis

### Compatible Systems

Any workflow that can produce a diff is compatible, including:

- rule-based utilities like Angular Schematics
- lightweight agent scaffolds (e.g., `mini-swe-agent`) that loop over "think → act → observe" steps
- full-featured commercial agents, provided they respect the sandbox and output a diff

Because the container ships without Git history, systems must rely on reasoning or documentation rather than looking ahead to future commits.

## Stage 2 – Patch Evaluation

The BenchMAC harness processes submissions independently of how they were generated:

1. Restore the repository to the baseline commit packaged with the instance.
2. Apply the submitted diff. Patch failures are captured as first-class metrics.
3. Execute the canonical command sequence (install → build).
4. Record stdout/stderr, exit codes, timestamps, and resource usage needed for downstream metrics.

Each evaluation runs in a brand-new container derived from the same image used during generation, guaranteeing the environment is identical across runs and over time.

## Artifacts and Observability

The harness stores:

- structured execution traces per command
- the applied diff and generated Git state
- metadata about the Docker image, Node.js version, and npm version

These artifacts enable qualitative analysis like classifying failure modes, comparing agent strategies, or reproducing regressions observed in CI.

## Extending the Harness

The decoupled design keeps evolution cheap:

- **New SUTs** plug into the generation stage without harness changes.
- **New metrics** or command hooks can be added in the evaluation stage while remaining backward compatible with existing submissions.
- **New instances** simply provide another Dockerfile and JSON line; the harness logic is shared across every task.
