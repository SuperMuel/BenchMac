# Key Concepts in BenchMAC

This document defines the core terminology and components of the BenchMAC system.

## Core Data Objects

### Instance
An `Instance` is a single, self-contained migration task. It represents the problem to be solved. It is also referred to as a "Task Instance". An Instance is defined by:
*   **`instance_id`**: A unique identifier (e.g., `gothinkster__angular-realworld-example-app_v15_to_v16`).
*   **Repository**: The source code repository and a specific `base_commit`.
*   **Versions**: The `source_angular_version` and `target_angular_version`.
*   **Commands**: The exact shell commands for `install` and `build`.
*   **Environment**: A dedicated Dockerfile that creates the execution environment.

The canonical list of all instances is located in `data/instances.jsonl`.

### Submission
A `Submission` is the output produced by an Agent for a specific Instance. A Submission consists of:
*   **`submission_id`**: A unique identifier for the attempt.
*   **`instance_id`**: The Instance this submission attempts to solve.
*   **`model_patch`**: The unified diff containing all proposed code changes. The name `model_patch` is used to distinguish the AI-generated solution from a human-authored one.

A Submission is the primary artifact that the Harness evaluates.

### Patch
A `Patch` is a unified diff file (`.patch`) that describes all code changes. It serves as the standard interface between the Experiment Framework and the Harness. This design decouples the process of generating a solution from the process of evaluating it, ensuring fair and consistent scoring.

## Evaluation Process and Artifacts

### Evaluation
`Evaluation` is the automated process of scoring a Submission against its corresponding Instance. The standard workflow is:
1.  Start a new, clean container using the Instance's reproducible environment.
2.  Apply the Submission's Patch to the baseline source code.
3.  Execute the series of canonical commands defined in the Instance (install, build).

### Execution Trace
An `Execution Trace` is a detailed, structured log of an Evaluation. It contains an ordered list of `CommandResult` objects. Each result includes the command executed, its exit code, stdout, stderr, and timing information. It is the raw source of truth for calculating Metrics.

### Metrics
`Metrics` are a set of objective, execution-based scores derived from an Execution Trace. Key metrics are tri-state (True, False, or None if not applicable).
*   **`patch_application_success`**: The patch applied cleanly.
*   **`install_success`**: Project dependencies installed successfully.
*   **`target_version_achieved`**: The project's Angular version matches the target.
*   **`build_success`**: The project compiled successfully.

## System Components

### Harness
The `Harness` is the core evaluation engine of BenchMAC. Its responsibility is to ingest Submissions, perform Evaluations in isolated environments, and produce Execution Traces and Metrics. It is also referred to as the "Benchmark Harness". The source code for the Harness is located in `src/bench_mac/`.

### Experiment Framework
The `Experiment Framework` is the toolchain for running Agents to produce Submissions. It manages agent configurations, orchestrates agent execution within the correct Docker environments, and saves experiment results and artifacts like cost and agent-specific logs. The source code is located in `experiments/`.

### Agent
An `Agent` is the System Under Test (SUT). It is any automated system that takes an Instance as input and produces a Patch as output. Agents can be AI-driven (e.g., LLM-based) or rule-based (e.g., Angular Schematics).

## Foundational Concepts

### Silver Patch
A `Silver Patch` is a known-good reference Patch for an Instance, derived from the repository's human-authored commit history. The term "Silver" is used deliberately instead of "Golden" to emphasize that it is a valid but not necessarily unique solution for a complex migration task.

A Silver Patch must pass all evaluation metrics. Its primary purpose is to validate the Harness itself and confirm that a perfect score is achievable for a given Instance. They are a developer tool for harness validation, not a golden oracle for agent evaluation. They are stored in `data/silver_patches/`.

### Reproducible Environment
A `Reproducible Environment` is an isolated and deterministic execution environment provided by a Docker container. Reproducibility is achieved through several mechanisms:
*   A dedicated Dockerfile for each Instance.
*   Base images pinned to an immutable SHA256 digest.
*   System-level dependencies (via `apt`) frozen to a specific date using `snapshot.debian.org`.
*   A history-free snapshot of the source code, preventing agents from accessing future commits.