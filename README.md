# BenchMAC

[![CI](https://github.com/SuperMuel/BenchMAC/actions/workflows/ci.yml/badge.svg)](https://github.com/SuperMuel/BenchMAC/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/SuperMuel/BenchMAC/branch/main/graph/badge.svg)](https://codecov.io/gh/SuperMuel/BenchMAC)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**BenchMAC** is a benchmark for evaluating AI agents on their ability to perform complex, real-world **M**igration tasks for **A**ngular **C**odebases.

Developped as part of my Master's thesis in collaboration with **[onepoint](https://groupeonepoint.com/)**.

## 🚀 Introduction

Migrating Angular applications across major versions is a nuanced task that goes beyond simple dependency updates. It requires adapting to breaking API changes, refactoring code and tests, and ensuring the entire project remains buildable, lint-free, and functionally correct.

BenchMAC provides a standardized, automated, and reproducible way to measure the performance of any AI system on these tasks.

### Key Features

* **Realistic Tasks:** Operates on real-world, open-source Angular projects.
* **Holistic Evaluation:** Measures success through a suite of metrics including build success, linting integrity, and test pass rates.
* **Reproducible Environments:** Uses a containerized Docker environment to ensure every evaluation is isolated and scientifically valid.
* **Decoupled Evaluation:** Employs a post-hoc model where the harness evaluates a static patch file, making it easy to test any SUT.

## Why BenchMAC

Angular upgrades are strategic modernization efforts. Delaying them increases security risk, fractures toolchains, and slows delivery. Existing LLM benchmarks mostly target backend code or isolated functions, so they fail to capture the repository-scale coordination, template changes, and ecosystem churn that make Angular migrations difficult. BenchMAC fills that gap with reproducible, apples-to-apples comparisons of systems attempting the same real upgrade.

## How It Works

BenchMAC separates migration **generation** from **evaluation** so any system under test (SUT) can plug in while results stay comparable.

1. **Patch Generation:** An SUT works inside an instance-specific Docker image. It can run commands, iterate on errors, and make arbitrary edits. Its only obligation is to output a unified diff describing the final migration.
2. **Patch Evaluation:** The harness resets the repository to the instance baseline, applies the diff, and runs the canonical install/build/test command sequence. All outputs are collected to compute metrics and capture failure evidence.

This architecture lets researchers innovate on agents, prompts, or rules without touching the evaluator, and keeps scoring deterministic across submissions.

## Evaluation Metrics

| Metric | What it checks | Why it matters |
| :-- | :-- | :-- |
| Patch application | Diff applies cleanly | Detects syntactic conflicts and missing files before deeper checks |
| Target version attainment | Framework versions match the goal | Ensures the upgrade actually lands on the requested release |
| Build success | Canonical `build` command exits 0 | Confirms the migrated project compiles in CI-like conditions |

Each run also archives command logs and agent output so teams can trace failure modes.

## Dataset at a Glance

BenchMAC v1.0 ships nine instances drawn from `gothinkster/angular-realworld-example-app`, covering consecutive upgrades from Angular 11→12 through 19→20. Leveraging real Git histories lets us define multiple tasks from one repository while keeping every baseline green and reproducible. The dataset definition lives in [`data/instances.jsonl`](data/instances.jsonl) with matching Dockerfiles under [`data/dockerfiles/`](data/dockerfiles/).

To keep the harness honest we also extract “silver” patches—human-authored migrations from the same history. Applying a silver patch must succeed end-to-end; failures indicate harness or environment drift rather than AI errors.

## Reproducible Environments

Every instance has a pinned Docker image that:

* Downloads a history-free archive of the baseline commit
* Pins Node.js and npm by digest, then freezes Debian packages via `snapshot.debian.org`
* Records toolchain versions for auditing and tags a baseline Git commit

The harness and the agents both rely on these images, ensuring that patches generated today will be evaluated the same way tomorrow.

## ⚙️ Installation

To get started with the BenchMAC evaluation harness, follow these steps.

### Prerequisites

*   [UV](https://docs.astral.sh/uv/)
*   [Docker](https://docs.docker.com/get-docker/) installed and running.

### Setup

1.  **Install dependencies:**
    ```bash
    uv sync
    ```

2.  **Set up pre-commit hooks:**
    ```bash
    uv run pre-commit install
    ```
    
    This ensures that code is automatically linted, formatted, and the lockfile stays up to date before each commit.

## 📘 Documentation

- [Architecture](docs/architecture.md)
- [Metrics](docs/metrics.md)
- [Dataset](docs/dataset.md)
- [Reproducibility](docs/reproducibility.md)
- [Add a new instance](docs/add-new-instance.md)

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📚 Citation


```bibtex
@misc{mallet2024benchmac,
  author       = {Samuel Mallet},
  title        = {BenchMAC: A Benchmark for Evaluating AI-Assisted Angular Version Migration},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/SuperMuel/BenchMAC}}
}
```
