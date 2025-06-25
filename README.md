# BenchMAC

[![CI](https://github.com/SuperMuel/BenchMAC/actions/workflows/ci.yml/badge.svg)](https://github.com/SuperMuel/BenchMAC/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/SuperMuel/BenchMAC/branch/main/graph/badge.svg)](https://codecov.io/gh/SuperMuel/BenchMAC)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**BenchMAC** is a benchmark for evaluating AI agents on their ability to perform complex, real-world **M**igration tasks for **A**ngular **C**odebases.

## 🚀 Introduction

Migrating Angular applications across major versions is a nuanced task that goes beyond simple dependency updates. It requires adapting to breaking API changes, refactoring code and tests, and ensuring the entire project remains buildable, lint-free, and functionally correct.

BenchMAC provides a standardized, robust, and realistic methodology to evaluate and compare AI Systems Under Test (SUTs) on their capability to perform these comprehensive Angular version migrations.

### Key Features

*   **Realistic Tasks:** Operates on real-world, open-source Angular projects.
*   **Holistic Evaluation:** Measures success through a suite of metrics including build success, linting integrity, and test pass rates.
*   **Reproducible Environments:** Uses a containerized Docker environment to ensure every evaluation is isolated and scientifically valid.
*   **Decoupled Evaluation:** Employs a post-hoc model where the harness evaluates a static patch file, making it easy to test any SUT.
*   **Anti-Cheating Mechanisms:** Actively detects and penalizes superficial fixes like commenting out failing tests or disabling lint rules.

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
