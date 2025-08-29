# BenchMAC Experiments

This directory contains reference implementations and tools for running AI agents (Systems Under Test) against the BenchMAC benchmark instances.

The goal is to provide a practical, easy-to-use starting point for generating `submissions.jsonl` files, which can then be evaluated by the main benchmark harness.

## `run_experiment.py`: A Reference SUT Runner

This script uses [`swe-agent-mini`](https://github.com/SWE-agent/mini-swe-agent) as a lightweight agentic framework to solve benchmark tasks. It orchestrates the entire process:

1.  Reads benchmark tasks from `data/instances.jsonl`.
2.  For each task, it creates an isolated sandbox environment.
3.  It clones the specified repository and checks out the correct commit.
4.  It invokes an LLM via `swe-agent-mini` with a prompt to perform the Angular migration.
5.  After the agent finishes, it generates a patch file of the changes.
6.  It writes the result to `submissions.jsonl`.

### Quickstart

**1. Install Dependencies**

This script has a few dependencies, including `swe-agent-mini`. 
They are defined in `../pyproject.toml` under the `experiments` group.
They should be installed in your current virtual environment if you ran `uv sync`

**2. Set API Keys**

The script uses `litellm` to connect to various LLM providers. Configure your API keys as environment variables.

For **Mistral**:
```bash
export MISTRAL_API_KEY="your-mistral-api-key"
```
For **OpenAI**:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```
For **Anthropic**:
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

**3. Run an Experiment**

Execute the script from the project root, specifying the model you want to test.

**Example using Mistral's Devstral:**
```bash
uv run experiments/run_experiment.py --model-name "mistral/devstral"
```

**Example using OpenAI's GPT-5-mini:**
```bash
uv run experiments/run_experiment.py --model-name "openai/gpt-5-mini"
```

**Example using Anthropic's Claude 4 Sonnet:**
```bash
uv run experiments/run_experiment.py --model-name "claude-4-sonnet-20250514"
```

**4. Evaluate the Results**

Once the script finishes, a `submissions.jsonl` file will be created in the `experiments/` directory. You can now evaluate it using the BenchMAC harness:

```bash
uv run benchmac evaluate experiments/submissions.jsonl
```
