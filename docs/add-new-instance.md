# How to Add a New Benchmark Instance to BenchMAC

Welcome! This guide walks you through adding a new migration task (“instance”) to the BenchMAC benchmark. An **instance** defines a real repo at a specific commit (the *source version*) and a target Angular major version to migrate to. The harness then evaluates a submitted patch that performs the migration.

> TL;DR: You’ll add one JSONL line to `data/instances.jsonl` and create one Dockerfile in `data/dockerfiles/`, then run the validation tests.

---
## What Makes a Good BenchMAC Instance?

We want instances that are realistic, reproducible, and informative for evaluating AI-assisted **major** migrations of Angular apps.

### Must-haves

1. **Open-source repo with a clear license**

   * Prefer permissive: MIT, Apache-2.0, BSD-3, MPL-2.0
   * A top-level `LICENSE` file or SPDX in package metadata

2. **Real application** (not a tiny demo or library)

3. **Runnable baseline**

   * At `base_commit`, the declared `install` and `build` and other commands succeed in our harness environment (see validation step)

4. **Deterministic & self-contained**

   * No secrets required, no external paid services to build

5. **Reasonable size**


### Nice-to-haves (quality hints)

* Stars/usage (community traction), active CI, unit tests present
* Lint tooling present (`eslint` / configs)
* Uses standard Angular build pipeline (not heavily customized builders)

### Multiple instances from the same repo

Great! We encourage consecutive majors (vN → vN+1) as **separate instances** if:

* You can identify a clean `base_commit` for each source major
* Each baseline is green


## 1) Pick an instance ID and commits

### Naming convention

Use:

```
<owner>__<repo>_v<sourceMajor>_to_v<targetMajor>
```

Examples (existing):

* `gothinkster__angular-realworld-example-app_v15_to_v16`
* `gothinkster__angular-realworld-example-app_v18_to_v19`

Notes:

* Replace `/` in `owner/repo` with `__`.
* Use **major** versions in the suffix (even if you later use full semver in fields).

### Choose the commit

* **base\_commit**: a commit that represents the starting (source) version state.
  This is the snapshot we put inside the Docker image (the repository code the agent will patch).
---

## 2) Create the Dockerfile for the instance

Create:
`data/dockerfiles/<instance_id>`

```dockerfile
# Pick a Node image compatible with the source Angular version
# (see Angular’s compatibility docs for exact ranges)
FROM node:18-bookworm-slim@sha256:<digest-from-docker-hub>

# Minimal, repeatable CI-ish environment
ENV CI=true \
    TZ=UTC \
    LANG=C.UTF-8 \
    NG_CLI_ANALYTICS=false \
    NPM_CONFIG_AUDIT=false \
    NPM_CONFIG_FUND=false

# Freeze apt to the day you validated the instance (snapshot.debian.org keeps historical mirrors)
RUN echo 'deb http://snapshot.debian.org/archive/debian/<yyyymmdd>T000000Z bookworm main contrib non-free' > /etc/apt/sources.list && \
    echo 'deb http://snapshot.debian.org/archive/debian-security/<yyyymmdd>T000000Z bookworm-security main contrib non-free' >> /etc/apt/sources.list && \
    echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99snapshot && \
    apt-get update && \
    apt-get install -y \
        curl \
        git \
        ca-certificates \
        --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Capture the toolchain versions baked into the image for later auditing
RUN node --version > /etc/benchmac-node-version && \
    npm --version > /etc/benchmac-npm-version

# Writable workspace (container keeps running as root for agent flexibility)
RUN mkdir -p /app/project
# This is where the repository will be available to the agent
WORKDIR /app/project

# Use a root-scoped npm cache so agents can install extra tooling
ENV NPM_CONFIG_CACHE=/root/.npm

# IMPORTANT: pull a history-free snapshot, NOT a git clone
# If the full git history is included, AI agents could "see the future" and cheat.
# Use your chosen base_commit here
RUN curl -L https://github.com/<owner>/<repo>/archive/<base_commit>.tar.gz \
    | tar -xz --strip-components=1

# Initialize git repo and create baseline commit for diffing
RUN git init --initial-branch=main && \
    git config --global --add safe.directory /app/project && \
    git config user.email "benchmark@benchmac.dev" && \
    git config user.name "BenchMAC Baseline" && \
    git add . && \
    git commit -m "baseline: <owner>/<repo>@<base_commit>" && \
    git tag baseline

# Keep container idle; the harness will exec into it
CMD ["bash", "-lc", "sleep infinity"]
```

> Snapshot hint: pick a date close to when you validated the instance (for example `20231101`). If your base image is built on Debian Bullseye instead of Bookworm, replace `bookworm` with `bullseye` in both snapshot URLs.

**Requirements & gotchas:**

* **Do not** `git clone` the repo. Always `curl | tar` an archive for a **history-free** snapshot.
* **Do not** run `npm ci`, builds, tests, or lint in the Dockerfile. The harness and AI agents does that.
* **Do create** a baseline git commit and tag after extracting source code. This allows the harness to get the progress using `git diff baseline`, even if the AI agent created commits.
* Choose a **Node image** compatible with your **source AND target** Angular version, and record the SHA256 digest when you pull it the first time so every rebuild sees identical bits.
* Keep images reproducible: pin apt through Debian snapshot mirrors, capture `node --version` / `npm --version`, and keep the container running as root so agents can install temporary tooling without permission friction.
* Remove optional packages (Chromium, etc.) unless an instance truly needs them—`npm ci` and build flows are the only operations we run by default.

---

## 3) Add a line to `data/instances.jsonl`

Append one JSON object (single line) with your instance:

```json
{
  "instance_id": "owner__repo_v15_to_v16",
  "repo": "owner/repo",
  "base_commit": "<7-40 char hex>",
  "source_angular_version": "15.2.3",
  "target_angular_version": "16",
  "commands": {
    "install": "npm ci",
    "build": "npx ng build --configuration production"
  }
}
```

### Field guidance

* **commands.install**:
  * For older Angular (e.g. v11–v14), you may set `"npm ci --legacy-peer-deps"`.
* **commands.build**:
  * Use whatever the project’s build requires, but keep it deterministic and non-interactive.


## 4) Validate your instance

Validates that each instance’s **baseline** (the Dockerfile snapshot) is “green enough” to run install & build as-is.

What it does:

* Builds the instance image.
* Starts a container.
* Verifies that the repo exists and can be downloaded.
* Verifies no git history was included to prevent cheating.
* Runs the declared `install`, `build`, etc. commands and expects all to succeed.

Verify that one instance is valid
```bash
  uv run pytest -m instance_validation tests/integration/test_baseline_validation.py -k "instance_id"
```

Verify that all the instances from a single repository are valid
```bash
  uv run pytest -m instance_validation tests/integration/test_baseline_validation.py -k "<owner>__<repo>"
```

## 5) Tips

### Research the target repository
Before creating your Dockerfile, research the repository's existing setup:

- **Review existing Dockerfiles**: Check if the repository already has Docker configuration
- **Examine CI/CD pipelines**: Look at `.github/workflows/` or other CI files to understand build requirements
- **Study BenchMAC examples**: Reference existing Dockerfiles in `data/dockerfiles/` for proven patterns and best practices

### Use an iterative development approach
Don't try to integrate the instance into the benchmark immediately. Instead, follow this development workflow:

1. **Clone locally**: Clone the codebase to a temporary directory for easy exploration (e.g., `.benchmac/temp-codebases`)
2. **Create standalone Dockerfile**: Build a new Dockerfile outside `data/dockerfiles/` to avoid disrupting the benchmark
3. **Iterate and test**:
   - Build the image and verify it works
   - Run a container and execute install/build commands
   - Analyze outputs and refine the Dockerfile as needed
4. **Integrate when ready**: Once your local tests pass, integrate the instance into the benchmark and run the full validation suite

---

## 6) PR checklist

* [ ] **Dockerfile** at `data/dockerfiles/<instance_id>`:

  * Uses `curl | tar` with the **base\_commit**.
  * Creates baseline git commit and tag for diffing.
  * Contains no baseline "work" (no `npm ci`, build, test, or lint).
  * Leaves container idle (e.g., `sleep infinity`).
* [ ] **Instance line** appended to `data/instances.jsonl` with correct fields.
* [ ] `uv run pytest -m instance_validation -k "<your new instance id>` passes.
