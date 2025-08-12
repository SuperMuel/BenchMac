# Test Fixtures for Docker Strategy Refactor

This directory contains test fixtures for the ID-based Dockerfile lookup.

## Structure

```
tests/fixtures/
├── README.md                           # This file
├── test_instances.jsonl                # Test instances (no dockerfile_path field)
└── dockerfiles/                        # Test Dockerfiles directory
    ├── test-instance-v15-to-v16        # Valid test Dockerfile (Angular 15→16)
    ├── test-instance-v16-to-v17        # Valid test Dockerfile (Angular 16→17)
    └── invalid-dockerfile-instance     # Invalid Dockerfile for error testing
```

## ID-Based Dockerfile Lookup

The system uses a convention-based approach where Dockerfiles are located at:
```
dockerfiles/{instance_id}
```

No `dockerfile_path` field is needed in the instance data - the path is derived from the `instance_id`.

## Test Dockerfiles

### Valid Test Dockerfiles

- **test-instance-v15-to-v16**: Minimal Dockerfile for testing Angular 15→16 migration
- **test-instance-v16-to-v17**: Minimal Dockerfile for testing Angular 16→17 migration

These Dockerfiles:
- Use Node.js 18.19.0 on Debian Bullseye slim
- Include minimal system dependencies (git, chromium, etc.)
- Create a minimal Angular project structure instead of cloning real repos
- Are designed to be fast to build for testing purposes

### Invalid Test Dockerfile

- **invalid-dockerfile-instance**: Intentionally invalid Dockerfile for testing error handling
  - References a non-existent base image
  - Contains invalid commands
  - Used to test Docker build failure scenarios

## Test Instances

The `test_instances.jsonl` file contains 4 test instances:

1. **test-instance-v15-to-v16**: Valid instance with matching Dockerfile at `dockerfiles/test-instance-v15-to-v16`
2. **test-instance-v16-to-v17**: Valid instance with matching Dockerfile at `dockerfiles/test-instance-v16-to-v17`
3. **invalid-dockerfile-instance**: Instance with invalid Dockerfile at `dockerfiles/invalid-dockerfile-instance`
4. **missing-dockerfile-instance**: Instance with no corresponding Dockerfile (for testing error handling)

## Usage in Tests

These fixtures can be used in unit and integration tests:

```python
# Load test instances
test_instances_file = Path("tests/fixtures/test_instances.jsonl")
with open(test_instances_file) as f:
    test_instances = [json.loads(line) for line in f]

# Derive Dockerfile path from instance_id
instance_id = "test-instance-v15-to-v16"
dockerfile_path = Path("tests/fixtures/dockerfiles") / instance_id
```

## Validation

Run the validation script to ensure fixtures are properly structured:

```bash
python validate_all_data.py
```
