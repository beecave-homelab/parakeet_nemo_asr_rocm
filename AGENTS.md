# AGENTS.md – OpenAI Codex & AI Agent Guide for parakeet_nemo_asr_rocm

This file provides authoritative instructions for OpenAI Codex and all AI agents working within this repository. It documents project structure, coding conventions, environment variable policy, testing protocols, and PR guidelines. **All agents must strictly adhere to these rules for any code or documentation changes.**

---

## 1. Project Structure

- **Root Directory**
  - `Dockerfile`, `docker-compose.yaml`: Containerization for ROCm/NeMo ASR service.
  - `pyproject.toml`: Python dependencies (PDM-managed).
  - `.env.example`: Environment variable template.
  - `README.md`: Quick-start and usage.
  - `project-overview.md`: In-depth codebase and architecture documentation.
- **parakeet_nemo_asr_rocm/**
  - `cli.py`: Typer CLI entry point.
  - `transcribe.py`: Batch transcription logic.
  - `chunking/merge.py`: Segment merging for long audio.
  - `timestamps/segmentation.py`: Subtitle segmentation.
  - `formatting/`: Output formatters (SRT, TXT, etc.).
  - `utils/`: Audio I/O, file utils, constants, and env loader.
  - `models/parakeet.py`: Model wrapper.
- **tests/**: Unit and integration tests for all major modules.
- **scripts/**: Utility scripts for requirements and dev shell.
- **data/**: Sample audio and output directory.

---

## 2. Coding Standards (MANDATORY)

All code changes must strictly follow these rules. **If code violates any rule, auto-correct before outputting.**

### Docstrings & Documentation

- Use **Google Style docstrings** for all functions, classes, and methods (including private/internal).
- Each docstring must include:
  - **Args**: List all input parameters with types and concise descriptions.
  - **Returns**: Describe the return value and its type.
  - **Raises** (if applicable): Document any exceptions the function may raise.

### Type Hints

- All function arguments and return values **must be annotated** with explicit type hints.

### Code Style

- Adhere to **PEP 8** standards:
  - 4-space indentation
  - Line length ≤ 79 characters
  - Snake_case for variables/functions, PascalCase for classes
  - Use `isort` and `black` compatible formatting
- **Never** use wildcard imports (`from module import *`).

### Imports

- Use **absolute (full-path) imports** consistently.
- Organize imports in this order with blank lines in between:
  1. Standard library
  2. Third-party packages
  3. Local application imports

### Error Handling & Auto-Fixing

- If code violates any of these rules, **fix it before outputting** (add missing type hints, convert relative imports to absolute, reformat code, rewrite docstrings, etc.).
- Always return Python files that are **PEP8-compliant, type-safe, properly documented, and consistently structured**.

---

## 3. Environment Variables Policy (STRICT)

- **Single Loading Point**: Environment variables must be parsed exactly once at application start using `load_project_env()` in `parakeet_nemo_asr_rocm/utils/env_loader.py`.
- **Central Import Location**: `load_project_env()` MUST be invoked only in `parakeet_nemo_asr_rocm/utils/constant.py`. No other file may import `env_loader` or call `load_project_env()` directly.
- **Constant Exposure**: After loading, `utils/constant.py` exposes all project-wide configuration constants (e.g., `DEFAULT_CHUNK_LEN_SEC`, `DEFAULT_BATCH_SIZE`). All other modules must import from `parakeet_nemo_asr_rocm.utils.constant` and must **never** read `os.environ` or `.env` directly.
- **Adding New Variables**: Define a sensible default in `utils/constant.py` (e.g., `os.getenv("VAR", "default")`), and document the variable in `.env.example`.
- **Enforcement**: PRs adding direct `os.environ[...]` or `env_loader` imports outside `utils/constant.py` **must be rejected**.

---

## 4. Dependency & Environment Management (PDM & pyproject.toml)

- **PDM is the canonical tool for all dependency, environment, and script management.**
- All dependencies (including dev and optional extras) are managed exclusively via `pyproject.toml`.
- **To add or update a dependency:**
  - Use `pdm add <package>` (for runtime), `pdm add -d <package>` (for dev), or `pdm add -G rocm <package>` (for ROCm extras).
  - Never edit `requirements-all.txt` or use pip directly.
  - After any dependency changes, run `pdm lock` and commit the updated lockfile.
- **To install the project and all dependencies:**

  ```bash
  pdm install
  # For ROCm GPU support:
  pdm install -G rocm
  # For development tools:
  pdm install -G dev
  ```

- **To run scripts or tools:**
  - Use `pdm run <script>` (e.g., `pdm run lint`, `pdm run type-check`, `pdm run parakeet-rocm`).
  - All CLI entry points are defined in `[project.scripts]` in `pyproject.toml`.
- **To update dependencies:**
  - Use `pdm update` or `pdm update <package>`.
- **Custom sources:**
  - Agents must respect all `[tool.pdm.source]` definitions in `pyproject.toml`, including custom URLs for ROCm wheels and PyPI.
- **Never** use pip, requirements.txt, or venv directly—always use PDM.
- **Document any new dependencies or scripts in `pyproject.toml` and update onboarding docs if needed.**

---

## 5. Testing Protocols

- All new or modified code must be covered by tests in the `tests/` directory.
- Use `pytest` as the test runner.
- To run all tests:

  ```bash
  pytest
  ```

- To run a specific test file:

  ```bash
  pytest tests/test_transcribe.py
  ```

- All tests must pass before any code is merged.

---

## 6. Pull Request (PR) Guidelines

- PRs must include:
  1. A clear, descriptive summary of the change.
  2. References to any related issues or TODOs.
  3. Evidence that all tests pass (paste output or CI link).
  4. Documentation updates as required by the change.
  5. PRs should be focused and address a single concern.
- PRs that violate the environment variable policy or coding conventions **will be rejected**.

---

## 7. Programmatic Checks

Before submitting or merging changes, always run:

```bash
# Linting (PEP8 compliance)
pdm run lint

# Run all tests
pytest

# Optionally, check type hints
pdm run type-check
```

All checks must pass before code is merged.

---

## 8. AGENTS.md Scope and Precedence

- This AGENTS.md applies to the entire repository.
- If more deeply nested AGENTS.md files are added, they take precedence for their directory tree.
- Direct developer instructions in a prompt override AGENTS.md, but agents must always follow programmatic checks and project policies.
