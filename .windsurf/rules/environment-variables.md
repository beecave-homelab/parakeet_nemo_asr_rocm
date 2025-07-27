---
trigger: model_decision
description: Environment variables are loaded once at startup via utils.env_loader.load_project_env (called only in utils.constant), which defines constants. All other modules import these constants from utils.constant rather than reading env vars directly.
---

# Workplace Rule: environment-variables

1. Single Loading Point
   • Environment variables must be parsed exactly once at application start.
   • The loader function is `load_project_env()` in `parakeet_nemo_asr_rocm/utils/env_loader.py`.

2. Central Import Location
   • `load_project_env()` MUST be invoked only inside `parakeet_nemo_asr_rocm/utils/constant.py`.
   • No other file should import `env_loader` or call `load_project_env()` directly.

3. Constant Exposure
   • After loading, `utils/constant.py` exposes project-wide configuration constants (e.g. `DEFAULT_CHUNK_LEN_SEC`, `DEFAULT_BATCH_SIZE`).
   • All other modules (e.g. `parakeet_nemo_asr_rocm/app.py`, `parakeet_nemo_asr_rocm/transcribe.py`) must import from `parakeet_nemo_asr_rocm.utils.constant` instead of reading `os.environ` or `.env`.

4. Adding new variables
   • Define a sensible default in `utils/constant.py` (`os.getenv("VAR", "default")`).
   • Document the variable in `.env.example`.

5. Enforcement
   • Pull requests adding direct `os.environ[...]` or `env_loader` imports outside `utils/constant.py` should be rejected.
