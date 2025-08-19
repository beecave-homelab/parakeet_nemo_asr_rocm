"""Unit tests for environment loader behavior.

These tests exercise reading from a ``.env`` file via either python-dotenv
or manual parsing and ensure idempotent behavior when files are missing.
"""

import os
from pathlib import Path
from typing import Any

from parakeet_nemo_asr_rocm.utils import env_loader


def test_load_project_env_dotenv(monkeypatch: Any, tmp_path: Path) -> None:
    """When python-dotenv is available, it should be invoked.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching modules.
        tmp_path (pathlib.Path): Temporary directory for the test.

    Returns:
        None: This is a pytest test function.
    """

    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\n")

    def fake_load_dotenv(*_args, **_kwargs):
        os.environ["FOO"] = "bar"
        fake_load_dotenv.called = True

    fake_load_dotenv.called = False
    monkeypatch.setattr(env_loader, "_ENV_FILE", env_file)
    monkeypatch.setattr(env_loader, "LOAD_DOTENV", fake_load_dotenv)
    env_loader.load_project_env(force=True)
    assert fake_load_dotenv.called
    assert os.getenv("FOO") == "bar"


def test_load_project_env_manual(monkeypatch: Any, tmp_path: Path) -> None:
    """Manual parsing should set environment variables when dotenv is absent.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching modules.
        tmp_path (pathlib.Path): Temporary directory for the test.

    Returns:
        None: This is a pytest test function.
    """

    env_file = tmp_path / ".env"
    env_file.write_text("HELLO=world\n")
    monkeypatch.setattr(env_loader, "_ENV_FILE", env_file)
    monkeypatch.setattr(env_loader, "LOAD_DOTENV", None)
    monkeypatch.delenv("HELLO", raising=False)
    env_loader.load_project_env.cache_clear()
    env_loader.load_project_env()
    assert os.getenv("HELLO") == "world"


def test_load_project_env_no_file(monkeypatch: Any, tmp_path: Path) -> None:
    """Missing env files should not crash the loader.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching modules.
        tmp_path (pathlib.Path): Temporary directory for the test.

    Returns:
        None: This is a pytest test function.
    """

    missing = tmp_path / "missing.env"
    monkeypatch.setattr(env_loader, "_ENV_FILE", missing)
    env_loader.load_project_env(force=True)
    assert True  # simply ensure no crash
