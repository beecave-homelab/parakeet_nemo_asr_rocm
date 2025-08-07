import os

from parakeet_nemo_asr_rocm.utils import env_loader


def test_load_project_env_dotenv(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\n")

    def fake_load_dotenv(dotenv_path, override):
        os.environ["FOO"] = "bar"
        fake_load_dotenv.called = True

    fake_load_dotenv.called = False
    monkeypatch.setattr(env_loader, "_ENV_FILE", env_file)
    monkeypatch.setattr(env_loader, "LOAD_DOTENV", fake_load_dotenv)
    env_loader.load_project_env(force=True)
    assert fake_load_dotenv.called
    assert os.getenv("FOO") == "bar"


def test_load_project_env_manual(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("HELLO=world\n")
    monkeypatch.setattr(env_loader, "_ENV_FILE", env_file)
    monkeypatch.setattr(env_loader, "LOAD_DOTENV", None)
    monkeypatch.delenv("HELLO", raising=False)
    env_loader.load_project_env.cache_clear()
    env_loader.load_project_env()
    assert os.getenv("HELLO") == "world"


def test_load_project_env_no_file(monkeypatch, tmp_path):
    missing = tmp_path / "missing.env"
    monkeypatch.setattr(env_loader, "_ENV_FILE", missing)
    env_loader.load_project_env(force=True)
    assert True  # simply ensure no crash
