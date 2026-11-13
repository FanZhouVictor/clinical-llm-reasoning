from pathlib import Path

import clinreason
from clinreason.config import load_config


def test_version_is_set():
    assert clinreason.__version__


def test_default_config_loads():
    cfg = load_config()
    assert {"data_dir", "llm_base_url", "llm_model", "seed"} <= cfg.keys()
    assert Path(cfg["data_dir"]).is_absolute()


def test_env_override_and_type_coercion(monkeypatch, tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("seed: 1\nllm_temperature: 0.0\nllm_model: a\ndata_dir: x\n")
    monkeypatch.setenv("CLINREASON_SEED", "42")
    monkeypatch.setenv("CLINREASON_LLM_TEMPERATURE", "0.5")
    monkeypatch.setenv("CLINREASON_LLM_MODEL", "b")
    monkeypatch.setenv("CLINREASON_DATA_DIR", str(tmp_path / "data"))
    cfg = load_config(cfg_file)
    assert cfg["seed"] == 42
    assert cfg["llm_temperature"] == 0.5
    assert cfg["llm_model"] == "b"
    assert cfg["data_dir"] == str((tmp_path / "data").resolve())
