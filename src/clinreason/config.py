"""Load configuration from a YAML file with environment-variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ENV_PREFIX = "CLINREASON_"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _coerce(raw: str, current: Any) -> Any:
    """Convert an environment string to the type of the existing value."""
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the config dict.

    Values come from ``path`` (default: ``configs/default.yaml``); each key can be
    overridden by ``CLINREASON_<KEY>``. ``data_dir`` is expanded to an absolute path.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    for key, value in list(cfg.items()):
        env_val = os.environ.get(ENV_PREFIX + key.upper())
        if env_val is not None:
            cfg[key] = _coerce(env_val, value)

    if "data_dir" in cfg:
        cfg["data_dir"] = str(Path(os.path.expanduser(str(cfg["data_dir"]))).resolve())
    return cfg
