"""Stored parameter files (``params/*.yaml``) for the app's configuration page."""

import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

DEFAULT_PARAMS_DIR = Path("params")
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def check_name(name: str) -> str:
    """Configuration names are file names: letters, digits, '_', '-', '.'."""
    if not _NAME.match(name):
        raise ValueError("name may only contain letters, digits, '_', '-' and '.'")
    return name


def list_configs(folder: str | Path = DEFAULT_PARAMS_DIR) -> List[str]:
    """Names of all stored configurations (file stems), sorted."""
    return sorted(p.stem for p in Path(folder).glob("*.yaml"))


def load_config(name: str, folder: str | Path = DEFAULT_PARAMS_DIR) -> Dict[str, Any]:
    with open(Path(folder) / f"{check_name(name)}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(name: str, data: Dict[str, Any], folder: str | Path = DEFAULT_PARAMS_DIR) -> Path:
    path = Path(folder) / f"{check_name(name)}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return path


def delete_config(name: str, folder: str | Path = DEFAULT_PARAMS_DIR) -> None:
    (Path(folder) / f"{check_name(name)}.yaml").unlink(missing_ok=True)
