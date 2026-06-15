import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_params(params_path: str = None) -> Dict[str, Any]:
    project_dir = Path(__file__).resolve().parents[1]
    params_path = Path(params_path) if params_path else project_dir / "params.yaml"
    params_path = params_path.resolve()

    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found at {params_path}")

    with params_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_param(section: str, name: str, default: Any = None) -> Any:
    params = load_params()
    return params.get(section, {}).get(name, default)


def get_section(section: str) -> Dict[str, Any]:
    params = load_params()
    return params.get(section, {})
