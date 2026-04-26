"""Small shared utilities for project setup, paths, seeds, and JSON files."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - only used if torch is absent
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PHASE1_RESULTS_DIR = RESULTS_DIR / "phase1_environment"


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    """Create a directory if needed and return it as a `Path`."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch if PyTorch is installed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cuda") -> str:
    """Return `cuda` when requested and available, otherwise `cpu`."""
    if preferred == "cuda" and torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _to_jsonable(value: Any) -> Any:
    """Convert common project objects into JSON-friendly values."""
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def save_json(data: Any, path: os.PathLike[str] | str) -> Path:
    """Save JSON with consistent formatting."""
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2)
    return out


def load_json(path: os.PathLike[str] | str) -> Any:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def timestamp() -> str:
    """Human-readable timestamp for logs and reports."""
    return time.strftime("%Y-%m-%d %H:%M:%S")
