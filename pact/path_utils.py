"""Path helpers for Akash's PACT package.

The PACT code lives outside the shared baseline folder and keeps Akash-specific
runtime code local. The shared baseline source remains available as a fallback
for unchanged utility modules, without relying on the caller's working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path


PACT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACT_DIR.parent
SHARED_BASELINE_DIR = PROJECT_DIR / "cs830_shared_baseline"
SHARED_SRC_DIR = SHARED_BASELINE_DIR / "src"


def configure_paths() -> None:
    """Put PACT before shared-baseline source on ``sys.path``.

    Keeping ``PACT_DIR`` first ensures adapted modules such as ``ppo_mapf`` and
    ``evaluate_fragility`` resolve from PACT before falling back to shared code.
    """
    for path in (PACT_DIR, SHARED_SRC_DIR):
        path_str = str(path)
        while path_str in sys.path:
            sys.path.remove(path_str)
    sys.path.insert(0, str(SHARED_SRC_DIR))
    sys.path.insert(0, str(PACT_DIR))


def shared_path(*parts: str) -> Path:
    """Return a path inside ``cs830_shared_baseline``."""
    return SHARED_BASELINE_DIR.joinpath(*parts)
