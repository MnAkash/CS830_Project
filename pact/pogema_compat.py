"""Compatibility helpers for POGEMA and Gymnasium wrapper versions."""

from __future__ import annotations


def apply_pogema_compat_patch() -> None:
    """Patch Gymnasium wrappers to forward attributes expected by POGEMA.

    POGEMA 1.4 metric wrappers access attributes such as ``grid_config`` and
    ``get_num_agents`` through nested Gymnasium wrappers. Newer Gymnasium
    versions do not forward arbitrary attributes, so environment construction
    can fail before the project code gets a usable env. This local patch keeps
    the behavior POGEMA expects without changing external packages.
    """
    try:
        from gymnasium.core import Wrapper
    except Exception:
        return

    if not hasattr(Wrapper, "grid_config"):
        Wrapper.grid_config = property(lambda self: self.env.grid_config)
    if not hasattr(Wrapper, "grid"):
        Wrapper.grid = property(lambda self: self.env.grid)
    if not hasattr(Wrapper, "was_on_goal"):
        Wrapper.was_on_goal = property(lambda self: self.env.was_on_goal)
    if not hasattr(Wrapper, "get_num_agents"):
        Wrapper.get_num_agents = lambda self: self.env.get_num_agents()