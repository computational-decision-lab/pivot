"""Auditable infrastructure for modern self-improvement transitions.

The package is intentionally separate from the frozen paper experiments.  It
provides protocol and development tooling; confirmatory claims are emitted
only when an explicitly locked external run is available.
"""

from .protocol import TERMINAL_STATES, AgentPolicy, TransitionRecord

__all__ = ["TERMINAL_STATES", "AgentPolicy", "TransitionRecord"]
