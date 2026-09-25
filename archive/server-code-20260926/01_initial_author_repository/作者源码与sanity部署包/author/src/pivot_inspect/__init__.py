"""Stable facade for the Inspect-controlled external-agent boundary.

Importing this module only loads contracts.  It does not install packages,
open a sealed task plane, invoke a model, or start a container.  Actual
execution remains explicit through the V15 command surface.
"""

from experiments.v15.adapters.inspect_ai import InspectControlPlane
from experiments.v15.adapters.mini_swe import MiniSWEAdapter
from experiments.v15.adapters.pi import PiAdapter
from experiments.v15.external_runtime import (
    ExecutionRecord,
    PairedExecutionRecord,
    RuntimeSettings,
    evaluate_paired_with_inspect,
    evaluate_with_inspect,
)

__all__ = [
    "ExecutionRecord",
    "InspectControlPlane",
    "MiniSWEAdapter",
    "PairedExecutionRecord",
    "PiAdapter",
    "RuntimeSettings",
    "evaluate_paired_with_inspect",
    "evaluate_with_inspect",
]
