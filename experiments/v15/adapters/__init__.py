"""Agent-scaffold adapter contracts used by the V15 control plane.

Adapters are deliberately dry-run safe: construction and status inspection do
not import an agent runtime, invoke a model, or open a sealed task.  A future
confirmatory runner can implement the same request/result contract without
changing PIVOT's scientific layer.
"""

from .base import AdapterRequest, AdapterResult, ExternalAdapter
from .inspect_ai import InspectControlPlane
from .mini_swe import MiniSWEAdapter
from .pi import PiAdapter

__all__ = [
    "AdapterRequest",
    "AdapterResult",
    "ExternalAdapter",
    "InspectControlPlane",
    "MiniSWEAdapter",
    "PiAdapter",
]
