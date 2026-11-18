"""Identity contract for resources created by the HighwayEnv run."""

from __future__ import annotations

INSTANCE_NAME_PREFIX = "pivot-highway-"
OWNER_TAG_KEY = "owned_by"
OWNER_TAG_VALUE = "pivot-highway"

def validate_instance_name(name: str) -> bool:
    """Return whether a provider instance name belongs to this run family."""

    return isinstance(name, str) and name.startswith(INSTANCE_NAME_PREFIX)
