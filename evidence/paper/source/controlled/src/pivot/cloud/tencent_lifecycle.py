"""Read-only-by-default Tencent CVM inventory and guarded lifecycle actions.

The module deliberately depends on a small client protocol.  A Tencent SDK
client can be supplied by callers, while tests and offline planning can use a
local fake without credentials or network access.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

PROTECTED_INSTANCE_IDS = frozenset({"ins-5qak5eao"})
_MUTATING_ACTIONS = {"start", "stop", "release"}
_SENSITIVE_KEY_PARTS = (
    "secret",
    "token",
    "password",
    "credential",
    "privatekey",
    "accesskey",
    "securitytoken",
)


class ResourceClient(Protocol):
    def describe_instances(self, instance_ids: Sequence[str] | None = None) -> Mapping[str, Any]: ...

    def start_instances(self, instance_ids: Sequence[str]) -> Mapping[str, Any]: ...

    def stop_instances(self, instance_ids: Sequence[str]) -> Mapping[str, Any]: ...

    def terminate_instances(self, instance_ids: Sequence[str]) -> Mapping[str, Any]: ...


class ProtectedResourceError(RuntimeError):
    """Raised whenever a protected instance is targeted for mutation."""


class AuthorizationError(RuntimeError):
    """Raised when the supplied owner or tags do not authorize a mutation."""


class ResourceNotFoundError(LookupError):
    """Raised when Tencent does not return the requested instance."""


def _tag_map(record: Mapping[str, Any]) -> dict[str, str]:
    tags = record.get("Tags") or record.get("tags") or []
    if isinstance(tags, Mapping):
        return {str(k): str(v) for k, v in tags.items()}
    result: dict[str, str] = {}
    for tag in tags:
        if isinstance(tag, Mapping) and "Key" in tag:
            result[str(tag["Key"])] = str(tag.get("Value", ""))
    return result


def _sanitize(value: Any, key: str = "") -> Any:
    lowered = key.replace("_", "").replace("-", "").lower()
    if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): _sanitize(v, str(k)) for k, v in value.items() if not _is_sensitive(str(k))}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _is_sensitive(key: str) -> bool:
    lowered = key.replace("_", "").replace("-", "").lower()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


class TencentResourceLifecycle:
    """Inventory and guarded CVM lifecycle operations.

    ``mutate`` defaults to ``dry_run=True``.  The caller must pass
    ``dry_run=False`` to perform a cloud mutation after authorization checks.
    """

    def __init__(self, client: ResourceClient) -> None:
        self.client = client

    def describe(self, instance_id: str | None = None) -> dict[str, Any]:
        ids = [instance_id] if instance_id else None
        payload = self.client.describe_instances(ids)
        records = payload.get("InstanceSet") or payload.get("instances") or []
        return {"instances": [_sanitize(record) for record in records]}

    def _authorized_record(
        self,
        instance_id: str,
        owner: str,
        tags: Mapping[str, str],
    ) -> Mapping[str, Any]:
        if instance_id in PROTECTED_INSTANCE_IDS:
            raise ProtectedResourceError(f"instance {instance_id} is protected")
        payload = self.client.describe_instances([instance_id])
        records = payload.get("InstanceSet") or payload.get("instances") or []
        record = next((item for item in records if item.get("InstanceId") == instance_id), None)
        if record is None:
            raise ResourceNotFoundError(instance_id)
        actual = _tag_map(record)
        if not owner or actual.get("owner") != owner:
            raise AuthorizationError("owner does not match the instance owner tag")
        for key, value in tags.items():
            if actual.get(key) != str(value):
                raise AuthorizationError(f"tag {key!r} does not match the instance")
        return record

    def mutate(
        self,
        action: str,
        instance_id: str,
        *,
        owner: str,
        tags: Mapping[str, str] | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        if action not in _MUTATING_ACTIONS:
            raise ValueError(f"unsupported lifecycle action: {action}")
        self._authorized_record(instance_id, owner, tags or {})
        if dry_run:
            return {"action": action, "instance_id": instance_id, "dry_run": True, "authorized": True}
        method = {
            "start": self.client.start_instances,
            "stop": self.client.stop_instances,
            "release": self.client.terminate_instances,
        }[action]
        response = method([instance_id])
        return {
            "action": action,
            "instance_id": instance_id,
            "dry_run": False,
            "authorized": True,
            "request": dict(_sanitize(response)),
        }
