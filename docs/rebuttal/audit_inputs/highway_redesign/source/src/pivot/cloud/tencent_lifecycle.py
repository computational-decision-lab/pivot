"""Read-only-by-default Tencent CVM inventory and guarded lifecycle actions.

The module deliberately depends on a small client protocol.  A Tencent SDK
client can be supplied by callers, while tests and offline planning can use a
local fake without credentials or network access.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .archive import atomic_json, validate_receipt
from .identity import INSTANCE_NAME_PREFIX, OWNER_TAG_KEY, OWNER_TAG_VALUE

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
    def describe_instances(
        self, instance_ids: Sequence[str] | None = None
    ) -> Mapping[str, Any]: ...

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

    def __init__(
        self,
        client: ResourceClient,
        *,
        manifest_path: Path | None = None,
        log_path: Path | None = None,
        region: str | None = None,
    ) -> None:
        self.client = client
        self.manifest_path = manifest_path
        self.log_path = log_path
        self.region = region

    def _event(self, event: str, **fields: Any) -> None:
        if self.log_path is None:
            raise AuthorizationError("a persistent log path is required for resource actions")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
            handle.write("\n" + json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _registered(
        self, instance_id: str, *, allowed_states: frozenset[str]
    ) -> tuple[dict[str, Any], str]:
        if self.manifest_path is None:
            raise AuthorizationError("resource is not registered in an owned manifest")
        try:
            manifest = json.loads(self.manifest_path.read_text())
            record = manifest["owned_resources"][instance_id]
            task = manifest["task_id"]
            run_id = record["run_id"]
            intent = manifest["intents"][run_id]
            if (
                not task
                or manifest["region"] != self.region
                or record["created_by_task"] != task
                or not record["create_request_id"]
                or record["state"] not in allowed_states
                or not self.region
                or record["region"] != self.region
                or not run_id
                or intent["created_by_task"] != task
                or intent["region"] != self.region
                or intent["run_id"] != run_id
                or intent["instance_id"] != instance_id
                or intent["state"] != "active"
                or intent["create_request_id"] != record["create_request_id"]
                or intent["request_sha256"] != record["request_sha256"]
                or intent["config_hash"] != record["config_hash"]
            ):
                raise ValueError("creation provenance mismatch")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            state_requirement = ", ".join(sorted(allowed_states))
            raise AuthorizationError(
                "resource is not registered with valid creation provenance "
                f"in required state: {state_requirement}"
            ) from exc
        return record, task

    def _reject_protected(self, instance_id: str) -> None:
        protected = set(PROTECTED_INSTANCE_IDS)
        if self.manifest_path is not None and self.manifest_path.exists():
            try:
                extra = json.loads(self.manifest_path.read_text()).get("protected_resources", [])
                if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
                    raise ValueError("invalid protected-resource list")
                protected.update(extra)
            except (OSError, ValueError, TypeError) as exc:
                raise AuthorizationError("cannot verify resource protection manifest") from exc
        if instance_id in protected:
            if self.log_path is not None:
                self._event("PROTECTED_RESOURCE_SKIPPED", instance_id=instance_id)
            raise ProtectedResourceError(f"instance {instance_id} is protected")

    @contextmanager
    def _manifest_lock(self):
        if self.manifest_path is None:
            raise AuthorizationError("resource is not registered in an owned manifest")
        lock_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

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
        self._reject_protected(instance_id)
        owned, task = self._registered(instance_id, allowed_states=frozenset({"active"}))
        if owner != OWNER_TAG_VALUE:
            raise AuthorizationError(f"owner must be {OWNER_TAG_VALUE}")
        payload = self.client.describe_instances([instance_id])
        records = payload.get("InstanceSet") or payload.get("instances") or []
        record = next((item for item in records if item.get("InstanceId") == instance_id), None)
        if record is None:
            raise ResourceNotFoundError(instance_id)
        actual = _tag_map(record)
        if actual.get(OWNER_TAG_KEY) != owner:
            raise AuthorizationError("owner does not match the instance owner tag")
        if actual.get("run_id") != owned["run_id"] or actual.get("task_id") != task:
            raise AuthorizationError("instance tags do not match the registered run/task")
        if (
            not str(record.get("InstanceName", "")).startswith(INSTANCE_NAME_PREFIX)
            or record.get("InstanceChargeType") != "SPOTPAID"
        ):
            raise AuthorizationError("instance name or billing type violates the spot-only plan")
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
        self._reject_protected(instance_id)
        with self._manifest_lock():
            return self._mutate_locked(action, instance_id, owner=owner, tags=tags, dry_run=dry_run)

    def _mutate_locked(
        self,
        action: str,
        instance_id: str,
        *,
        owner: str,
        tags: Mapping[str, str] | None,
        dry_run: bool,
    ) -> dict[str, Any]:
        if action not in _MUTATING_ACTIONS:
            raise ValueError(f"unsupported lifecycle action: {action}")
        self._authorized_record(instance_id, owner, tags or {})
        if action in {"stop", "release"}:
            owned, _ = self._registered(instance_id, allowed_states=frozenset({"active"}))
            try:
                validate_receipt(
                    Path(owned["archive_receipt"]), instance_id=instance_id, run_id=owned["run_id"]
                )
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise AuthorizationError("archive integrity and restore evidence required") from exc
        if dry_run:
            return {
                "action": action,
                "instance_id": instance_id,
                "dry_run": True,
                "authorized": True,
            }
        self._event("RESOURCE_ACTION_INTENT", action=action, instance_id=instance_id)
        method = {
            "start": self.client.start_instances,
            "stop": self.client.stop_instances,
            "release": self.client.terminate_instances,
        }[action]
        response = method([instance_id])
        if "Error" in response or not response.get("RequestId"):
            self._event("RESOURCE_ACTION_UNCONFIRMED", action=action, instance_id=instance_id)
            raise RuntimeError("resource API did not return an accepted request identifier")
        self._event(
            "RESOURCE_ACTION_ACCEPTED",
            action=action,
            instance_id=instance_id,
            request_id=response["RequestId"],
        )
        if action == "release":
            manifest = json.loads(self.manifest_path.read_text())
            manifest["owned_resources"][instance_id].update(
                state="release_pending",
                release_request_id=response["RequestId"],
                absence_observations=[],
            )
            atomic_json(self.manifest_path, manifest)
        return {
            "action": action,
            "instance_id": instance_id,
            "dry_run": False,
            "authorized": True,
            "request": dict(_sanitize(response)),
        }

    def confirm_release(self, instance_id: str) -> dict[str, Any]:
        """Poll a known accepted release; do not equate absence with all billing cleanup.

        Call again on the same ID after a pending response. Two distinct successful
        targeted inventories with zero TotalCount are required. Errors/timeouts or
        malformed/empty payloads never count as an observation of absence.
        """
        self._reject_protected(instance_id)
        with self._manifest_lock():
            self._reject_protected(instance_id)
            owned, _ = self._registered(instance_id, allowed_states=frozenset({"release_pending"}))
            if owned["state"] != "release_pending" or not owned.get("release_request_id"):
                raise AuthorizationError("accepted release provenance is required")
            response = self.client.describe_instances([instance_id])
            if (
                response.get("Error")
                or not response.get("RequestId")
                or not isinstance(response.get("InstanceSet"), list)
                or not isinstance(response.get("TotalCount"), int)
                or isinstance(response.get("TotalCount"), bool)
            ):
                self._event("RELEASE_OBSERVATION_UNCONFIRMED", instance_id=instance_id)
                raise RuntimeError("provider did not return a valid targeted inventory")
            absent = response["InstanceSet"] == [] and response["TotalCount"] == 0
            manifest = json.loads(self.manifest_path.read_text())
            record = manifest["owned_resources"][instance_id]
            observations = record.get("absence_observations", []) if absent else []
            if absent and response["RequestId"] not in {row["request_id"] for row in observations}:
                observations.append(
                    {
                        "request_id": response["RequestId"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            record["absence_observations"] = observations
            verified = len(observations) >= 2
            if verified:
                record["state"] = "released"
                record["instance_absence_verified"] = True
                record["ancillary_cleanup_verified"] = False
            atomic_json(self.manifest_path, manifest)
            self._event(
                "INSTANCE_ABSENCE_VERIFIED" if verified else "RELEASE_STILL_PENDING",
                instance_id=instance_id,
                describe_request_id=response["RequestId"],
            )
            return {
                "instance_id": instance_id,
                "instance_absence_verified": verified,
                "ancillary_cleanup_verified": False,
            }
