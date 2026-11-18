"""Durable pre-creation controls for Tencent CVM provisioning.

This module uses a small injected client protocol and never constructs SDK
credentials. It persists an intent before ``RunInstances`` and treats every
ambiguous result as reconciliation-required state.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from .archive import atomic_json
from .budget import BudgetExceeded, BudgetLedger
from .identity import INSTANCE_NAME_PREFIX, OWNER_TAG_KEY, OWNER_TAG_VALUE
from .tencent_lifecycle import PROTECTED_INSTANCE_IDS, ProtectedResourceError
from .tencent_sdk import TencentSdkError

_REQUIRED_TAGS = {"owned_by", "task_id", "run_id"}
_FORBIDDEN_PERSISTED_KEYS = {"LoginSettings", "UserData"}
_SAFE_PROVIDER_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _safe_provider_identifier(value: Any) -> str | None:
    candidate = str(value) if value is not None else ""
    return candidate if _SAFE_PROVIDER_IDENTIFIER.fullmatch(candidate) else None


class ProvisionClient(Protocol):
    def run_instances(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def describe_instances(
        self,
        *,
        instance_ids: Sequence[str] | None = None,
        filters: Sequence[Mapping[str, Any]] | None = None,
    ) -> Mapping[str, Any]: ...


class IdentityConflictError(RuntimeError):
    """A run identity was reused with different immutable inputs."""


class ProvisioningRejectedError(RuntimeError):
    """RunInstances returned an explicit provider error."""


class ProvisioningUnknownError(RuntimeError):
    """The provider may have accepted a request, so retry is unsafe."""


class ReconciliationRequiredError(RuntimeError):
    """Creation needs provider evidence that public inventory cannot supply."""


class OwnershipMismatchError(RuntimeError):
    """Described resource identity or associations differ from the intent."""


class TencentProvisioner:
    """Prepare, create, and reconcile one-instance spot provisioning intents."""

    def __init__(
        self,
        client: ProvisionClient,
        *,
        budget_ledger: BudgetLedger,
        manifest_path: Path,
        log_path: Path,
        task_id: str,
        region: str,
    ) -> None:
        if not task_id or not region:
            raise ValueError("task_id and region are required")
        self.client = client
        self.budget_ledger = budget_ledger
        self.manifest_path = Path(manifest_path)
        self.log_path = Path(log_path)
        self.task_id = str(task_id)
        self.region = str(region)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def prepare(
        self,
        *,
        run_id: str,
        config_hash: str,
        request: Mapping[str, Any],
        reservation: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected = self._validate_request(run_id, config_hash, request)
        request_hash = self._request_hash(request)
        client_token = self._client_token(run_id, config_hash, request_hash)
        identity = self._identity(run_id, config_hash, request_hash, client_token)

        required_parts = {"compute", "disks", "network", "contingency"}
        if set(reservation) != required_parts:
            raise ValueError(f"reservation requires exactly {sorted(required_parts)}")
        self.budget_ledger.reserve(run_id, **dict(reservation))

        with self._locked_manifest() as manifest:
            previous = manifest["intents"].get(run_id)
            if previous is not None:
                self._require_identity(previous, identity)
                return copy.deepcopy(previous)
            record = {
                **identity,
                "state": "intent",
                "created_by_task": self.task_id,
                "region": self.region,
                "expected": expected,
                "reservation_total_cny": self.budget_ledger.read()["reservations"][run_id][
                    "total"
                ],
            }
            manifest["intents"][run_id] = record
            atomic_json(self.manifest_path, manifest)
        self._event(
            "PROVISION_INTENT_PREPARED",
            run_id=run_id,
            config_hash=config_hash,
            request_sha256=request_hash,
            client_token_sha256=self._digest(client_token),
        )
        return copy.deepcopy(record)

    def create(
        self,
        *,
        run_id: str,
        config_hash: str,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._validate_request(run_id, config_hash, request)
        request_hash = self._request_hash(request)
        client_token = self._client_token(run_id, config_hash, request_hash)
        identity = self._identity(run_id, config_hash, request_hash, client_token)

        with self._locked_manifest() as manifest:
            intent = self._intent(manifest, run_id)
            self._require_identity(intent, identity)
            if intent["state"] in {"provisional", "active"}:
                return copy.deepcopy(intent)
            if intent["state"] != "intent":
                raise ReconciliationRequiredError(
                    f"run {run_id} is {intent['state']}; reconcile before another create"
                )

            with self._validated_budget(run_id, intent):
                intent["state"] = "creating"
                atomic_json(self.manifest_path, manifest)
                api_request = copy.deepcopy(dict(request))
                api_request["ClientToken"] = client_token
                self._event(
                    "RUN_INSTANCES_INTENT",
                    run_id=run_id,
                    request_sha256=request_hash,
                    client_token_sha256=self._digest(client_token),
                )
                try:
                    response = dict(self.client.run_instances(api_request))
                except TencentSdkError as error:
                    intent["state"] = "rejected"
                    intent["create_request_id"] = error.request_id
                    intent["provider_error_code"] = error.code
                    intent["last_error_type"] = error.error_type
                    atomic_json(self.manifest_path, manifest)
                    self._event(
                        "RUN_INSTANCES_REJECTED",
                        run_id=run_id,
                        request_id=error.request_id,
                        provider_error_code=error.code,
                    )
                    raise ProvisioningRejectedError(
                        "RunInstances returned a definitive provider error"
                    ) from error
                except Exception as error:
                    provider_code = _safe_provider_identifier(getattr(error, "code", None))
                    provider_request_id = _safe_provider_identifier(
                        getattr(error, "request_id", None)
                    )
                    if provider_code and provider_request_id:
                        intent["state"] = "rejected"
                        intent["create_request_id"] = provider_request_id
                        intent["provider_error_code"] = provider_code
                        intent["last_error_type"] = type(error).__name__
                        atomic_json(self.manifest_path, manifest)
                        self._event(
                            "RUN_INSTANCES_REJECTED",
                            run_id=run_id,
                            request_id=provider_request_id,
                            provider_error_code=provider_code,
                        )
                        raise ProvisioningRejectedError(
                            "RunInstances returned a definitive provider error"
                        ) from error
                    intent["state"] = "unknown"
                    intent["last_error_type"] = type(error).__name__
                    atomic_json(self.manifest_path, manifest)
                    self._event(
                        "RUN_INSTANCES_UNKNOWN",
                        run_id=run_id,
                        error_type=type(error).__name__,
                    )
                    raise ProvisioningUnknownError(
                        "RunInstances outcome is unknown; automatic retry is disabled"
                    ) from error

                request_id = str(response.get("RequestId", ""))
                instance_ids = response.get("InstanceIdSet") or []
                if response.get("Error") or not request_id or not isinstance(instance_ids, list):
                    intent["state"] = "rejected"
                    intent["create_request_id"] = request_id or None
                    intent["provider_error_code"] = self._provider_error_code(response)
                    atomic_json(self.manifest_path, manifest)
                    self._event(
                        "RUN_INSTANCES_REJECTED",
                        run_id=run_id,
                        request_id=request_id or None,
                        provider_error_code=intent["provider_error_code"],
                    )
                    raise ProvisioningRejectedError(
                        "RunInstances did not return one accepted instance identifier"
                    )
                if len(instance_ids) != 1 or not str(instance_ids[0]).startswith("ins-"):
                    intent["state"] = "unknown"
                    intent["create_request_id"] = request_id
                    atomic_json(self.manifest_path, manifest)
                    raise ProvisioningUnknownError(
                        "RunInstances returned an ambiguous instance identifier set"
                    )

                instance_id = str(instance_ids[0])
                if instance_id in self._protected_resources(manifest):
                    intent["state"] = "unknown"
                    intent["create_request_id"] = request_id
                    atomic_json(self.manifest_path, manifest)
                    self._event(
                        "PROTECTED_RESOURCE_SKIPPED", run_id=run_id, instance_id=instance_id
                    )
                    raise ProtectedResourceError(f"instance {instance_id} is protected")

                intent.update(
                    {
                        "state": "provisional",
                        "instance_id": instance_id,
                        "create_request_id": request_id,
                    }
                )
                manifest["owned_resources"][instance_id] = self._owned_record(intent)
                atomic_json(self.manifest_path, manifest)
        self._event(
            "RUN_INSTANCES_PROVISIONAL",
            run_id=run_id,
            instance_id=instance_id,
            request_id=request_id,
        )
        return copy.deepcopy(intent)

    def reconcile(self, *, run_id: str) -> dict[str, Any]:
        with self._locked_manifest() as manifest:
            intent = self._intent(manifest, run_id)
            instance_id = intent.get("instance_id")
            filters: list[dict[str, Any]] | None = None
            instance_ids: list[str] | None = None
            if instance_id:
                instance_ids = [str(instance_id)]
            else:
                expected = intent["expected"]
                filters = [
                    {"Name": "instance-name", "Values": [expected["instance_name"]]},
                    {"Name": f"tag:{OWNER_TAG_KEY}", "Values": [OWNER_TAG_VALUE]},
                    {"Name": "tag:task_id", "Values": [self.task_id]},
                    {"Name": "tag:run_id", "Values": [run_id]},
                ]
            try:
                response = self.client.describe_instances(
                    instance_ids=instance_ids,
                    filters=filters,
                )
            except Exception as error:
                intent["state"] = "unknown"
                intent["last_error_type"] = type(error).__name__
                atomic_json(self.manifest_path, manifest)
                raise ProvisioningUnknownError("DescribeInstances outcome is unknown") from error

            if response.get("Error") or not response.get("RequestId"):
                intent["state"] = "unknown"
                atomic_json(self.manifest_path, manifest)
                raise ProvisioningUnknownError(
                    "DescribeInstances did not return an accepted inventory response"
                )
            records = response.get("InstanceSet") or response.get("instances") or []
            if not isinstance(records, list):
                intent["state"] = "unknown"
                atomic_json(self.manifest_path, manifest)
                raise ProvisioningUnknownError("DescribeInstances returned an invalid inventory")
            if not records:
                intent["state"] = "unknown"
                atomic_json(self.manifest_path, manifest)
                raise ProvisioningUnknownError(
                    "empty DescribeInstances inventory does not prove absence"
                )
            if len(records) != 1:
                intent["state"] = "unknown"
                atomic_json(self.manifest_path, manifest)
                raise OwnershipMismatchError("inventory did not identify exactly one request resource")

            record = records[0]
            described_id = str(record.get("InstanceId", ""))
            if described_id in self._protected_resources(manifest):
                intent["state"] = "unknown"
                atomic_json(self.manifest_path, manifest)
                self._event(
                    "PROTECTED_RESOURCE_SKIPPED", run_id=run_id, instance_id=described_id
                )
                raise ProtectedResourceError(f"instance {described_id} is protected")
            if instance_id and described_id != instance_id:
                intent["state"] = "unknown"
                atomic_json(self.manifest_path, manifest)
                raise OwnershipMismatchError("described instance does not match provisional id")

            try:
                self._validate_described(record, intent["expected"])
            except OwnershipMismatchError:
                intent["state"] = "unknown"
                atomic_json(self.manifest_path, manifest)
                raise
            if not instance_id:
                intent["state"] = "unknown"
                atomic_json(self.manifest_path, manifest)
                raise ProvisioningUnknownError(
                    "DescribeInstances does not expose ClientToken proof for this intent"
                )
            if record.get("InstanceState") != "RUNNING":
                intent["state"] = "provisional"
                manifest["owned_resources"][described_id] = self._owned_record(intent)
                atomic_json(self.manifest_path, manifest)
                return copy.deepcopy(intent)

            intent["state"] = "active"
            manifest["owned_resources"][described_id] = self._owned_record(intent)
            atomic_json(self.manifest_path, manifest)
        self._event(
            "PROVISION_ACTIVE",
            run_id=run_id,
            instance_id=described_id,
            describe_request_id=response.get("RequestId"),
        )
        return copy.deepcopy(intent)

    @contextmanager
    def _locked_manifest(self):
        lock_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.manifest_path.exists():
                manifest = json.loads(self.manifest_path.read_text())
            else:
                manifest = {
                    "version": 1,
                    "task_id": self.task_id,
                    "region": self.region,
                    "protected_resources": sorted(PROTECTED_INSTANCE_IDS),
                    "intents": {},
                    "owned_resources": {},
                }
            if manifest.get("task_id") != self.task_id or manifest.get("region") != self.region:
                raise IdentityConflictError("manifest task or region does not match controller")
            protected = self._protected_resources(manifest)
            if not PROTECTED_INSTANCE_IDS <= protected:
                raise IdentityConflictError("manifest omitted required protected resources")
            yield manifest

    @contextmanager
    def _validated_budget(self, run_id: str, intent: Mapping[str, Any]):
        # Keep the ledger lock through RunInstances so settlement cannot race the gate.
        with self.budget_ledger._locked() as ledger:
            reservations = ledger.get("reservations")
            if not isinstance(reservations, Mapping):
                raise BudgetExceeded("BUDGET_LIMIT_REACHED: invalid reservation ledger")
            reservation = reservations.get(run_id)
            if not isinstance(reservation, Mapping) or reservation.get("status") != "reserved":
                raise BudgetExceeded("BUDGET_LIMIT_REACHED: run reservation is not active")
            if self._budget_amount(reservation.get("total")) != self._budget_amount(
                intent.get("reservation_total_cny")
            ):
                raise BudgetExceeded("BUDGET_LIMIT_REACHED: reservation differs from intent")

            reserved_total = Decimal(0)
            for record in reservations.values():
                if not isinstance(record, Mapping) or record.get("status") not in {
                    "reserved",
                    "settled",
                }:
                    raise BudgetExceeded("BUDGET_LIMIT_REACHED: invalid reservation record")
                if record["status"] == "reserved":
                    reserved_total += self._budget_amount(record.get("total"))
            committed = self._budget_amount(ledger.get("actual_cost_cny")) + reserved_total
            if committed > BudgetLedger.LIMIT:
                raise BudgetExceeded(
                    "BUDGET_LIMIT_REACHED: actual cost plus reservations exceeds CNY 500"
                )
            yield

    def _validate_request(
        self, run_id: str, config_hash: str, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not run_id or not config_hash or len(config_hash) != 64:
            raise ValueError("run_id and a 64-character config_hash are required")
        if request.get("ClientToken") is not None:
            raise ValueError("ClientToken is controller-generated")
        if request.get("InstanceChargeType") != "SPOTPAID":
            raise ValueError("only SPOTPAID instances are allowed")
        name = str(request.get("InstanceName", ""))
        if not name.startswith(INSTANCE_NAME_PREFIX):
            raise ValueError(f"instance name must start with {INSTANCE_NAME_PREFIX}")
        if request.get("InstanceCount") != 1:
            raise ValueError("provisioning contract creates exactly one instance")
        placement = self._mapping(request, "Placement")
        vpc = self._mapping(request, "VirtualPrivateCloud")
        system_disk = self._mapping(request, "SystemDisk")
        if not all(
            str(value)
            for value in (
                placement.get("Zone"),
                request.get("InstanceType"),
                request.get("ImageId"),
                vpc.get("VpcId"),
                vpc.get("SubnetId"),
                system_disk.get("DiskType"),
                system_disk.get("DiskSize"),
            )
        ):
            raise ValueError("request must pin zone, type, image, VPC, subnet, and system disk")
        security_groups = request.get("SecurityGroupIds")
        if not isinstance(security_groups, list) or not security_groups:
            raise ValueError("request must pin at least one SecurityGroupId")
        tags = self._request_tags(request)
        expected_tags = {OWNER_TAG_KEY: OWNER_TAG_VALUE, "task_id": self.task_id, "run_id": run_id}
        if tags != expected_tags:
            raise ValueError("instance tags must exactly identify owner, task, and run")
        data_disks = request.get("DataDisks") or []
        if not isinstance(data_disks, list):
            raise TypeError("DataDisks must be a list")
        return {
            "instance_name": name,
            "instance_charge_type": "SPOTPAID",
            "zone": str(placement["Zone"]),
            "instance_type": str(request["InstanceType"]),
            "image_id": str(request["ImageId"]),
            "system_disk": self._disk_identity(system_disk),
            "data_disks": sorted(self._disk_identity(self._ensure_mapping(item)) for item in data_disks),
            "vpc_id": str(vpc["VpcId"]),
            "subnet_id": str(vpc["SubnetId"]),
            "security_group_ids": sorted(str(value) for value in security_groups),
            "tags": expected_tags,
        }

    def _validate_described(
        self, record: Mapping[str, Any], expected: Mapping[str, Any]
    ) -> None:
        placement = self._mapping(record, "Placement")
        vpc = self._mapping(record, "VirtualPrivateCloud")
        system_disk = self._mapping(record, "SystemDisk")
        data_disks = record.get("DataDisks") or []
        actual = {
            "instance_name": str(record.get("InstanceName", "")),
            "instance_charge_type": str(record.get("InstanceChargeType", "")),
            "zone": str(placement.get("Zone", "")),
            "instance_type": str(record.get("InstanceType", "")),
            "image_id": str(record.get("ImageId", "")),
            "system_disk": self._disk_identity(system_disk),
            "data_disks": sorted(
                self._disk_identity(self._ensure_mapping(item)) for item in data_disks
            ),
            "vpc_id": str(vpc.get("VpcId", "")),
            "subnet_id": str(vpc.get("SubnetId", "")),
            "security_group_ids": sorted(
                str(value) for value in (record.get("SecurityGroupIds") or [])
            ),
            "tags": self._described_tags(record),
        }
        if actual != dict(expected):
            raise OwnershipMismatchError("described tags or resource associations differ from intent")

    def _owned_record(self, intent: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "state": intent["state"],
            "created_by_task": self.task_id,
            "run_id": intent["run_id"],
            "region": self.region,
            "create_request_id": intent.get("create_request_id"),
            "client_token_sha256": self._digest(str(intent["client_token"])),
            "request_sha256": intent["request_sha256"],
            "config_hash": intent["config_hash"],
            "expected": copy.deepcopy(intent["expected"]),
        }

    def _identity(
        self, run_id: str, config_hash: str, request_hash: str, client_token: str
    ) -> dict[str, str]:
        return {
            "run_id": run_id,
            "config_hash": config_hash,
            "request_sha256": request_hash,
            "client_token": client_token,
        }

    @staticmethod
    def _require_identity(record: Mapping[str, Any], identity: Mapping[str, str]) -> None:
        if any(record.get(key) != value for key, value in identity.items()):
            raise IdentityConflictError("run/config/request identity cannot be reused")

    @staticmethod
    def _intent(manifest: Mapping[str, Any], run_id: str) -> dict[str, Any]:
        try:
            return manifest["intents"][run_id]
        except (KeyError, TypeError) as error:
            raise IdentityConflictError("run has no prepared provisioning intent") from error

    def _client_token(self, run_id: str, config_hash: str, request_hash: str) -> str:
        payload = f"{self.task_id}|{self.region}|{run_id}|{config_hash}|{request_hash}"
        return f"pivot-{self._digest(payload)[:48]}"

    @classmethod
    def _request_hash(cls, request: Mapping[str, Any]) -> str:
        payload = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return cls._digest(payload)

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    @staticmethod
    def _mapping(record: Mapping[str, Any], key: str) -> Mapping[str, Any]:
        return TencentProvisioner._ensure_mapping(record.get(key))

    @staticmethod
    def _ensure_mapping(value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError("expected a mapping in provisioning request or inventory")
        return value

    @staticmethod
    def _budget_amount(value: Any) -> Decimal:
        try:
            amount = Decimal(str(value))
        except (InvalidOperation, ValueError) as error:
            raise BudgetExceeded("BUDGET_LIMIT_REACHED: invalid budget amount") from error
        if not amount.is_finite() or amount < 0:
            raise BudgetExceeded("BUDGET_LIMIT_REACHED: invalid budget amount")
        return amount

    @staticmethod
    def _protected_resources(manifest: Mapping[str, Any]) -> set[str]:
        values = manifest.get("protected_resources")
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise IdentityConflictError("manifest has invalid protected resources")
        return set(values)

    @staticmethod
    def _disk_identity(disk: Mapping[str, Any]) -> list[Any]:
        return [str(disk.get("DiskType", "")), int(disk.get("DiskSize", 0))]

    @staticmethod
    def _request_tags(request: Mapping[str, Any]) -> dict[str, str]:
        specifications = request.get("TagSpecification")
        if not isinstance(specifications, list):
            raise TypeError("TagSpecification is required")
        instance_specs = [
            item
            for item in specifications
            if isinstance(item, Mapping) and item.get("ResourceType") == "instance"
        ]
        if len(instance_specs) != 1:
            raise ValueError("exactly one instance TagSpecification is required")
        tags = instance_specs[0].get("Tags")
        if not isinstance(tags, list):
            raise TypeError("instance Tags must be a list")
        result = {
            str(item.get("Key")): str(item.get("Value", ""))
            for item in tags
            if isinstance(item, Mapping)
        }
        if set(result) != _REQUIRED_TAGS or len(result) != len(tags):
            raise ValueError("instance tags must be unique required ownership tags")
        return result

    @staticmethod
    def _described_tags(record: Mapping[str, Any]) -> dict[str, str]:
        tags = record.get("Tags") or []
        if isinstance(tags, Mapping):
            result = {str(key): str(value) for key, value in tags.items()}
        else:
            result = {
                str(item.get("Key")): str(item.get("Value", ""))
                for item in tags
                if isinstance(item, Mapping)
            }
        return {key: result.get(key, "") for key in sorted(_REQUIRED_TAGS)}

    @staticmethod
    def _provider_error_code(response: Mapping[str, Any]) -> str | None:
        error = response.get("Error")
        return str(error.get("Code")) if isinstance(error, Mapping) and error.get("Code") else None

    def _event(self, event: str, **fields: Any) -> None:
        if any(key in fields for key in _FORBIDDEN_PERSISTED_KEYS):
            raise ValueError("sensitive provisioning request fields cannot be logged")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "task_id": self.task_id,
            "region": self.region,
            **fields,
        }
        with self.log_path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


__all__ = [
    "IdentityConflictError",
    "OwnershipMismatchError",
    "ProvisioningRejectedError",
    "ProvisioningUnknownError",
    "ReconciliationRequiredError",
    "TencentProvisioner",
]
