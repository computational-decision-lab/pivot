"""Tencent CVM/VPC SDK adapter with guarded mutations and redacted failures."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from .identity import INSTANCE_NAME_PREFIX, OWNER_TAG_KEY, OWNER_TAG_VALUE
from .tencent_lifecycle import PROTECTED_INSTANCE_IDS, ProtectedResourceError

_CREDENTIAL_KEYS = frozenset(
    {"TENCENTCLOUD_SECRET_ID", "TENCENTCLOUD_SECRET_KEY", "TENCENTCLOUD_TOKEN"}
)
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_REQUIRED_TAGS = frozenset({"owned_by", "task_id", "run_id"})


class CredentialFileError(RuntimeError):
    """Credential material is unavailable without exposing its contents."""


class TencentSdkError(RuntimeError):
    """A redacted Tencent SDK failure safe for persistence and console output."""

    def __init__(
        self,
        operation: str,
        error_type: str,
        *,
        code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.operation = operation
        self.error_type = error_type
        self.code = code
        self.request_id = request_id
        fields = [f"type={error_type}"]
        if code:
            fields.append(f"code={code}")
        if request_id:
            fields.append(f"request_id={request_id}")
        super().__init__(f"Tencent {operation} failed ({', '.join(fields)})")


@dataclass(frozen=True, repr=False)
class TencentCredentials:
    secret_id: str = field(repr=False)
    secret_key: str = field(repr=False)
    token: str | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return "TencentCredentials([REDACTED])"


def inspect_credential_file(path: Path) -> dict[str, Any]:
    """Inspect ownership and mode without reading credential contents."""

    path = Path(path)
    try:
        if path.is_symlink():
            raise CredentialFileError("credential file must not be a symlink")
        status = path.stat()
    except OSError:
        raise CredentialFileError("credential file is unavailable") from None
    mode = stat.S_IMODE(status.st_mode)
    return {
        "path": str(path),
        "mode": f"{mode:04o}",
        "owner_matches_process": status.st_uid == os.getuid(),
        "regular_file": stat.S_ISREG(status.st_mode),
        "private": mode & 0o077 == 0 and bool(mode & 0o400) and not bool(mode & 0o111),
    }


def load_tencent_credentials(path: Path) -> TencentCredentials:
    """Parse only exact Tencent variables without invoking a shell."""

    inspection = inspect_credential_file(path)
    if not inspection["regular_file"]:
        raise CredentialFileError("credential path must be a regular file")
    if not inspection["owner_matches_process"]:
        raise CredentialFileError("credential file must be owned by the current process user")
    if not inspection["private"]:
        raise CredentialFileError(
            f"credential file must be mode 0600 or stricter; found {inspection['mode']}"
        )

    parsed: dict[str, str] = {}
    try:
        lines = Path(path).read_text().splitlines()
    except (OSError, UnicodeError):
        raise CredentialFileError("credential file could not be read") from None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or key not in _CREDENTIAL_KEYS:
            continue
        if key in parsed:
            raise CredentialFileError(f"duplicate credential variable: {key}")
        parsed[key] = _literal_env_value(raw_value, key)

    missing = sorted(
        {"TENCENTCLOUD_SECRET_ID", "TENCENTCLOUD_SECRET_KEY"}.difference(parsed)
    )
    if missing:
        raise CredentialFileError(f"missing exact credential variables: {', '.join(missing)}")
    return TencentCredentials(
        secret_id=parsed["TENCENTCLOUD_SECRET_ID"],
        secret_key=parsed["TENCENTCLOUD_SECRET_KEY"],
        token=parsed.get("TENCENTCLOUD_TOKEN"),
    )


def _literal_env_value(raw_value: str, key: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] in {"'", '"'}:
        if value[-1] != value[0]:
            raise CredentialFileError(f"invalid quoted value for {key}")
        value = value[1:-1]
    if not value or "\x00" in value or "\n" in value or "\r" in value:
        raise CredentialFileError(f"empty or invalid value for {key}")
    return value


class TencentCvmSdkAdapter:
    """Adapter satisfying provision and lifecycle client protocols."""

    def __init__(
        self,
        *,
        region: str,
        cvm_client: Any,
        vpc_client: Any,
        cvm_models: ModuleType,
        vpc_models: ModuleType,
        timeout_seconds: int = 15,
    ) -> None:
        if not region:
            raise ValueError("region is required")
        if not 1 <= int(timeout_seconds) <= 120:
            raise ValueError("timeout_seconds must be between 1 and 120")
        self.region = region
        self.timeout_seconds = int(timeout_seconds)
        self._cvm_client = cvm_client
        self._vpc_client = vpc_client
        self._cvm_models = cvm_models
        self._vpc_models = vpc_models

    @classmethod
    def from_credentials(
        cls,
        *,
        region: str,
        secret_id: str,
        secret_key: str,
        token: str | None = None,
        timeout_seconds: int = 15,
    ) -> TencentCvmSdkAdapter:
        if not secret_id or not secret_key:
            raise CredentialFileError("Tencent credentials are unavailable")
        try:
            from tencentcloud.common import credential
            from tencentcloud.common.profile.client_profile import ClientProfile
            from tencentcloud.common.profile.http_profile import HttpProfile
            from tencentcloud.cvm.v20170312 import cvm_client
            from tencentcloud.cvm.v20170312 import models as cvm_models
            from tencentcloud.vpc.v20170312 import models as vpc_models
            from tencentcloud.vpc.v20170312 import vpc_client
        except ImportError:
            raise RuntimeError("pinned Tencent CVM/VPC SDK packages are unavailable") from None
        http_profile = HttpProfile(reqTimeout=int(timeout_seconds), keepAlive=True)
        client_profile = ClientProfile(httpProfile=http_profile, disable_region_breaker=True)
        sdk_credential = credential.Credential(secret_id, secret_key, token)
        return cls(
            region=region,
            cvm_client=cvm_client.CvmClient(sdk_credential, region, client_profile),
            vpc_client=vpc_client.VpcClient(sdk_credential, region, client_profile),
            cvm_models=cvm_models,
            vpc_models=vpc_models,
            timeout_seconds=timeout_seconds,
        )

    def run_instances(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_run_request(request)
        response = self._invoke(
            self._cvm_client,
            self._cvm_models,
            "RunInstances",
            "RunInstancesRequest",
            request,
        )
        protected = PROTECTED_INSTANCE_IDS.intersection(
            str(value) for value in response.get("InstanceIdSet", [])
        )
        if protected:
            raise ProtectedResourceError("provider returned a protected instance identifier")
        return response

    def describe_instances(
        self,
        instance_ids: Sequence[str] | None = None,
        filters: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if instance_ids and filters:
            raise ValueError("DescribeInstances accepts instance_ids or filters, not both")
        ids = [str(value) for value in instance_ids] if instance_ids else []
        self._reject_protected(ids)
        normalized_filters = [dict(value) for value in filters] if filters else []
        filtered_ids = [
            str(value)
            for item in normalized_filters
            for value in item.get("Values", [])
            if item.get("Name") == "instance-id"
        ]
        self._reject_protected(filtered_ids)
        payload: dict[str, Any] = {}
        if ids:
            payload["InstanceIds"] = ids
        elif normalized_filters:
            payload["Filters"] = normalized_filters
        response = self._invoke(
            self._cvm_client,
            self._cvm_models,
            "DescribeInstances",
            "DescribeInstancesRequest",
            payload,
        )
        records = response.get("InstanceSet")
        if isinstance(records, list):
            response["InstanceSet"] = [
                record
                for record in records
                if not isinstance(record, Mapping)
                or record.get("InstanceId") not in PROTECTED_INSTANCE_IDS
            ]
        return response

    def start_instances(self, instance_ids: Sequence[str]) -> dict[str, Any]:
        return self._instance_mutation("StartInstances", "StartInstancesRequest", instance_ids)

    def stop_instances(self, instance_ids: Sequence[str]) -> dict[str, Any]:
        return self._instance_mutation("StopInstances", "StopInstancesRequest", instance_ids)

    def terminate_instances(self, instance_ids: Sequence[str]) -> dict[str, Any]:
        return self._instance_mutation(
            "TerminateInstances", "TerminateInstancesRequest", instance_ids
        )

    def describe_regions(self) -> dict[str, Any]:
        return self._cvm_read("DescribeRegions", "DescribeRegionsRequest")

    def describe_zones(self) -> dict[str, Any]:
        return self._cvm_read("DescribeZones", "DescribeZonesRequest")

    def describe_zone_instance_configs(self) -> dict[str, Any]:
        return self._cvm_read(
            "DescribeZoneInstanceConfigInfos",
            "DescribeZoneInstanceConfigInfosRequest",
            {"Filters": [{"Name": "instance-charge-type", "Values": ["SPOTPAID"]}]},
        )

    def describe_images(self) -> dict[str, Any]:
        return self._cvm_read(
            "DescribeImages",
            "DescribeImagesRequest",
            {
                "Filters": [
                    {"Name": "image-type", "Values": ["PUBLIC_IMAGE"]},
                    {"Name": "platform", "Values": ["Ubuntu"]},
                ],
                "Limit": 100,
            },
        )

    def describe_account_quota(self) -> dict[str, Any]:
        return self._cvm_read(
            "DescribeAccountQuota",
            "DescribeAccountQuotaRequest",
            {"Filters": [{"Name": "quota-type", "Values": ["SpotPaidQuotaSet"]}]},
        )

    def inquiry_price_run_instances(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_price_request(request)
        return self._invoke(
            self._cvm_client,
            self._cvm_models,
            "InquiryPriceRunInstances",
            "InquiryPriceRunInstancesRequest",
            request,
        )

    def describe_vpcs(self) -> dict[str, Any]:
        return self._vpc_read("DescribeVpcs", "DescribeVpcsRequest", {"Limit": "100"})

    def describe_subnets(self) -> dict[str, Any]:
        return self._vpc_read("DescribeSubnets", "DescribeSubnetsRequest", {"Limit": "100"})

    def describe_security_groups(self) -> dict[str, Any]:
        return self._vpc_read(
            "DescribeSecurityGroups", "DescribeSecurityGroupsRequest", {"Limit": "100"}
        )

    def _instance_mutation(
        self, operation: str, request_model: str, instance_ids: Sequence[str]
    ) -> dict[str, Any]:
        ids = [str(value) for value in instance_ids]
        if not ids:
            raise ValueError("at least one instance ID is required")
        self._reject_protected(ids)
        return self._invoke(
            self._cvm_client,
            self._cvm_models,
            operation,
            request_model,
            {"InstanceIds": ids},
        )

    def _cvm_read(
        self, operation: str, request_model: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._invoke(
            self._cvm_client, self._cvm_models, operation, request_model, payload or {}
        )

    def _vpc_read(
        self, operation: str, request_model: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        return self._invoke(
            self._vpc_client, self._vpc_models, operation, request_model, payload
        )

    @staticmethod
    def _invoke(
        client: Any,
        models: ModuleType,
        operation: str,
        request_model: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            request = getattr(models, request_model)()
            request.from_json_string(json.dumps(dict(payload), allow_nan=False))
            response = getattr(client, operation)(request)
            decoded = json.loads(response.to_json_string())
            if not isinstance(decoded, dict):
                raise TypeError("SDK response is not an object")
            return decoded
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:  # noqa: BLE001 - redact every SDK/serialization failure
            raise _redacted_sdk_error(operation, error) from None

    @staticmethod
    def _reject_protected(instance_ids: Sequence[str]) -> None:
        if PROTECTED_INSTANCE_IDS.intersection(instance_ids):
            raise ProtectedResourceError("protected instance cannot be targeted")

    @staticmethod
    def _validate_run_request(request: Mapping[str, Any]) -> None:
        token = request.get("ClientToken")
        if not isinstance(token, str) or not token or len(token) > 64:
            raise ValueError("ClientToken must be a nonempty string of at most 64 ASCII bytes")
        try:
            token.encode("ascii")
        except UnicodeEncodeError:
            raise ValueError("ClientToken must contain only ASCII") from None
        if not str(request.get("InstanceName", "")).startswith(INSTANCE_NAME_PREFIX):
            raise ValueError(f"InstanceName must start with {INSTANCE_NAME_PREFIX}")
        if request.get("InstanceChargeType") != "SPOTPAID":
            raise ValueError("only SPOTPAID creation is permitted")
        if request.get("InstanceCount") != 1:
            raise ValueError("exactly one instance must be requested")
        specifications = request.get("TagSpecification")
        if not isinstance(specifications, list):
            raise TypeError("exact instance ownership tags are required")
        instance_specs = [
            item
            for item in specifications
            if isinstance(item, Mapping) and item.get("ResourceType") == "instance"
        ]
        if len(instance_specs) != 1 or not isinstance(instance_specs[0].get("Tags"), list):
            raise ValueError("exact instance ownership tags are required")
        tags = instance_specs[0]["Tags"]
        tag_map = {
            str(item.get("Key")): str(item.get("Value", ""))
            for item in tags
            if isinstance(item, Mapping)
        }
        if (
            len(tags) != len(_REQUIRED_TAGS)
            or set(tag_map) != _REQUIRED_TAGS
            or tag_map.get(OWNER_TAG_KEY) != OWNER_TAG_VALUE
            or not tag_map.get("task_id")
            or not tag_map.get("run_id")
        ):
            raise ValueError("exact owned_by, task_id, and run_id tags are required")

    @staticmethod
    def _validate_price_request(request: Mapping[str, Any]) -> None:
        if request.get("InstanceChargeType") != "SPOTPAID":
            raise ValueError("price inquiry is restricted to SPOTPAID")
        if request.get("InstanceCount") != 1:
            raise ValueError("price inquiry must describe exactly one instance")
        for forbidden in ("ClientToken", "InstanceName", "LoginSettings", "TagSpecification", "UserData"):
            if request.get(forbidden) is not None:
                raise ValueError(f"price inquiry must not contain {forbidden}")
        placement = request.get("Placement")
        network = request.get("VirtualPrivateCloud")
        system_disk = request.get("SystemDisk")
        if not isinstance(placement, Mapping) or not placement.get("Zone"):
            raise ValueError("price inquiry requires an exact zone")
        if not request.get("InstanceType") or not request.get("ImageId"):
            raise ValueError("price inquiry requires an instance type and image")
        if not isinstance(system_disk, Mapping) or not system_disk.get("DiskType"):
            raise ValueError("price inquiry requires a system disk")
        if network is not None and (
            not isinstance(network, Mapping)
            or not network.get("VpcId")
            or not network.get("SubnetId")
        ):
            raise ValueError("specified price inquiry network requires an exact VPC and subnet")
        security_groups = request.get("SecurityGroupIds")
        if security_groups is not None and (
            not isinstance(security_groups, list) or not security_groups
        ):
            raise ValueError("specified price inquiry security groups must be a nonempty list")


def _safe_exception_field(value: Any) -> str | None:
    rendered = str(value) if value is not None else ""
    return rendered if _SAFE_IDENTIFIER.fullmatch(rendered) else None


def _redacted_sdk_error(operation: str, error: Exception) -> TencentSdkError:
    return TencentSdkError(
        operation,
        type(error).__name__,
        code=_safe_exception_field(getattr(error, "code", None)),
        request_id=_safe_exception_field(
            getattr(error, "requestId", None) or getattr(error, "request_id", None)
        ),
    )


__all__ = [
    "CredentialFileError",
    "TencentCredentials",
    "TencentCvmSdkAdapter",
    "TencentSdkError",
    "inspect_credential_file",
    "load_tencent_credentials",
]
