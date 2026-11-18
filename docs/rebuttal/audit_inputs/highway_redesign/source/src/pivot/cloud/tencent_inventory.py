"""Normalize read-only Tencent capacity evidence for provisioning review."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Protocol

OFFICIAL_SDK_COMMIT = "f9ee6e4baf999099d3dd7dfe197f921e5c740a4a"
OFFICIAL_SDK_SOURCE = (
    "https://github.com/TencentCloud/tencentcloud-sdk-python/"
    f"tree/{OFFICIAL_SDK_COMMIT}"
)


class InventoryAdapter(Protocol):
    def describe_regions(self) -> dict[str, Any]: ...

    def describe_zones(self) -> dict[str, Any]: ...

    def describe_zone_instance_configs(self) -> dict[str, Any]: ...

    def describe_images(self) -> dict[str, Any]: ...

    def describe_account_quota(self) -> dict[str, Any]: ...

    def describe_vpcs(self) -> dict[str, Any]: ...

    def describe_subnets(self) -> dict[str, Any]: ...

    def describe_security_groups(self) -> dict[str, Any]: ...

    def inquiry_price_run_instances(self, request: Mapping[str, Any]) -> dict[str, Any]: ...


def collect_read_only_inventory(
    *,
    adapter_factory: Callable[[str], InventoryAdapter],
    bootstrap_region: str,
    regions: Sequence[str],
    target_vcpus: Sequence[int] = (8, 16, 32),
    sdk_version: str,
) -> dict[str, Any]:
    targets = sorted({int(value) for value in target_vcpus})
    if not targets or any(value <= 0 for value in targets):
        raise ValueError("target_vcpus must contain positive values")
    requested_regions = list(dict.fromkeys(str(value) for value in regions))
    if not requested_regions or any(not value for value in requested_regions):
        raise ValueError("at least one region is required")

    region_response = adapter_factory(bootstrap_region).describe_regions()
    catalog = sorted(
        (
            {
                "region": str(item.get("Region", "")),
                "name": str(item.get("RegionName", "")),
                "state": str(item.get("RegionState", "")),
            }
            for item in _records(region_response, "RegionSet")
            if item.get("Region")
        ),
        key=lambda item: item["region"],
    )
    available = {item["region"] for item in catalog if item["state"] == "AVAILABLE"}
    unavailable = [region for region in requested_regions if region not in available]
    if unavailable:
        raise ValueError(f"requested regions are not reported AVAILABLE: {', '.join(unavailable)}")

    collected = [
        _collect_region(adapter_factory(region), region=region, target_vcpus=set(targets))
        for region in requested_regions
    ]
    return {
        "schema_version": 1,
        "status": "complete",
        "read_only": True,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "sdk_version": sdk_version,
        "official_sdk_source": OFFICIAL_SDK_SOURCE,
        "official_sdk_commit": OFFICIAL_SDK_COMMIT,
        "target_vcpus": targets,
        "region_catalog": catalog,
        "region_catalog_request_id": _request_id(region_response),
        "regions": collected,
        "mutating_api_calls": 0,
        "instance_inventory_calls": 0,
    }


def _collect_region(
    adapter: InventoryAdapter, *, region: str, target_vcpus: set[int]
) -> dict[str, Any]:
    zones = adapter.describe_zones()
    configs = adapter.describe_zone_instance_configs()
    images = adapter.describe_images()
    quota = adapter.describe_account_quota()
    vpcs = adapter.describe_vpcs()
    subnets = adapter.describe_subnets()
    security_groups = adapter.describe_security_groups()
    candidates = _spot_candidates(configs, target_vcpus)
    normalized_images = _ubuntu_images(images)
    normalized_vpcs = _vpcs(vpcs)
    normalized_subnets = _subnets(subnets)
    normalized_security_groups = _security_groups(security_groups)
    price_quotes, price_quote_gaps = _price_quotes(
        adapter,
        candidates=candidates,
        images=normalized_images,
        vpcs=normalized_vpcs,
        subnets=normalized_subnets,
        security_groups=normalized_security_groups,
        target_vcpus=target_vcpus,
    )
    return {
        "region": region,
        "zones": sorted(
            (
                {
                    "zone": str(item.get("Zone", "")),
                    "name": str(item.get("ZoneName", "")),
                    "state": str(item.get("ZoneState", "")),
                }
                for item in _records(zones, "ZoneSet")
                if item.get("Zone")
            ),
            key=lambda item: item["zone"],
        ),
        "spot_candidates": candidates,
        "price_quotes": price_quotes,
        "price_quote_gaps": price_quote_gaps,
        "ubuntu_images": normalized_images,
        "spot_quota": _spot_quota(quota),
        "network": {
            "vpcs": normalized_vpcs,
            "subnets": normalized_subnets,
            "security_groups": normalized_security_groups,
        },
        "request_ids": {
            "zones": _request_id(zones),
            "spot_configs": _request_id(configs),
            "images": _request_id(images),
            "quota": _request_id(quota),
            "vpcs": _request_id(vpcs),
            "subnets": _request_id(subnets),
            "security_groups": _request_id(security_groups),
        },
    }


def _spot_candidates(response: Mapping[str, Any], targets: set[int]) -> list[dict[str, Any]]:
    candidates = []
    for item in _records(response, "InstanceTypeQuotaSet"):
        cpu = _integer(item.get("Cpu"))
        if (
            cpu not in targets
            or item.get("InstanceChargeType") != "SPOTPAID"
            or _number(item.get("Gpu"), default=0) != 0
        ):
            continue
        price = item.get("Price") if isinstance(item.get("Price"), Mapping) else {}
        candidates.append(
            {
                "zone": str(item.get("Zone", "")),
                "instance_type": str(item.get("InstanceType", "")),
                "cpu": cpu,
                "memory_gib": _number(item.get("Memory")),
                "status": str(item.get("Status", "")),
                "stock": str(item.get("StatusCategory", "")),
                "catalog_price": {
                    "charge_unit": str(price.get("ChargeUnit", "")),
                    "unit_price": _number(price.get("UnitPrice")),
                    "unit_price_discount": _number(price.get("UnitPriceDiscount")),
                },
            }
        )
    return sorted(candidates, key=lambda item: (item["cpu"], item["zone"], item["instance_type"]))


def _price_quotes(
    adapter: InventoryAdapter,
    *,
    candidates: Sequence[Mapping[str, Any]],
    images: Sequence[Mapping[str, Any]],
    vpcs: Sequence[Mapping[str, Any]],
    subnets: Sequence[Mapping[str, Any]],
    security_groups: Sequence[Mapping[str, Any]],
    target_vcpus: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    usable_images = [
        item
        for item in images
        if item.get("state") == "NORMAL"
        and item.get("type") == "PUBLIC_IMAGE"
        and item.get("architecture") == "x86_64"
        and item.get("image_id")
    ]
    usable_images.sort(key=_image_preference_key)
    known_vpcs = {str(item.get("vpc_id")) for item in vpcs if item.get("vpc_id")}
    usable_subnets = [
        item
        for item in subnets
        if item.get("subnet_id")
        and item.get("vpc_id") in known_vpcs
        and _integer(item.get("available_ip_count")) > 0
    ]
    usable_security_groups = [
        item for item in security_groups if item.get("security_group_id")
    ]
    quotes: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for cpu in sorted(target_vcpus):
        candidate_options = [
            item
            for item in candidates
            if item.get("cpu") == cpu and item.get("status") == "SELL"
        ]
        network_zones = (
            {str(item.get("zone")) for item in usable_subnets if item.get("zone")}
            if usable_security_groups
            else set()
        )
        candidate_options.sort(
            key=lambda item: (
                item.get("zone") not in network_zones,
                *_quote_candidate_key(item),
            )
        )
        if not candidate_options or not usable_images:
            missing = []
            if not candidate_options:
                missing.append("sellable_candidate")
            if not usable_images:
                missing.append("normal_public_x86_64_ubuntu_image")
            gaps.append({"cpu": cpu, "missing": missing})
            continue
        candidate = candidate_options[0]
        image = usable_images[0]
        request = {
            "InstanceChargeType": "SPOTPAID",
            "Placement": {"Zone": candidate["zone"]},
            "InstanceType": candidate["instance_type"],
            "ImageId": image["image_id"],
            "SystemDisk": {"DiskType": "CLOUD_BSSD", "DiskSize": 50},
            "InstanceCount": 1,
        }
        subnet = next(
            (
                item
                for item in usable_subnets
                if item.get("zone") == candidate.get("zone")
            ),
            None,
        )
        security_group = usable_security_groups[0] if usable_security_groups else None
        network = None
        if subnet is not None and security_group is not None:
            network = {
                "vpc_id": subnet["vpc_id"],
                "subnet_id": subnet["subnet_id"],
                "security_group_id": security_group["security_group_id"],
            }
            request["VirtualPrivateCloud"] = {
                "VpcId": subnet["vpc_id"],
                "SubnetId": subnet["subnet_id"],
            }
            request["SecurityGroupIds"] = [security_group["security_group_id"]]
        base = {
            "source": "InquiryPriceRunInstances",
            "cpu": cpu,
            "zone": candidate["zone"],
            "instance_type": candidate["instance_type"],
            "image_id": image["image_id"],
            "network": network,
            "launchable_with_observed_network": network is not None,
            "catalog_unit_price_discount": candidate["catalog_price"].get(
                "unit_price_discount"
            ),
        }
        try:
            response = adapter.inquiry_price_run_instances(request)
            price = response.get("Price")
            instance_price = price.get("InstancePrice") if isinstance(price, Mapping) else None
            if not isinstance(instance_price, Mapping):
                raise TypeError("price response omitted InstancePrice")
            quotes.append(
                {
                    **base,
                    "status": "quoted",
                    "quote": {
                        "charge_unit": str(instance_price.get("ChargeUnit", "")),
                        "unit_price": _number(instance_price.get("UnitPrice")),
                        "unit_price_discount": _number(
                            instance_price.get("UnitPriceDiscount")
                        ),
                    },
                    "system_disk": {
                        "disk_type": "CLOUD_BSSD",
                        "disk_size_gib": 50,
                    },
                    "request_id": _request_id(response),
                }
            )
        except Exception as error:  # noqa: BLE001 - preserve a redacted per-candidate gap
            gaps.append(
                {
                    **base,
                    "status": "quote_failed",
                    "error_type": type(error).__name__,
                }
            )
    return quotes, gaps


def _image_preference_key(item: Mapping[str, Any]) -> tuple[int, int, str, str]:
    name = str(item.get("name", ""))
    lowered = name.lower()
    return (
        int("ubuntu server" not in lowered),
        int("24.04" not in lowered),
        name,
        str(item.get("image_id", "")),
    )


def _quote_candidate_key(item: Mapping[str, Any]) -> tuple[float, str, str]:
    catalog = item.get("catalog_price")
    discount = catalog.get("unit_price_discount") if isinstance(catalog, Mapping) else None
    price = _number(discount, default=float("inf"))
    if price <= 0:
        price = float("inf")
    return price, str(item.get("zone", "")), str(item.get("instance_type", ""))


def _ubuntu_images(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    images = []
    for item in _records(response, "ImageSet"):
        if "ubuntu" not in str(item.get("Platform", "")).lower():
            continue
        images.append(
            {
                "image_id": str(item.get("ImageId", "")),
                "name": str(item.get("ImageName", "")),
                "platform": str(item.get("Platform", "")),
                "architecture": str(item.get("Architecture", "")),
                "state": str(item.get("ImageState", "")),
                "type": str(item.get("ImageType", "")),
                "cloud_init": bool(item.get("IsSupportCloudinit", False)),
            }
        )
    return sorted(images, key=lambda item: (item["name"], item["image_id"]))


def _spot_quota(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    overview = response.get("AccountQuotaOverview")
    account = overview.get("AccountQuota") if isinstance(overview, Mapping) else None
    records = account.get("SpotPaidQuotaSet", []) if isinstance(account, Mapping) else []
    if not isinstance(records, list):
        return []
    return sorted(
        (
            {
                "zone": str(item.get("Zone", "")),
                "used_vcpus": _integer(item.get("UsedQuota")),
                "remaining_vcpus": _integer(item.get("RemainingQuota")),
                "total_vcpus": _integer(item.get("TotalQuota")),
            }
            for item in records
            if isinstance(item, Mapping)
        ),
        key=lambda item: item["zone"],
    )


def _vpcs(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "vpc_id": str(item.get("VpcId", "")),
                "name": str(item.get("VpcName", "")),
                "default": bool(item.get("IsDefault", False)),
            }
            for item in _records(response, "VpcSet")
            if item.get("VpcId")
        ),
        key=lambda item: item["vpc_id"],
    )


def _subnets(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "subnet_id": str(item.get("SubnetId", "")),
                "name": str(item.get("SubnetName", "")),
                "vpc_id": str(item.get("VpcId", "")),
                "zone": str(item.get("Zone", "")),
                "available_ip_count": _integer(item.get("AvailableIpAddressCount")),
                "default": bool(item.get("IsDefault", False)),
            }
            for item in _records(response, "SubnetSet")
            if item.get("SubnetId")
        ),
        key=lambda item: item["subnet_id"],
    )


def _security_groups(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "security_group_id": str(item.get("SecurityGroupId", "")),
                "name": str(item.get("SecurityGroupName", "")),
            }
            for item in _records(response, "SecurityGroupSet")
            if item.get("SecurityGroupId")
        ),
        key=lambda item: item["security_group_id"],
    )


def _records(response: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    records = response.get(key, [])
    if not isinstance(records, list):
        return []
    return [item for item in records if isinstance(item, Mapping)]


def _request_id(response: Mapping[str, Any]) -> str | None:
    value = response.get("RequestId")
    return str(value) if value else None


def _integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _number(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


__all__ = ["collect_read_only_inventory"]
