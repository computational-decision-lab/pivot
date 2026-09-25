"""Protected cloud-resource control helpers."""

from .tencent_lifecycle import (
    PROTECTED_INSTANCE_IDS,
    AuthorizationError,
    ProtectedResourceError,
    ResourceNotFoundError,
    TencentResourceLifecycle,
)

__all__ = [
    "PROTECTED_INSTANCE_IDS",
    "AuthorizationError",
    "ProtectedResourceError",
    "ResourceNotFoundError",
    "TencentResourceLifecycle",
]
