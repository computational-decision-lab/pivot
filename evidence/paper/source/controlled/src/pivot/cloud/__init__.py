"""Protected cloud-resource control helpers."""

from .tencent_lifecycle import (
    AuthorizationError,
    PROTECTED_INSTANCE_IDS,
    ProtectedResourceError,
    ResourceNotFoundError,
    TencentResourceLifecycle,
)

__all__ = [
    "AuthorizationError",
    "PROTECTED_INSTANCE_IDS",
    "ProtectedResourceError",
    "ResourceNotFoundError",
    "TencentResourceLifecycle",
]
