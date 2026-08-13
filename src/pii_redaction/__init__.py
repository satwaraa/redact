"""Public API surface for pii_redaction."""

from __future__ import annotations

from pii_redaction.models import (
    DocumentError,
    LeakDetectedError,
    ModelUnavailableError,
    PIIEntity,
    PIIType,
    RedactionError,
    RedactionResult,
    RedactorConfig,
)
from pii_redaction.redactor import Redactor

__version__ = "0.1.0"

__all__ = [
    "DocumentError",
    "LeakDetectedError",
    "ModelUnavailableError",
    "PIIEntity",
    "PIIType",
    "RedactionError",
    "RedactionResult",
    "Redactor",
    "RedactorConfig",
    "__version__",
]
