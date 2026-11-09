"""API-key security utilities for the MIT-WPU RAG Engine."""

from __future__ import annotations

import secrets

from fastapi import Depends, Security
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError

# ---------------------------------------------------------------------------
# Header extractor
# ---------------------------------------------------------------------------

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,  # We raise our own AuthenticationError for uniform error shape
    description="Bearer API key — set via the X-API-Key request header.",
)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def verify_api_key(
    api_key: str | None = Security(api_key_header),
    settings: Settings = Depends(get_settings),
) -> str:
    """FastAPI dependency that validates the ``X-API-Key`` header.

    Uses :func:`secrets.compare_digest` for a constant-time comparison to
    protect against timing-based side-channel attacks.

    Args:
        api_key: Value extracted from the ``X-API-Key`` request header.
                 ``None`` when the header is absent.
        settings: Injected application settings.

    Returns:
        The validated API key string.

    Raises:
        AuthenticationError: If the header is missing or the key does not
            match the configured ``RAG_API_KEY``.
    """
    if not api_key:
        raise AuthenticationError(
            message="Missing API key. Supply a valid key in the X-API-Key header.",
            status_code=401,
        )

    # Both operands must be the same type for compare_digest.
    expected = settings.RAG_API_KEY
    if not secrets.compare_digest(api_key.encode(), expected.encode()):
        raise AuthenticationError(
            message="Invalid API key.",
            status_code=401,
        )

    return api_key


async def get_current_institution(
    settings: Settings = Depends(get_settings),
) -> str:
    """FastAPI dependency that returns the configured institution identifier.

    Args:
        settings: Injected application settings.

    Returns:
        The ``INSTITUTION_ID`` string from settings (e.g. ``"MIT-WPU"``).
    """
    return settings.INSTITUTION_ID
