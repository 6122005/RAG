"""Authentication dependency enforcing constant-time X-API-Key verification."""

import secrets
from fastapi import Header, HTTPException, status
from ..config import get_settings


async def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")) -> str:
    """Validate incoming X-API-Key header against configured API_KEY in constant time."""
    settings = get_settings()
    expected_key = settings.API_KEY

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison prevents timing-attack side channels
    if not secrets.compare_digest(x_api_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid 'X-API-Key' provided.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return x_api_key
