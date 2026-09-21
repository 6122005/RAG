"""API package: authentication, rate limiting, and routes."""

from .auth import verify_api_key
from .routes import router

__all__ = ["verify_api_key", "router"]
