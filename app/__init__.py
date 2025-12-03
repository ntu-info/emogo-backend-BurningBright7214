"""
EmoGo backend FastAPI application package.

The heavy lifting is done in submodules such as:
- core: configuration and shared settings
- db: database connection helpers
- api: route registrations
"""

from .core.config import settings

__all__ = ["settings"]

