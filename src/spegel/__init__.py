"""Spegel – Reflect the web through AI (package entry)."""

from ._internal.debug import _get_version

__version__: str = _get_version()

from .main import Spegel as _SpegelApp  # noqa: F401

__all__ = ["__version__", "_SpegelApp"]
