"""Spegel – Reflect the web through AI (package entry)."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("spegel")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0-dev"

from .main import Spegel as _SpegelApp, main  # noqa: F401

__all__ = ["__version__", "_SpegelApp", "main"]
