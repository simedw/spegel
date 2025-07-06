from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
from importlib.metadata import PackageNotFoundError, metadata, version
import os
import platform
import sys

__PACKAGE_NAME__ = "spegel"


@dataclass
class _Package:
    """Dataclass to store package information."""

    name: str = __PACKAGE_NAME__
    """Package name."""
    version: str = "0.0.0-dev"
    """Package version."""
    description: str = "No description available."
    """Package description."""

    def __str__(self) -> str:
        """String representation of the package information."""
        return f"{self.name} v{self.version}: {self.description}"


@dataclass
class _Variable:
    """Dataclass describing an environment variable."""

    name: str
    """Variable name."""
    value: str
    """Variable value."""


@dataclass
class _Environment:
    """Dataclass to store environment information."""

    interpreter_name: str
    """Python interpreter name."""
    interpreter_version: str
    """Python interpreter version."""
    interpreter_path: str
    """Path to Python executable."""
    platform: str
    """Operating System."""
    packages: list[_Package]
    """Installed packages."""
    variables: list[_Variable]
    """Environment variables."""

    def __str__(self) -> str:
        """String representation of the environment information."""
        return (
            f"Python {self.interpreter_name} {self.interpreter_version} "
            f"({self.interpreter_path}) on {self.platform}\n"
            f"Packages:\n{', '.join(str(pkg) for pkg in self.packages)}\n"
            f"Variables:\n{', '.join(f'{var.name}={var.value}' for var in self.variables)}"
        )


def _interpreter_name_version() -> tuple[str, str]:
    if hasattr(sys, "implementation"):
        impl: sys._version_info = sys.implementation.version
        version = f"{impl.major}.{impl.minor}.{impl.micro}"
        kind = impl.releaselevel
        if kind != "final":
            version += kind[0] + str(impl.serial)
        return sys.implementation.name, version
    return "", "0.0.0"


def _get_package_info(dist: str = __PACKAGE_NAME__) -> _Package:
    try:
        return _Package(
            name=dist,
            version=version(dist),
            description=metadata(dist)["Summary"],
        )
    except PackageNotFoundError:
        return _Package(name=dist)


def _get_name(dist: str = __PACKAGE_NAME__) -> str:
    """Get name of the given distribution.

    Parameters:
        dist: A distribution name.

    Returns:
        A package name.
    """
    return _get_package_info(dist).name


def _get_version(dist: str = __PACKAGE_NAME__) -> str:
    """Get version of the given distribution.

    Parameters:
        dist: A distribution name.

    Returns:
        A version number.
    """
    return _get_package_info(dist).version


def _get_description(dist: str = __PACKAGE_NAME__) -> str:
    """Get description of the given distribution.

    Parameters:
        dist: A distribution name.

    Returns:
        A description string.
    """
    return _get_package_info(dist).description


def _get_debug_info() -> _Environment:
    """Get debug/environment information.

    Returns:
        Environment information.
    """
    py_name, py_version = _interpreter_name_version()
    packages: list[str] = [__PACKAGE_NAME__]
    variables: list[str] = [
        "PYTHONPATH",
        *[
            var
            for var in os.environ
            if var.startswith(__PACKAGE_NAME__.replace("-", "_"))
        ],
    ]
    return _Environment(
        interpreter_name=py_name,
        interpreter_version=py_version,
        interpreter_path=sys.executable,
        platform=platform.platform(),
        variables=[_Variable(var, val) for var in variables if (val := os.getenv(var))],
        packages=[_Package(pkg, _get_version(pkg)) for pkg in packages],
    )


def _get_installed_packages() -> list[_Package]:
    """Get all installed packages in current environment"""
    packages = []
    for dist in importlib.metadata.distributions():
        packages.append({"name": dist.metadata["Name"], "version": dist.version})
    return packages


if __name__ == "__main__":
    print(_get_debug_info())
