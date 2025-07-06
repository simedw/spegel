from __future__ import annotations

import nox

PYTHON_VERSIONS: list[str] = ["3.11", "3.12", "3.13"]


@nox.session(venv_backend="uv", tags=["fix"])
def coverage(session: nox.Session) -> None:
    """Run tests with coverage reporting."""
    session.install("coverage[toml]", "pytest", "pytest-cov", "pytest-asyncio")
    session.install("-e", ".")

    session.run(
        "pytest",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-config=pyproject.toml",
    )

    session.log("Coverage HTML report generated in htmlcov/")


@nox.session(venv_backend="uv", tags=["lint"])
def ruff_check(session: nox.Session) -> None:
    """Run ruff linting and formatting checks (CI-friendly, no changes)."""
    session.install("ruff")
    session.run(
        "ruff",
        "check",
        ".",
        "--config",
        "pyproject.toml",
    )
    session.run(
        "ruff",
        "format",
        ".",
        "--check",
        "--config",
        "pyproject.toml",
    )


@nox.session(venv_backend="uv", tags=["lint", "fix"])
def ruff_check_fix(session: nox.Session) -> None:
    """Run ruff linting and formatting but with a fix attempt (development)."""
    session.install("ruff")
    session.run(
        "ruff",
        "check",
        ".",
        "--fix",
        "--config",
        "pyproject.toml",
    )
    session.run(
        "ruff",
        "check",
        ".",
        "--config",
        "pyproject.toml",
    )
    session.run(
        "ruff",
        "format",
        ".",
        "--check",
        "--config",
        "pyproject.toml",
    )


@nox.session(venv_backend="uv", tags=["lint", "fix"])
def ruff_fix(session: nox.Session) -> None:
    """Run ruff linting and formatting with auto-fix (development)."""
    session.install("ruff")
    session.run(
        "ruff",
        "check",
        ".",
        "--fix",
        "--config",
        "pyproject.toml",
    )
    session.run(
        "ruff",
        "format",
        ".",
        "--config",
        "pyproject.toml",
    )


@nox.session(python=PYTHON_VERSIONS, venv_backend="uv")
def all_versions(session: nox.Session) -> None:
    """Run the unit test suite."""
    session.install("-e", ".")
    session.install("pytest", "pytest-asyncio")
    session.run("pytest")
