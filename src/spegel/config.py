from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import tomllib
from pydantic import BaseModel, Field, model_validator


"""Configuration handling for Spegel.

This module is responsible for:
• Defining strongly-typed configuration models using Pydantic.
• Loading configuration TOML files from the well-known locations.
• Providing fallback defaults so the app can run with zero user config.
"""

__all__ = [
    "View",
    "AI",
    "Settings",
    "UI",
    "FullConfig",
    "load_config",
    "DEFAULT_CONFIG_DICT",
]


class View(BaseModel):
    id: str = Field(..., description="Unique identifier for the view (used as tab id)")
    name: str
    hotkey: str = Field(..., description="Keyboard shortcut to activate this view")
    order: int = 0
    enabled: bool = True
    auto_load: bool = False
    description: str = ""
    icon: str = ""
    prompt: str = ""
    model: str = ""  # Optional model override for this view

    @model_validator(mode="after")
    def validate_hotkey(cls, values):  # type: ignore[override]
        if len(values.hotkey) != 1:
            raise ValueError("Hotkey must be a single character")
        return values


class AI(BaseModel):
    default_model: str = "gemini/gemini-2.5-flash-lite-preview-06-17"


class Settings(BaseModel):
    default_view: str = "terminal"
    max_history: int = 50
    stream_delay: float = 0.01
    app_title: str = "Spegel"


class UI(BaseModel):
    show_icons: bool = True
    compact_mode: bool = False


class FullConfig(BaseModel):
    settings: Settings = Settings()
    ai: AI = AI()
    ui: UI = UI()
    views: List[View] = Field(default_factory=list)

    def view_map(self) -> Dict[str, View]:
        """Return a mapping of view_id → View for quick lookup."""
        return {v.id: v for v in self.views if v.enabled}


# --------------------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------------------

DEFAULT_CONFIG_DICT: Dict[str, Any] = {
    "settings": {
        "default_view": "terminal",
        "max_history": 50,
        "stream_delay": 0.01,
        "app_title": "Spegel",
    },
    "ai": {
        "default_model": "gemini/gemini-2.5-flash-lite-preview-06-17",
    },
    "ui": {"show_icons": True, "compact_mode": False},
    "views": [
        {
            "id": "raw",
            "name": "Raw View",
            "hotkey": "1",
            "order": 1,
            "enabled": True,
            "auto_load": True,
            "description": "Clean HTML rendering (no LLM)",
            "icon": "📄",
            "prompt": "",
        },
        {
            "id": "terminal",
            "name": "Terminal",
            "hotkey": "2",
            "order": 2,
            "enabled": True,
            "auto_load": True,
            "description": "Terminal-optimized markdown for efficient browsing",
            "icon": "💻",
            "prompt": "Transform this webpage into the perfect terminal browsing experience! ...",
        },
    ],
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dicts (override wins)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        elif key == "views" and isinstance(value, list):
            # Special handling for views: if custom config provides views, replace defaults entirely
            result[key] = value
        else:
            result[key] = value
    return result


def load_config() -> FullConfig:
    """Load configuration from TOML files or fall back to defaults.

    Search order:
      • ./.spegel.toml
      • ~/.spegel.toml
      • ~/.config/spegel/config.toml
    """

    config_paths = [
        Path(".spegel.toml"),
        Path.home() / ".spegel.toml",
        Path.home() / ".config" / "spegel" / "config.toml",
    ]

    merged: Dict[str, Any] = DEFAULT_CONFIG_DICT

    # Only load the first config file found, not all of them
    for path in config_paths:
        if path.is_file():
            try:
                with open(path, "rb") as f:
                    user_cfg = tomllib.load(f)
                    merged = _deep_merge(merged, user_cfg)  # type: ignore[arg-type]
                    break  # Stop after loading the first config file
            except Exception as exc:
                print(f"⚠️  Failed to load config from {path}: {exc}")
                continue  # Try the next config file if this one fails

    try:
        return FullConfig.model_validate(merged)
    except Exception as exc:  # pragma: no cover
        print("👉 Falling back to default config due to validation error:", exc)
        return FullConfig()
