import os
from pathlib import Path
import tempfile

from pydantic import ValidationError
import pytest

from spegel.config import DEFAULT_CONFIG_DICT, FullConfig, View, load_config


def test_view_validation():
    """Test View model validation."""
    # Valid view
    view = View(id="test", name="Test View", hotkey="t", prompt="Test prompt")
    assert view.id == "test"
    assert view.hotkey == "t"

    # Invalid hotkey (too long)
    with pytest.raises(ValidationError):
        View(id="test", name="Test", hotkey="ab", prompt="")

    # Invalid hotkey (empty)
    with pytest.raises(ValidationError):
        View(id="test", name="Test", hotkey="", prompt="")


def test_default_config():
    """Test that default config loads without errors."""
    config = FullConfig.model_validate(DEFAULT_CONFIG_DICT)

    assert config.settings.default_view == "terminal"
    assert config.settings.app_title == "Spegel"
    assert len(config.views) >= 2  # raw and terminal views

    # Check that we have the expected default views
    view_ids = {v.id for v in config.views}
    assert "raw" in view_ids
    assert "terminal" in view_ids


def test_load_config_with_custom_file():
    """Test loading config from a custom TOML file."""
    custom_config = """
    [settings]
    default_view = "custom"
    app_title = "Custom Spegel"
    
    [[views]]
    id = "custom"
    name = "Custom View"
    hotkey = "c"
    prompt = "Custom prompt"
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(custom_config)
        f.flush()

        # Temporarily set the config file path
        original_cwd = os.getcwd()
        temp_dir = Path(f.name).parent
        os.chdir(temp_dir)

        # Rename to .spegel.toml so load_config finds it
        config_path = temp_dir / ".spegel.toml"
        Path(f.name).rename(config_path)

        try:
            config = load_config()
            assert config.settings.default_view == "custom"
            assert config.settings.app_title == "Custom Spegel"
            assert len(config.views) == 1
            assert config.views[0].id == "custom"
        finally:
            os.chdir(original_cwd)
            config_path.unlink(missing_ok=True)


def test_view_map():
    """Test the view_map helper method."""
    config = FullConfig.model_validate(DEFAULT_CONFIG_DICT)
    view_map = config.view_map()

    assert isinstance(view_map, dict)
    assert "raw" in view_map
    assert "terminal" in view_map
    assert view_map["raw"].name == "Raw View"


def test_config_merge_behavior():
    """Test that custom config merges with defaults properly."""
    # This tests the _deep_merge function indirectly
    custom_config = """
    [settings]
    app_title = "My Custom Spegel"
    # default_view should remain "terminal" from defaults
    
    [[views]]
    id = "raw"
    name = "My Raw View"
    hotkey = "1"
    enabled = true
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(custom_config)
        f.flush()

        original_cwd = os.getcwd()
        temp_dir = Path(f.name).parent
        os.chdir(temp_dir)

        config_path = temp_dir / ".spegel.toml"
        Path(f.name).rename(config_path)

        try:
            config = load_config()
            # Custom setting should override
            assert config.settings.app_title == "My Custom Spegel"
            # Default setting should remain
            assert config.settings.default_view == "terminal"
            # Should have both default views plus any custom ones
            assert len(config.views) >= 1
        finally:
            os.chdir(original_cwd)
            config_path.unlink(missing_ok=True)
