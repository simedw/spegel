from unittest.mock import Mock, patch

import pytest

from spegel.main import Spegel, main


class TestMainCLI:
    """Test the main CLI entry point."""

    def test_main_no_args(self):
        """Main with no arguments should start app with no initial URL."""
        with patch("spegel.main.Spegel") as mock_spegel_class:
            mock_app = Mock()
            mock_spegel_class.return_value = mock_app

            with patch("sys.argv", ["spegel"]):
                main()

            # Should create app with no initial URL
            mock_spegel_class.assert_called_once_with(initial_url=None)
            mock_app.run.assert_called_once()

    def test_main_with_url(self):
        """Main with URL argument should start app with that URL."""
        with patch("spegel.main.Spegel") as mock_spegel_class:
            mock_app = Mock()
            mock_spegel_class.return_value = mock_app

            with patch("sys.argv", ["spegel", "example.com"]):
                main()

            # Should create app with https:// prepended
            mock_spegel_class.assert_called_once_with(initial_url="https://example.com")
            mock_app.run.assert_called_once()

    def test_main_with_full_url(self):
        """Main with full URL should not modify it."""
        with patch("spegel.main.Spegel") as mock_spegel_class:
            mock_app = Mock()
            mock_spegel_class.return_value = mock_app

            with patch("sys.argv", ["spegel", "https://example.com"]):
                main()

            # Should use URL as-is
            mock_spegel_class.assert_called_once_with(initial_url="https://example.com")

    def test_main_with_http_url(self):
        """Main with http:// URL should not modify it."""
        with patch("spegel.main.Spegel") as mock_spegel_class:
            mock_app = Mock()
            mock_spegel_class.return_value = mock_app

            with patch("sys.argv", ["spegel", "http://example.com"]):
                main()

            # Should use URL as-is
            mock_spegel_class.assert_called_once_with(initial_url="http://example.com")

    def test_main_help(self):
        """Main with --help should show help and exit."""
        with patch("sys.argv", ["spegel", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # argparse exits with code 0 for help
            assert exc_info.value.code == 0


class TestSpegelApp:
    """Test the Spegel application class."""

    def test_init_no_initial_url(self):
        """Spegel should initialize with no startup URL by default."""
        with patch("spegel.main.load_config") as mock_load_config:
            mock_config = Mock()
            mock_config.views = []
            mock_config.settings.default_view = "raw"
            mock_config.settings.app_title = "Test"
            mock_load_config.return_value = mock_config

            with patch("spegel.main.create_client") as mock_create_client:
                mock_create_client.return_value = None

                app = Spegel()
                assert app._startup_url is None

    def test_init_with_initial_url(self):
        """Spegel should store initial URL when provided."""
        with patch("spegel.main.load_config") as mock_load_config:
            mock_config = Mock()
            mock_config.views = []
            mock_config.settings.default_view = "raw"
            mock_config.settings.app_title = "Test"
            mock_load_config.return_value = mock_config

            with patch("spegel.main.create_client") as mock_create_client:
                mock_create_client.return_value = None

                app = Spegel(initial_url="https://example.com")
                assert app._startup_url == "https://example.com"

    def test_on_mount_with_startup_url(self):
        """on_mount should trigger URL fetch if startup URL is provided."""
        with patch("spegel.main.load_config") as mock_load_config:
            mock_config = Mock()
            mock_config.views = []
            mock_config.settings.default_view = "raw"
            mock_config.settings.app_title = "Test"
            mock_load_config.return_value = mock_config

            with patch("spegel.main.create_client") as mock_create_client:
                mock_create_client.return_value = None

                app = Spegel(initial_url="https://example.com")

                # Mock the run_async_task method
                app.run_async_task = Mock()
                app._setup_bindings = Mock()

                app.on_mount()

                # Should have called run_async_task with fetch_and_display_url
                app.run_async_task.assert_called_once()
                # The argument should be a coroutine for fetch_and_display_url
                call_args = app.run_async_task.call_args[0][0]
                assert hasattr(call_args, "__await__")  # It's a coroutine
                # Properly close the coroutine to avoid RuntimeWarning
                call_args.close()

    def test_on_mount_no_startup_url(self):
        """on_mount should not trigger URL fetch if no startup URL."""
        with patch("spegel.main.load_config") as mock_load_config:
            mock_config = Mock()
            mock_config.views = []
            mock_config.settings.default_view = "raw"
            mock_config.settings.app_title = "Test"
            mock_load_config.return_value = mock_config

            with patch("spegel.main.create_client") as mock_create_client:
                mock_create_client.return_value = None

                app = Spegel()

                # Mock the run_async_task method
                app.run_async_task = Mock()
                app._setup_bindings = Mock()

                app.on_mount()

                # Should not have called run_async_task
                app.run_async_task.assert_not_called()


class TestURLHandling:
    """Test URL processing and validation."""

    @pytest.mark.parametrize(
        "input_url,expected",
        [
            ("example.com", "https://example.com"),
            ("www.example.com", "https://www.example.com"),
            ("subdomain.example.com", "https://subdomain.example.com"),
            ("https://example.com", "https://example.com"),
            ("http://example.com", "http://example.com"),
            ("https://www.example.com/path", "https://www.example.com/path"),
            ("example.com/path?query=1", "https://example.com/path?query=1"),
        ],
    )
    def test_url_preprocessing(self, input_url, expected):
        """Test that URLs are correctly preprocessed in main()."""
        with patch("spegel.main.Spegel") as mock_spegel_class:
            mock_app = Mock()
            mock_spegel_class.return_value = mock_app

            with patch("sys.argv", ["spegel", input_url]):
                main()

            mock_spegel_class.assert_called_once_with(initial_url=expected)


class TestAppConfiguration:
    """Test app configuration loading."""

    def test_app_uses_config_title(self):
        """App should use title from configuration."""
        with patch("spegel.main.load_config") as mock_load_config:
            mock_config = Mock()
            mock_config.views = []
            mock_config.settings.default_view = "raw"
            mock_config.settings.app_title = "Custom Title"
            mock_load_config.return_value = mock_config

            with patch("spegel.main.create_client") as mock_create_client:
                mock_create_client.return_value = None

                app = Spegel()
                assert app.title == "Custom Title"

    def test_app_uses_config_default_view(self):
        """App should use default view from configuration."""
        with patch("spegel.main.load_config") as mock_load_config:
            mock_config = Mock()
            mock_config.views = [
                Mock(id="custom", enabled=True),
                Mock(id="raw", enabled=True),
            ]
            mock_config.settings.default_view = "custom"
            mock_config.settings.app_title = "Test"
            mock_load_config.return_value = mock_config

            with patch("spegel.main.create_client") as mock_create_client:
                mock_create_client.return_value = None

                app = Spegel()
                assert app.current_view == "custom"
