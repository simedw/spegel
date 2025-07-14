from unittest.mock import AsyncMock, Mock, patch

import pytest

from spegel.main import LinkManager, Spegel


class TestLinkManager:
    """Test the LinkManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_app = Mock()
        self.mock_app.current_view = "test"
        self.mock_app.original_content = {}
        self.link_manager = LinkManager(self.mock_app)

    def test_extract_links_from_markdown(self):
        """Test link extraction from markdown content."""
        content = """
        # Test Page
        Check out [Example](https://example.com) and [Local](/path).
        Also see [Fragment](#section) and [Vote](vote?id=123).
        """

        links = self.link_manager.extract_links_from_markdown(content)

        # Should extract valid links and filter out fragments and voting links
        assert len(links) == 2

        # Check first link
        text1, url1, start1, end1 = links[0]
        assert text1 == "Example"
        assert url1 == "https://example.com"
        assert start1 > 0
        assert end1 > start1

        # Check second link
        text2, url2, start2, end2 = links[1]
        assert text2 == "Local"
        assert url2 == "/path"

    def test_extract_links_handles_angle_brackets(self):
        """Test that angle brackets from html2text are removed."""
        content = "Visit [Site](<https://example.com>) for more info."

        links = self.link_manager.extract_links_from_markdown(content)

        assert len(links) == 1
        text, url, _, _ = links[0]
        assert text == "Site"
        assert url == "https://example.com"  # Angle brackets removed

    def test_extract_links_filters_invalid_urls(self):
        """Test that invalid URLs are filtered out."""
        content = """
        [Empty]() link and [Fragment](#section) link.
        [Vote](vote?id=123) and [Site](from?site=hn.com) links.
        [Valid](https://example.com) link.
        """

        links = self.link_manager.extract_links_from_markdown(content)

        # Should only extract the valid link
        assert len(links) == 1
        text, url, _, _ = links[0]
        assert text == "Valid"
        assert url == "https://example.com"

    def test_update_links_current_view(self):
        """Test updating links when viewing the current tab."""
        content = "Visit [Example](https://example.com) for more."

        self.link_manager.update_links(content, "test")

        assert len(self.link_manager.current_links) == 1
        assert self.link_manager.current_link_index == -1

    def test_update_links_different_view(self):
        """Test updating links when not viewing the current tab."""
        content = "Visit [Example](https://example.com) for more."

        self.link_manager.update_links(content, "other")

        # Links should not be updated for different view
        assert len(self.link_manager.current_links) == 0
        assert self.link_manager.current_link_index == -1

    def test_navigate_next_link(self):
        """Test navigating to next link."""
        # Set up links
        self.link_manager.current_links = [
            ("Link1", "http://example1.com", 0, 10),
            ("Link2", "http://example2.com", 20, 30),
        ]
        self.link_manager.current_link_index = -1

        # Navigate next
        self.link_manager.navigate_next_link()
        assert self.link_manager.current_link_index == 0

        # Navigate next again
        self.link_manager.navigate_next_link()
        assert self.link_manager.current_link_index == 1

        # Should wrap around
        self.link_manager.navigate_next_link()
        assert self.link_manager.current_link_index == 0

    def test_navigate_prev_link(self):
        """Test navigating to previous link."""
        # Set up links
        self.link_manager.current_links = [
            ("Link1", "http://example1.com", 0, 10),
            ("Link2", "http://example2.com", 20, 30),
        ]
        self.link_manager.current_link_index = 1

        # Navigate previous
        self.link_manager.navigate_prev_link()
        assert self.link_manager.current_link_index == 0

        # Should wrap around
        self.link_manager.navigate_prev_link()
        assert self.link_manager.current_link_index == 1

    def test_navigate_no_links(self):
        """Test navigation when no links are available."""
        self.link_manager.current_links = []
        original_index = self.link_manager.current_link_index

        self.link_manager.navigate_next_link()
        assert self.link_manager.current_link_index == original_index

        self.link_manager.navigate_prev_link()
        assert self.link_manager.current_link_index == original_index

    @pytest.mark.asyncio
    async def test_open_current_link(self):
        """Test opening the currently selected link."""
        # Set up mock app methods
        self.mock_app._resolve_url = Mock(return_value="https://resolved.com")
        self.mock_app.notify = Mock()
        self.mock_app.fetch_and_display_url = AsyncMock()

        # Set up links
        self.link_manager.current_links = [
            ("Example", "/relative", 0, 10),
        ]
        self.link_manager.current_link_index = 0

        await self.link_manager.open_current_link()

        # Should resolve URL and navigate
        self.mock_app._resolve_url.assert_called_once_with("/relative")
        self.mock_app.notify.assert_called_once()
        self.mock_app.fetch_and_display_url.assert_called_once_with(
            "https://resolved.com"
        )

    @pytest.mark.asyncio
    async def test_open_current_link_no_selection(self):
        """Test opening link when no link is selected."""
        self.mock_app.notify = Mock()
        self.mock_app.fetch_and_display_url = AsyncMock()

        self.link_manager.current_links = [("Example", "http://example.com", 0, 10)]
        self.link_manager.current_link_index = -1  # No selection

        await self.link_manager.open_current_link()

        # Should show warning, not navigate
        self.mock_app.notify.assert_called_once_with(
            "No link selected", severity="warning"
        )
        self.mock_app.fetch_and_display_url.assert_not_called()

    def test_highlight_current_link(self):
        """Test highlighting the currently selected link."""
        content = "Visit [Example](https://example.com) for more info."
        self.link_manager.current_links = [
            ("Example", "https://example.com", 6, 36),  # Correct end position
        ]
        self.link_manager.current_link_index = 0

        highlighted = self.link_manager.highlight_current_link(content)

        expected = "Visit **→ [Example](https://example.com) ←** for more info."
        assert highlighted == expected

    def test_highlight_current_link_no_selection(self):
        """Test highlighting when no link is selected."""
        content = "Visit [Example](https://example.com) for more info."
        self.link_manager.current_links = [
            ("Example", "https://example.com", 6, 35),
        ]
        self.link_manager.current_link_index = -1  # No selection

        highlighted = self.link_manager.highlight_current_link(content)

        # Should return content unchanged
        assert highlighted == content

    def test_escape_markup(self):
        """Test markup escaping for notifications."""
        text = "Link [text] with ! marks"
        escaped = self.link_manager._escape_markup(text)

        assert escaped == "Link \\[text\\] with \\! marks"


class TestURLResolution:
    """Test URL resolution functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("spegel.main.load_config") as mock_load_config:
            mock_config = Mock()
            mock_config.views = []
            mock_config.settings.default_view = "raw"
            mock_config.settings.app_title = "Test"
            mock_load_config.return_value = mock_config

            with patch("spegel.main.create_client") as mock_create_client:
                mock_create_client.return_value = None
                self.app = Spegel()
                self.app.current_url = "https://example.com/page"

    def test_resolve_absolute_url(self):
        """Test resolving absolute URLs."""
        url = "https://other.com/path"
        resolved = self.app._resolve_url(url)
        assert resolved == "https://other.com/path"

    def test_resolve_relative_url_with_base(self):
        """Test resolving relative URLs with base URL."""
        url = "/other-page"
        resolved = self.app._resolve_url(url)
        assert resolved == "https://example.com/other-page"

    def test_resolve_relative_url_no_base(self):
        """Test resolving relative URLs without base URL."""
        self.app.current_url = None
        url = "/page"
        resolved = self.app._resolve_url(url)
        assert resolved == "https://example.com/page"  # Fallback

    def test_resolve_relative_path_with_base(self):
        """Test resolving relative paths with base URL."""
        url = "other-page"
        resolved = self.app._resolve_url(url)
        assert resolved == "https://example.com/other-page"

    def test_resolve_relative_path_no_base(self):
        """Test resolving relative paths without base URL."""
        self.app.current_url = None
        url = "page"
        resolved = self.app._resolve_url(url)
        assert resolved == "https://page"


class TestHistoryManagement:
    """Test URL history management."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("spegel.main.load_config") as mock_load_config:
            mock_config = Mock()
            mock_config.views = []
            mock_config.settings.default_view = "raw"
            mock_config.settings.app_title = "Test"
            mock_load_config.return_value = mock_config

            with patch("spegel.main.create_client") as mock_create_client:
                mock_create_client.return_value = None
                self.app = Spegel()

    @pytest.mark.asyncio
    async def test_history_adds_new_urls(self):
        """Test that new URLs are added to history."""
        with patch.object(
            self.app, "_process_all_views_parallel", new_callable=AsyncMock
        ):
            with patch(
                "spegel.main.fetch_url_blocking", return_value="<html>Test</html>"
            ):
                with patch.object(self.app, "query_one", return_value=Mock()):
                    await self.app.fetch_and_display_url("https://example.com")
                    await self.app.fetch_and_display_url("https://other.com")

                    assert len(self.app.url_history) == 2
                    assert self.app.url_history == [
                        "https://example.com",
                        "https://other.com",
                    ]

    @pytest.mark.asyncio
    async def test_history_no_duplicate_consecutive(self):
        """Test that consecutive duplicate URLs are not added."""
        with patch.object(
            self.app, "_process_all_views_parallel", new_callable=AsyncMock
        ):
            with patch(
                "spegel.main.fetch_url_blocking", return_value="<html>Test</html>"
            ):
                with patch.object(self.app, "query_one", return_value=Mock()):
                    await self.app.fetch_and_display_url("https://example.com")
                    await self.app.fetch_and_display_url(
                        "https://example.com"
                    )  # Same URL

                    assert len(self.app.url_history) == 1
                    assert self.app.url_history == ["https://example.com"]

    @pytest.mark.asyncio
    async def test_history_size_limit(self):
        """Test that history is limited to 50 entries."""
        with patch.object(
            self.app, "_process_all_views_parallel", new_callable=AsyncMock
        ):
            with patch(
                "spegel.main.fetch_url_blocking", return_value="<html>Test</html>"
            ):
                with patch.object(self.app, "query_one", return_value=Mock()):
                    # Add 60 URLs to test size limit
                    for i in range(60):
                        await self.app.fetch_and_display_url(f"https://example{i}.com")

                    assert len(self.app.url_history) == 50
                    # Should keep the last 50
                    assert self.app.url_history[0] == "https://example10.com"
                    assert self.app.url_history[-1] == "https://example59.com"

    @pytest.mark.asyncio
    async def test_go_back_navigation(self):
        """Test go back navigation functionality."""
        # Set up history
        self.app.url_history = ["https://first.com", "https://second.com"]

        with patch.object(
            self.app, "fetch_and_display_url", new_callable=AsyncMock
        ) as mock_fetch:
            self.app.action_go_back()

            # Should navigate to previous URL
            mock_fetch.assert_called_once_with("https://first.com")
            # History should be updated
            assert len(self.app.url_history) == 0  # Both entries removed

    def test_go_back_insufficient_history(self):
        """Test go back when there's insufficient history."""
        self.app.url_history = ["https://only.com"]
        self.app.notify = Mock()

        self.app.action_go_back()

        # Should show warning message
        self.app.notify.assert_called_once_with(
            "No previous page to go back to", severity="warning", timeout=2
        )

    def test_go_back_empty_history(self):
        """Test go back when history is empty."""
        self.app.url_history = []
        self.app.notify = Mock()

        self.app.action_go_back()

        # Should show warning message
        self.app.notify.assert_called_once_with(
            "No previous page to go back to", severity="warning", timeout=2
        )


class TestURLInputHandling:
    """Test URL input and submission handling."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("spegel.main.load_config") as mock_load_config:
            mock_config = Mock()
            mock_config.views = []
            mock_config.settings.default_view = "raw"
            mock_config.settings.app_title = "Test"
            mock_load_config.return_value = mock_config

            with patch("spegel.main.create_client") as mock_create_client:
                mock_create_client.return_value = None
                self.app = Spegel()

    @pytest.mark.asyncio
    async def test_handle_url_submission_adds_https(self):
        """Test that URL submission adds https:// if missing."""
        mock_event = Mock()
        mock_event.value = "example.com"

        with patch.object(self.app, "action_hide_overlays") as mock_hide:
            with patch.object(
                self.app, "fetch_and_display_url", new_callable=AsyncMock
            ) as mock_fetch:
                await self.app.handle_url_submission(mock_event)

                mock_hide.assert_called_once()
                mock_fetch.assert_called_once_with("https://example.com")

    @pytest.mark.asyncio
    async def test_handle_url_submission_preserves_scheme(self):
        """Test that existing URL schemes are preserved."""
        mock_event = Mock()
        mock_event.value = "http://example.com"

        with patch.object(self.app, "action_hide_overlays") as mock_hide:
            with patch.object(
                self.app, "fetch_and_display_url", new_callable=AsyncMock
            ) as mock_fetch:
                await self.app.handle_url_submission(mock_event)

                mock_hide.assert_called_once()
                mock_fetch.assert_called_once_with("http://example.com")

    @pytest.mark.asyncio
    async def test_handle_url_submission_empty_url(self):
        """Test handling of empty URL submission."""
        mock_event = Mock()
        mock_event.value = "   "  # Whitespace only

        with patch.object(
            self.app, "fetch_and_display_url", new_callable=AsyncMock
        ) as mock_fetch:
            await self.app.handle_url_submission(mock_event)

            # Should not fetch anything
            mock_fetch.assert_not_called()

    def test_action_show_url_input(self):
        """Test showing the URL input overlay."""
        self.app.url_input_visible = False
        self.app.prompt_editor_visible = False
        self.app.current_url = "https://example.com"

        # Mock UI elements
        mock_input = Mock()
        self.app.query_one = Mock(return_value=mock_input)
        self.app.add_class = Mock()

        self.app.action_show_url_input()

        assert self.app.url_input_visible
        self.app.add_class.assert_called_once_with("url-input-visible")
        mock_input.focus.assert_called_once()
        assert mock_input.value == "https://example.com"  # Should pre-fill current URL

    def test_action_hide_overlays(self):
        """Test hiding URL input and prompt editor overlays."""
        self.app.url_input_visible = True
        self.app.prompt_editor_visible = True
        self.app.current_view = "test"

        # Mock UI elements
        mock_content = Mock()
        self.app.query_one = Mock(return_value=mock_content)
        self.app.remove_class = Mock()

        self.app.action_hide_overlays()

        assert not self.app.url_input_visible
        assert not self.app.prompt_editor_visible
        self.app.remove_class.assert_any_call("url-input-visible")
        self.app.remove_class.assert_any_call("prompt-editor-visible")
        mock_content.focus.assert_called_once()
