import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from spegel.main import Spegel


class TestContentFetching:
    """Test content fetching and processing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("spegel.main.load_config") as mock_load_config:
            mock_config = Mock()
            mock_config.views = [
                Mock(id="raw", enabled=True, auto_load=True),
                Mock(id="summary", enabled=True, auto_load=False),
            ]
            mock_config.settings.default_view = "raw"
            mock_config.settings.app_title = "Test"
            mock_load_config.return_value = mock_config

            with patch("spegel.main.create_client") as mock_create_client:
                mock_create_client.return_value = None
                self.app = Spegel()

                # Mock UI components
                self.app.query_one = Mock()
                self.mock_content_widget = Mock()
                self.app.query_one.return_value = self.mock_content_widget

    @pytest.mark.asyncio
    async def test_fetch_and_display_url_success(self):
        """Test successful URL fetching and content display."""
        url = "https://example.com"
        html_content = (
            "<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>"
        )

        with patch(
            "spegel.main.fetch_url_blocking", return_value=html_content
        ) as mock_fetch:
            with patch.object(
                self.app, "_process_all_views_parallel", new_callable=AsyncMock
            ) as mock_process:
                await self.app.fetch_and_display_url(url)

                # Should update current state
                assert self.app.current_url == url
                assert self.app.title == f"LLM Browser - {url}"
                assert self.app.raw_html == html_content

                # Should call fetch_url_blocking with URL
                mock_fetch.assert_called_once_with(url)

                # Should process views
                mock_process.assert_called_once()

                # Should add URL to history
                assert url in self.app.url_history

    @pytest.mark.asyncio
    async def test_fetch_and_display_url_network_failure(self):
        """Test handling of network failures during URL fetching."""
        url = "https://example.com"

        with patch("spegel.main.fetch_url_blocking", return_value=None) as mock_fetch:
            await self.app.fetch_and_display_url(url)

            # Should attempt to fetch
            mock_fetch.assert_called_once_with(url)

            # Should show failure message
            self.mock_content_widget.update.assert_called_with(f"Failed to load {url}")

            # Should not update current state
            assert self.app.current_url != url
            assert self.app.raw_html == ""

    @pytest.mark.asyncio
    async def test_fetch_and_display_url_exception(self):
        """Test handling of exceptions during URL fetching."""
        url = "https://example.com"
        error_message = "Connection timeout"

        with patch(
            "spegel.main.fetch_url_blocking", side_effect=Exception(error_message)
        ):
            await self.app.fetch_and_display_url(url)

            # Should show error message
            expected_message = f"Error loading {url}: {error_message}"
            self.mock_content_widget.update.assert_called_with(expected_message)

    @pytest.mark.asyncio
    async def test_fetch_and_display_url_history_management(self):
        """Test URL history management during fetching."""
        urls = [
            "https://first.com",
            "https://second.com",
            "https://first.com",
        ]  # Duplicate

        with patch("spegel.main.fetch_url_blocking", return_value="<html>Test</html>"):
            with patch.object(
                self.app, "_process_all_views_parallel", new_callable=AsyncMock
            ):
                for url in urls:
                    await self.app.fetch_and_display_url(url)

                # Should have all URLs in history (including duplicate)
                assert self.app.url_history == urls

    @pytest.mark.asyncio
    async def test_fetch_and_display_url_resets_view_state(self):
        """Test that fetching URL resets view processing state."""
        self.app.views_loaded = {"summary"}
        self.app.views_loading = {"analysis"}

        with patch("spegel.main.fetch_url_blocking", return_value="<html>Test</html>"):
            with patch.object(
                self.app, "_process_all_views_parallel", new_callable=AsyncMock
            ):
                with patch.object(self.app, "_reset_tab_names") as mock_reset:
                    await self.app.fetch_and_display_url("https://example.com")

                    # Should reset view states
                    assert self.app.views_loaded == set()
                    assert self.app.views_loading == set()

                    # Should reset tab names
                    mock_reset.assert_called_once()


class TestViewProcessing:
    """Test view processing and management."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("spegel.main.load_config") as mock_load_config:
            # Create mock views with different auto_load settings
            mock_raw_view = Mock()
            mock_raw_view.id = "raw"
            mock_raw_view.auto_load = True
            mock_raw_view.enabled = True

            mock_summary_view = Mock()
            mock_summary_view.id = "summary"
            mock_summary_view.auto_load = True
            mock_summary_view.enabled = True

            mock_analysis_view = Mock()
            mock_analysis_view.id = "analysis"
            mock_analysis_view.auto_load = False
            mock_analysis_view.enabled = True

            mock_config = Mock()
            mock_config.views = [mock_raw_view, mock_summary_view, mock_analysis_view]
            mock_config.settings.default_view = "raw"
            mock_config.settings.app_title = "Test"
            mock_load_config.return_value = mock_config

            with patch("spegel.main.create_client") as mock_create_client:
                mock_create_client.return_value = None
                self.app = Spegel()

                # Set up views dict
                self.app.views = {
                    "raw": mock_raw_view,
                    "summary": mock_summary_view,
                    "analysis": mock_analysis_view,
                }

    @pytest.mark.asyncio
    async def test_process_all_views_parallel_auto_load_only(self):
        """Test that only auto-load views are processed initially."""
        with patch.object(self.app, "_process_single_view", new_callable=AsyncMock):
            # Mock asyncio.create_task to track what gets created
            created_tasks = []
            original_create_task = asyncio.create_task

            def mock_create_task(coro):
                created_tasks.append(coro)
                return original_create_task(coro)

            with patch("asyncio.create_task", side_effect=mock_create_task):
                await self.app._process_all_views_parallel()

                # Should mark auto-load views as loading
                assert "raw" in self.app.views_loading
                assert "summary" in self.app.views_loading
                assert "analysis" not in self.app.views_loading  # Not auto-load

                # Should create tasks for auto-load views
                assert len(created_tasks) == 2

    @pytest.mark.asyncio
    async def test_process_single_view_raw_view(self):
        """Test processing of raw view."""
        self.app.raw_html = "<html><body><h1>Test</h1></body></html>"

        # Mock UI components
        mock_content_widget = Mock()
        self.app.query_one = Mock(return_value=mock_content_widget)

        # Mock view processing
        with patch.object(
            self.app, "update_view_content", new_callable=AsyncMock
        ) as mock_update:
            await self.app._process_single_view("raw")

            # Should update loading message first
            mock_content_widget.update.assert_called()

            # Should process the view
            mock_update.assert_called_once_with("raw")

            # Should mark as loaded and update tab
            assert "raw" in self.app.views_loaded
            assert "raw" not in self.app.views_loading

    @pytest.mark.asyncio
    async def test_process_single_view_llm_view(self):
        """Test processing of LLM-based view."""
        self.app.raw_html = "<html><body><h1>Test</h1></body></html>"
        self.app.llm_client = Mock()  # Mock LLM client to simulate availability

        # Mock UI components
        mock_content_widget = Mock()
        self.app.query_one = Mock(return_value=mock_content_widget)

        # Mock view processing
        with patch.object(
            self.app, "update_view_content", new_callable=AsyncMock
        ) as mock_update:
            await self.app._process_single_view("summary")

            # Should show AI preparation message
            calls = mock_content_widget.update.call_args_list
            ai_message_found = any(
                "⏳ Preparing AI Analysis" in str(call) for call in calls
            )
            assert ai_message_found

            # Should process the view
            mock_update.assert_called_once_with("summary")

    @pytest.mark.asyncio
    async def test_process_single_view_no_llm(self):
        """Test processing of LLM view when LLM is not available."""
        self.app.raw_html = "<html><body><h1>Test</h1></body></html>"
        self.app.llm_client = None  # No LLM client to simulate unavailability

        # Mock UI components
        mock_content_widget = Mock()
        self.app.query_one = Mock(return_value=mock_content_widget)

        # Mock view processing
        with patch.object(
            self.app, "update_view_content", new_callable=AsyncMock
        ) as mock_update:
            await self.app._process_single_view("summary")

            # Should show LLM unavailable message
            calls = mock_content_widget.update.call_args_list
            llm_message_found = any("LLM not available" in str(call) for call in calls)
            assert llm_message_found

            # Should still process the view
            mock_update.assert_called_once_with("summary")

    @pytest.mark.asyncio
    async def test_process_single_view_error_handling(self):
        """Test error handling during view processing."""
        self.app.raw_html = "<html><body><h1>Test</h1></body></html>"

        # Mock UI components
        mock_content_widget = Mock()
        self.app.query_one = Mock(return_value=mock_content_widget)

        # Mock view processing to raise exception
        error_message = "Processing failed"
        with patch.object(
            self.app, "update_view_content", side_effect=Exception(error_message)
        ):
            await self.app._process_single_view("summary")

            # Should handle error and show error message
            calls = mock_content_widget.update.call_args_list
            error_message_found = any(
                "❌ Error" in str(call) and error_message in str(call) for call in calls
            )
            assert error_message_found

            # Should still mark as loaded
            assert "summary" in self.app.views_loaded
            assert "summary" not in self.app.views_loading

    @pytest.mark.asyncio
    async def test_update_view_content_raw_view(self):
        """Test updating content for raw view."""
        self.app.raw_html = (
            "<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>"
        )
        self.app.current_url = "https://example.com"
        self.app.current_view = "raw"

        # Mock UI components and managers
        mock_content_widget = Mock()
        self.app.query_one = Mock(return_value=mock_content_widget)
        self.app.link_manager = Mock()
        self.app.link_manager.current_links = []  # Mock as empty list
        self.app.notify = Mock()

        # Mock html_to_markdown
        with patch(
            "spegel.main.html_to_markdown", return_value="# Test\nHello world"
        ) as mock_convert:
            await self.app.update_view_content("raw")

            # Should convert HTML to markdown
            mock_convert.assert_called_once_with(
                self.app.raw_html, self.app.current_url
            )

            # Should update content widget
            mock_content_widget.update.assert_called_with("# Test\nHello world")

            # Should update links
            self.app.link_manager.update_links.assert_called_once_with(
                "# Test\nHello world", "raw"
            )

            # Should store original content
            assert self.app.original_content["raw"] == "# Test\nHello world"

    @pytest.mark.asyncio
    async def test_update_view_content_llm_view(self):
        """Test updating content for LLM-based view."""
        self.app.raw_html = "<html><body><h1>Test</h1></body></html>"
        self.app.current_view = "summary"

        # Mock UI components and managers
        mock_content_widget = Mock()
        self.app.query_one = Mock(return_value=mock_content_widget)
        self.app.link_manager = Mock()
        self.app.link_manager.current_links = []  # Mock as empty list
        self.app.scroll_manager = Mock()

        # Mock view configuration
        mock_view = Mock()
        self.app.views = {"summary": mock_view}

        # Mock streaming
        async def mock_stream_view(*args):
            yield "This is "
            yield "a summary."

        with patch("spegel.main.stream_view", mock_stream_view):
            await self.app.update_view_content("summary")

            # Should show streaming message first
            mock_content_widget.update.assert_any_call("*Streaming response...*\n\n")

            # Should preserve scroll during streaming
            self.app.scroll_manager.update_content_preserve_scroll.assert_called()

            # Should update links with final content
            self.app.link_manager.update_links.assert_called_with(
                "This is a summary.", "summary"
            )

            # Should store original content
            assert self.app.original_content["summary"] == "This is a summary."


class TestTabManagement:
    """Test tab switching and view management."""

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

                # Set up views
                self.app.views = {
                    "raw": Mock(auto_load=True),
                    "summary": Mock(auto_load=False),
                }

    @pytest.mark.asyncio
    async def test_handle_tab_change_to_loaded_view(self):
        """Test switching to an already loaded view."""
        # Set up existing content
        self.app.original_content = {"summary": "Existing content"}
        self.app.views_loaded = {"summary"}

        # Mock link manager
        self.app.link_manager = Mock()
        self.app.link_manager.current_links = []  # Mock as empty list
        self.app.notify = Mock()

        # Mock tab event
        mock_event = Mock()
        mock_event.tab.id = "--content-tab-summary"

        await self.app.handle_tab_change(mock_event)

        # Should update current view
        assert self.app.current_view == "summary"

        # Should update links for the view
        self.app.link_manager.update_links.assert_called_once_with(
            "Existing content", "summary"
        )

    @pytest.mark.asyncio
    async def test_handle_tab_change_to_unloaded_view(self):
        """Test switching to a view that needs on-demand loading."""
        # Set up state
        self.app.raw_html = "<html>Test</html>"
        self.app.views_loaded = set()
        self.app.views_loading = set()

        # Mock tab event
        mock_event = Mock()
        mock_event.tab.id = "--content-tab-summary"

        def mock_create_task(coro):
            # Close the coroutine to avoid RuntimeWarning
            coro.close()
            return Mock()

        with patch(
            "asyncio.create_task", side_effect=mock_create_task
        ) as mock_create_task_patch:
            with patch.object(self.app, "_update_tab_name") as mock_update_tab:
                await self.app.handle_tab_change(mock_event)

                # Should mark as loading and start processing
                assert "summary" in self.app.views_loading
                mock_update_tab.assert_called_with("summary")
                mock_create_task_patch.assert_called_once()

    def test_action_switch_tab(self):
        """Test programmatic tab switching."""
        # Mock TabbedContent
        mock_tabbed = Mock()
        self.app.query_one = Mock(return_value=mock_tabbed)

        self.app.action_switch_tab("summary")

        # Should update current view and activate tab
        assert self.app.current_view == "summary"
        mock_tabbed.active = "summary"

    def test_update_tab_name_loading(self):
        """Test updating tab name with loading indicator."""
        self.app.views_loading = {"summary"}
        self.app.views_loaded = set()

        mock_view = Mock()
        mock_view.name = "Summary"
        self.app.views = {"summary": mock_view}

        # Mock UI update
        with patch.object(self.app, "call_later") as mock_call_later:
            self.app._update_tab_name("summary")

            # Should schedule UI update
            mock_call_later.assert_called_once()

    def test_update_tab_name_loaded(self):
        """Test updating tab name with loaded indicator."""
        self.app.views_loading = set()
        self.app.views_loaded = {"summary"}

        mock_view = Mock()
        mock_view.name = "Summary"
        self.app.views = {"summary": mock_view}

        # Mock UI update
        with patch.object(self.app, "call_later") as mock_call_later:
            self.app._update_tab_name("summary")

            # Should schedule UI update
            mock_call_later.assert_called_once()

    def test_reset_tab_names(self):
        """Test resetting all tab names to base names."""
        mock_view1 = Mock()
        mock_view1.name = "Raw"
        mock_view2 = Mock()
        mock_view2.name = "Summary"

        self.app.views = {
            "raw": mock_view1,
            "summary": mock_view2,
        }

        # Mock tab panes
        mock_tab_panes = [Mock(), Mock()]
        self.app.query_one = Mock(side_effect=mock_tab_panes)

        self.app._reset_tab_names()

        # Should update labels for both tabs
        assert mock_tab_panes[0].label == "Raw"
        assert mock_tab_panes[1].label == "Summary"


class TestContentState:
    """Test content state management."""

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

    def test_original_content_storage(self):
        """Test that original content is stored for each view."""
        # Simulate content processing
        self.app.original_content["raw"] = "# Raw Content"
        self.app.original_content["summary"] = "Summary content"

        assert len(self.app.original_content) == 2
        assert self.app.original_content["raw"] == "# Raw Content"
        assert self.app.original_content["summary"] == "Summary content"

    def test_view_state_tracking(self):
        """Test tracking of view loading states."""
        # Simulate view processing states
        self.app.views_loading.add("summary")
        self.app.views_loaded.add("raw")

        assert "summary" in self.app.views_loading
        assert "raw" in self.app.views_loaded
        assert "summary" not in self.app.views_loaded
        assert "raw" not in self.app.views_loading

    def test_current_view_tracking(self):
        """Test current view state tracking."""
        self.app.current_view = "summary"
        assert self.app.current_view == "summary"

        self.app.current_view = "raw"
        assert self.app.current_view == "raw"
