from unittest.mock import AsyncMock, Mock, patch

import pytest

from spegel.main import Spegel


class TestKeyboardNavigation:
    """Test keyboard navigation and key handling."""

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

                # Mock managers
                self.app.link_manager = Mock()
                self.app.link_manager.current_links = [
                    ("Link1", "http://example.com", 0, 10)
                ]
                self.app.link_manager.current_link_index = 0

    @pytest.mark.asyncio
    async def test_tab_key_navigation_with_links(self):
        """Test Tab key navigation when links are available."""
        # Mock event
        mock_event = Mock()
        mock_event.key = "tab"

        # Set up state
        self.app.url_input_visible = False
        self.app.prompt_editor_visible = False

        # Mock action methods
        self.app.action_next_link = Mock()

        await self.app.on_key(mock_event)

        # Should navigate to next link
        self.app.action_next_link.assert_called_once()
        mock_event.prevent_default.assert_called_once()

    @pytest.mark.asyncio
    async def test_shift_tab_key_navigation_with_links(self):
        """Test Shift+Tab key navigation when links are available."""
        # Mock event
        mock_event = Mock()
        mock_event.key = "shift+tab"

        # Set up state
        self.app.url_input_visible = False
        self.app.prompt_editor_visible = False

        # Mock action methods
        self.app.action_prev_link = Mock()

        await self.app.on_key(mock_event)

        # Should navigate to previous link
        self.app.action_prev_link.assert_called_once()
        mock_event.prevent_default.assert_called_once()

    @pytest.mark.asyncio
    async def test_enter_key_opens_link(self):
        """Test Enter key opens currently selected link."""
        # Mock event
        mock_event = Mock()
        mock_event.key = "enter"

        # Set up state
        self.app.url_input_visible = False
        self.app.prompt_editor_visible = False

        # Mock action methods
        self.app.action_open_link = AsyncMock()

        await self.app.on_key(mock_event)

        # Should open current link
        self.app.action_open_link.assert_called_once()
        mock_event.prevent_default.assert_called_once()

    @pytest.mark.asyncio
    async def test_key_navigation_disabled_when_input_visible(self):
        """Test that link navigation is disabled when URL input is visible."""
        # Mock event
        mock_event = Mock()
        mock_event.key = "tab"

        # Set up state - URL input is visible
        self.app.url_input_visible = True
        self.app.prompt_editor_visible = False

        # Mock action methods
        self.app.action_next_link = Mock()

        await self.app.on_key(mock_event)

        # Should not navigate links
        self.app.action_next_link.assert_not_called()
        mock_event.prevent_default.assert_not_called()

    @pytest.mark.asyncio
    async def test_key_navigation_disabled_when_prompt_editor_visible(self):
        """Test that link navigation is disabled when prompt editor is visible."""
        # Mock event
        mock_event = Mock()
        mock_event.key = "tab"

        # Set up state - prompt editor is visible
        self.app.url_input_visible = False
        self.app.prompt_editor_visible = True

        # Mock action methods
        self.app.action_next_link = Mock()

        await self.app.on_key(mock_event)

        # Should not navigate links
        self.app.action_next_link.assert_not_called()
        mock_event.prevent_default.assert_not_called()

    @pytest.mark.asyncio
    async def test_arrow_keys_for_scrolling(self):
        """Test arrow keys for content scrolling."""
        # Test up arrow
        mock_event = Mock()
        mock_event.key = "up"

        self.app.url_input_visible = False
        self.app.prompt_editor_visible = False
        self.app.action_scroll_up = Mock()

        await self.app.on_key(mock_event)

        self.app.action_scroll_up.assert_called_once()
        mock_event.prevent_default.assert_called_once()

        # Test down arrow
        mock_event = Mock()
        mock_event.key = "down"

        self.app.action_scroll_down = Mock()

        await self.app.on_key(mock_event)

        self.app.action_scroll_down.assert_called_once()
        mock_event.prevent_default.assert_called_once()

    @pytest.mark.asyncio
    async def test_ctrl_s_saves_prompt(self):
        """Test Ctrl+S saves prompt when editor is visible."""
        # Mock event
        mock_event = Mock()
        mock_event.key = "ctrl+s"

        # Set up state
        self.app.prompt_editor_visible = True
        self.app.current_view = "summary"
        self.app.raw_html = "<html>Test</html>"

        # Mock UI elements
        mock_prompt_editor = Mock()
        mock_prompt_editor.text = "Updated prompt"
        self.app.query_one = Mock(return_value=mock_prompt_editor)

        # Mock views
        mock_view = Mock()
        self.app.views = {"summary": mock_view}

        # Mock methods
        self.app.notify = Mock()
        self.app.update_view_content = AsyncMock()
        self.app.action_hide_overlays = Mock()

        await self.app.on_key(mock_event)

        # Should save prompt
        assert mock_view.prompt == "Updated prompt"
        self.app.notify.assert_called_once()
        self.app.update_view_content.assert_called_once_with("summary")
        self.app.action_hide_overlays.assert_called_once()


class TestOverlayManagement:
    """Test overlay (URL input and prompt editor) management."""

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

    def test_show_url_input_when_hidden(self):
        """Test showing URL input when it's currently hidden."""
        # Set up initial state
        self.app.url_input_visible = False
        self.app.prompt_editor_visible = False
        self.app.current_url = "https://example.com"

        # Mock UI elements
        mock_url_input = Mock()
        self.app.query_one = Mock(return_value=mock_url_input)
        self.app.add_class = Mock()

        self.app.action_show_url_input()

        # Should show URL input
        assert self.app.url_input_visible
        self.app.add_class.assert_called_once_with("url-input-visible")
        mock_url_input.focus.assert_called_once()
        assert mock_url_input.value == "https://example.com"

    def test_show_url_input_when_prompt_editor_visible(self):
        """Test that URL input doesn't show when prompt editor is visible."""
        # Set up state with prompt editor visible
        self.app.url_input_visible = False
        self.app.prompt_editor_visible = True

        # Mock UI elements
        self.app.add_class = Mock()

        self.app.action_show_url_input()

        # Should not show URL input
        assert not self.app.url_input_visible
        self.app.add_class.assert_not_called()

    def test_hide_overlays_when_both_visible(self):
        """Test hiding overlays when both are visible."""
        # Set up state
        self.app.url_input_visible = True
        self.app.prompt_editor_visible = True
        self.app.current_view = "test"

        # Mock UI elements
        mock_content = Mock()
        self.app.query_one = Mock(return_value=mock_content)
        self.app.remove_class = Mock()

        self.app.action_hide_overlays()

        # Should hide both overlays
        assert not self.app.url_input_visible
        assert not self.app.prompt_editor_visible

        # Should remove both CSS classes
        self.app.remove_class.assert_any_call("url-input-visible")
        self.app.remove_class.assert_any_call("prompt-editor-visible")

        # Should focus content
        mock_content.focus.assert_called_once()

    def test_show_prompt_editor_for_llm_view(self):
        """Test showing prompt editor for LLM view."""
        # Set up state
        self.app.current_view = "summary"
        self.app.url_input_visible = False
        self.app.prompt_editor_visible = False

        # Mock views
        mock_view = Mock()
        mock_view.prompt = "Existing prompt"
        self.app.views = {"summary": mock_view}

        # Mock UI elements
        mock_prompt_editor = Mock()
        self.app.query_one = Mock(return_value=mock_prompt_editor)
        self.app.add_class = Mock()

        self.app.action_edit_prompt()

        # Should show prompt editor
        assert self.app.prompt_editor_visible
        self.app.add_class.assert_called_once_with("prompt-editor-visible")
        assert mock_prompt_editor.text == "Existing prompt"
        mock_prompt_editor.focus.assert_called_once()

    def test_show_prompt_editor_for_raw_view(self):
        """Test that prompt editor doesn't show for raw view."""
        # Set up state
        self.app.current_view = "raw"
        self.app.notify = Mock()

        self.app.action_edit_prompt()

        # Should show warning
        self.app.notify.assert_called_once_with(
            "Raw view doesn't use prompts", severity="warning"
        )

        # Should not show prompt editor
        assert not self.app.prompt_editor_visible

    def test_show_prompt_editor_invalid_view(self):
        """Test handling of invalid view for prompt editing."""
        # Set up state
        self.app.current_view = "nonexistent"
        self.app.views = {}  # Empty views
        self.app.notify = Mock()

        self.app.action_edit_prompt()

        # Should show error
        self.app.notify.assert_called_once_with(
            "Invalid view: nonexistent", severity="error"
        )


class TestScrollActions:
    """Test scroll action functionality."""

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
                self.app.current_view = "test"

    def test_action_scroll_up(self):
        """Test scroll up action."""
        # Mock content widget
        mock_content = Mock()
        self.app.query_one = Mock(return_value=mock_content)

        self.app.action_scroll_up()

        # Should call scroll up on content widget
        from spegel.main import HTMLContent

        self.app.query_one.assert_called_once_with("#content-test", HTMLContent)
        mock_content.action_scroll_up.assert_called_once()

    def test_action_scroll_down(self):
        """Test scroll down action."""
        # Mock content widget
        mock_content = Mock()
        self.app.query_one = Mock(return_value=mock_content)

        self.app.action_scroll_down()

        # Should call scroll down on content widget
        mock_content.action_scroll_down.assert_called_once()

    def test_scroll_action_exception_handling(self):
        """Test that scroll actions handle exceptions gracefully."""
        # Mock query_one to raise exception
        self.app.query_one = Mock(side_effect=Exception("Widget not found"))

        # Should not raise exception
        self.app.action_scroll_up()
        self.app.action_scroll_down()


class TestTabSwitching:
    """Test tab switching functionality."""

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
                    "raw": Mock(),
                    "summary": Mock(),
                }

    def test_action_switch_tab_valid_view(self):
        """Test switching to a valid tab."""
        # Mock TabbedContent
        mock_tabbed = Mock()
        self.app.query_one = Mock(return_value=mock_tabbed)

        self.app.action_switch_tab("summary")

        # Should update current view and activate tab
        assert self.app.current_view == "summary"
        assert mock_tabbed.active == "summary"

    def test_action_switch_tab_invalid_view(self):
        """Test switching to an invalid tab."""
        # Mock TabbedContent
        mock_tabbed = Mock()
        self.app.query_one = Mock(return_value=mock_tabbed)

        original_view = self.app.current_view

        self.app.action_switch_tab("nonexistent")

        # Should not update current view
        assert self.app.current_view == original_view
        # Should not set active tab
        assert not hasattr(mock_tabbed, "active") or mock_tabbed.active != "nonexistent"


class TestDynamicKeyBindings:
    """Test dynamic key binding setup."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("spegel.main.load_config") as mock_load_config:
            # Create mock views with hotkeys
            mock_view1 = Mock()
            mock_view1.id = "summary"
            mock_view1.name = "Summary"
            mock_view1.hotkey = "s"
            mock_view1.enabled = True

            mock_view2 = Mock()
            mock_view2.id = "analysis"
            mock_view2.name = "Analysis"
            mock_view2.hotkey = "a"
            mock_view2.enabled = True

            mock_config = Mock()
            mock_config.views = [mock_view1, mock_view2]
            mock_config.settings.default_view = "raw"
            mock_config.settings.app_title = "Test"
            mock_load_config.return_value = mock_config

            with patch("spegel.main.create_client") as mock_create_client:
                mock_create_client.return_value = None
                self.app = Spegel()

                # Set up views
                self.app.views = {
                    "summary": mock_view1,
                    "analysis": mock_view2,
                }

    def test_setup_bindings_creates_view_hotkeys(self):
        """Test that dynamic key bindings are created for views."""
        # Mock bind method
        self.app.bind = Mock()

        self.app._setup_bindings()

        # Should bind essential keys
        self.app.bind.assert_any_call(
            "slash", "show_url_input", description="Open URL", show=True
        )
        self.app.bind.assert_any_call(
            "escape", "hide_overlays", description="Cancel", show=False
        )
        self.app.bind.assert_any_call(
            "e", "edit_prompt", description="Edit Prompt", show=True
        )
        self.app.bind.assert_any_call("b", "go_back", description="Back", show=True)
        self.app.bind.assert_any_call("q", "quit", description="Quit", show=True)

        # Should bind view-specific hotkeys
        self.app.bind.assert_any_call(
            "s", "switch_tab('summary')", description="Summary", show=True
        )
        self.app.bind.assert_any_call(
            "a", "switch_tab('analysis')", description="Analysis", show=True
        )

    def test_setup_bindings_handles_duplicate_hotkeys(self):
        """Test handling of duplicate hotkey conflicts."""
        # Create view with conflicting hotkey
        mock_view = Mock()
        mock_view.id = "conflict"
        mock_view.name = "Conflict"
        mock_view.hotkey = "q"  # Conflicts with quit
        mock_view.enabled = True

        self.app.views["conflict"] = mock_view

        # Mock bind method to raise ValueError for duplicate
        def mock_bind(*args, **kwargs):
            if args[0] == "q" and "switch_tab" in args[1]:
                raise ValueError("Key already bound")

        self.app.bind = Mock(side_effect=mock_bind)
        self.app.notify = Mock()

        self.app._setup_bindings()

        # Should notify about conflict
        self.app.notify.assert_called_with(
            "Hotkey 'q' already bound; skipping Conflict", severity="warning", timeout=3
        )

    def test_setup_bindings_skips_disabled_views(self):
        """Test that disabled views don't get hotkey bindings."""
        # Create disabled view
        mock_view = Mock()
        mock_view.id = "disabled"
        mock_view.name = "Disabled"
        mock_view.hotkey = "d"
        mock_view.enabled = False

        self.app.views["disabled"] = mock_view

        # Mock bind method
        self.app.bind = Mock()

        self.app._setup_bindings()

        # Should not bind disabled view hotkey
        for call in self.app.bind.call_args_list:
            args, kwargs = call
            assert "switch_tab('disabled')" not in str(args)


class TestInternalLinkHandling:
    """Test internal link click handling."""

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

    def test_handle_internal_link_click_relative_url(self):
        """Test handling internal link clicks with relative URLs."""
        # Mock methods
        self.app._resolve_url = Mock(return_value="https://example.com/resolved")
        self.app.notify = Mock()
        self.app.fetch_and_display_url = AsyncMock()
        self.app.call_later = Mock()

        self.app.handle_internal_link_click("/relative-path")

        # Should resolve URL and navigate
        self.app._resolve_url.assert_called_once_with("/relative-path")
        self.app.notify.assert_called_once_with(
            "Navigating to: https://example.com/resolved"
        )
        self.app.call_later.assert_called_once()

    def test_handle_internal_link_click_mailto(self):
        """Test handling mailto links."""
        self.app.notify = Mock()
        self.app.call_later = Mock()

        self.app.handle_internal_link_click("mailto:test@example.com")

        # Due to current implementation, mailto gets resolved to https://mailto:...
        # Should show navigation notification
        self.app.notify.assert_called_once_with(
            "Navigating to: https://mailto:test@example.com"
        )
        self.app.call_later.assert_called_once()

    def test_handle_internal_link_click_javascript(self):
        """Test handling javascript links."""
        self.app.notify = Mock()
        self.app.call_later = Mock()

        self.app.handle_internal_link_click("javascript:alert('test')")

        # Due to current implementation, javascript gets resolved to https://javascript:...
        # Should show navigation notification
        self.app.notify.assert_called_once_with(
            "Navigating to: https://javascript:alert('test')"
        )
        self.app.call_later.assert_called_once()

    def test_handle_internal_link_click_absolute_url(self):
        """Test handling absolute URLs."""
        self.app._resolve_url = Mock(return_value="https://other.com/page")
        self.app.notify = Mock()
        self.app.call_later = Mock()

        self.app.handle_internal_link_click("https://other.com/page")

        # Should resolve and navigate
        self.app._resolve_url.assert_called_once_with("https://other.com/page")
        self.app.notify.assert_called_once_with("Navigating to: https://other.com/page")
        self.app.call_later.assert_called_once()


class TestUIStateManagement:
    """Test UI state management and consistency."""

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

    def test_overlay_visibility_consistency(self):
        """Test that overlay visibility flags are consistent."""
        # Initially both should be false
        assert not self.app.url_input_visible
        assert not self.app.prompt_editor_visible

        # Show URL input
        self.app.url_input_visible = True

        # Hiding overlays should reset both
        self.app.action_hide_overlays()

        assert not self.app.url_input_visible
        assert not self.app.prompt_editor_visible

    def test_current_view_tracking(self):
        """Test that current view tracking is consistent."""
        original_view = self.app.current_view

        # Switch views by mocking the UI components
        self.app.views = {"test": Mock()}

        # Mock the UI components needed for tab switching

        mock_tabbed_content = Mock()
        self.app.query_one = Mock(return_value=mock_tabbed_content)

        self.app.action_switch_tab("test")

        assert self.app.current_view == "test"
        assert self.app.current_view != original_view
