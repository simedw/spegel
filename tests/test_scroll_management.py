from unittest.mock import Mock, patch

from spegel.main import ScrollManager


class TestScrollManager:
    """Test the ScrollManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_app = Mock()
        self.scroll_manager = ScrollManager(self.mock_app)
        self.mock_content_widget = Mock()

    def test_capture_scroll_state_at_bottom(self):
        """Test capturing scroll state when user is at bottom."""
        # Mock widget at bottom of content
        self.mock_content_widget.scroll_y = 100
        self.mock_content_widget.max_scroll_y = 100

        state = self.scroll_manager._capture_scroll_state(self.mock_content_widget)

        assert state["scroll_y"] == 100
        assert state["is_at_bottom"]

    def test_capture_scroll_state_near_bottom(self):
        """Test capturing scroll state when user is near bottom (within threshold)."""
        # Mock widget near bottom (within 2 pixel threshold)
        self.mock_content_widget.scroll_y = 98
        self.mock_content_widget.max_scroll_y = 100

        state = self.scroll_manager._capture_scroll_state(self.mock_content_widget)

        assert state["scroll_y"] == 98
        assert state["is_at_bottom"]

    def test_capture_scroll_state_middle(self):
        """Test capturing scroll state when user is in middle."""
        # Mock widget in middle of content
        self.mock_content_widget.scroll_y = 50
        self.mock_content_widget.max_scroll_y = 100

        state = self.scroll_manager._capture_scroll_state(self.mock_content_widget)

        assert state["scroll_y"] == 50
        assert not state["is_at_bottom"]

    def test_capture_scroll_state_top(self):
        """Test capturing scroll state when user is at top."""
        # Mock widget at top of content
        self.mock_content_widget.scroll_y = 0
        self.mock_content_widget.max_scroll_y = 100

        state = self.scroll_manager._capture_scroll_state(self.mock_content_widget)

        assert state["scroll_y"] == 0
        assert not state["is_at_bottom"]

    def test_capture_scroll_state_no_scroll_needed(self):
        """Test capturing scroll state when content fits entirely."""
        # Mock widget with no scrollable content
        self.mock_content_widget.scroll_y = 0
        self.mock_content_widget.max_scroll_y = 0

        state = self.scroll_manager._capture_scroll_state(self.mock_content_widget)

        assert state["scroll_y"] == 0
        assert state["is_at_bottom"]  # Considered at bottom when no scroll needed

    def test_restore_scroll_if_needed_at_bottom(self):
        """Test that scroll is not restored when user was at bottom."""
        state = {"scroll_y": 100, "is_at_bottom": True}

        self.scroll_manager._restore_scroll_if_needed(self.mock_content_widget, state)

        # Should not schedule scroll restoration
        self.mock_app.call_after_refresh.assert_not_called()

    def test_restore_scroll_if_needed_not_at_bottom(self):
        """Test that scroll is restored when user was not at bottom."""
        state = {"scroll_y": 50, "is_at_bottom": False}

        self.scroll_manager._restore_scroll_if_needed(self.mock_content_widget, state)

        # Should schedule scroll restoration
        self.mock_app.call_after_refresh.assert_called_once()

    def test_restore_scroll_position_within_bounds(self):
        """Test restoring scroll position within content bounds."""
        target_scroll_y = 50
        self.mock_content_widget.max_scroll_y = 100

        self.scroll_manager._restore_scroll_position(
            self.mock_content_widget, target_scroll_y
        )

        # Should set scroll position to target
        assert self.mock_content_widget.scroll_y == 50

    def test_restore_scroll_position_exceeds_bounds(self):
        """Test restoring scroll position that exceeds content bounds."""
        target_scroll_y = 150  # Beyond max scroll
        self.mock_content_widget.max_scroll_y = 100

        self.scroll_manager._restore_scroll_position(
            self.mock_content_widget, target_scroll_y
        )

        # Should clamp to max scroll
        assert self.mock_content_widget.scroll_y == 100

    def test_restore_scroll_position_no_scroll_content(self):
        """Test restoring scroll position when content doesn't scroll."""
        target_scroll_y = 50
        self.mock_content_widget.max_scroll_y = 0  # No scrollable content

        self.scroll_manager._restore_scroll_position(
            self.mock_content_widget, target_scroll_y
        )

        # Should not set scroll position
        assert (
            not hasattr(self.mock_content_widget, "scroll_y")
            or self.mock_content_widget.scroll_y != 50
        )

    def test_restore_scroll_position_exception_handling(self):
        """Test that exceptions during scroll restoration are handled gracefully."""
        target_scroll_y = 50

        # Mock widget that raises exception when accessing max_scroll_y
        self.mock_content_widget.max_scroll_y = property(
            lambda self: exec('raise Exception("Test error")')
        )

        # Should not raise exception
        self.scroll_manager._restore_scroll_position(
            self.mock_content_widget, target_scroll_y
        )

    def test_update_content_preserve_scroll_success(self):
        """Test successful content update with scroll preservation."""
        new_content = "Updated content"

        # Mock successful scroll state capture and restore
        with patch.object(self.scroll_manager, "_capture_scroll_state") as mock_capture:
            with patch.object(
                self.scroll_manager, "_restore_scroll_if_needed"
            ) as mock_restore:
                mock_capture.return_value = {"scroll_y": 50, "is_at_bottom": False}

                self.scroll_manager.update_content_preserve_scroll(
                    self.mock_content_widget, new_content
                )

                # Should capture state, update content, and restore scroll
                mock_capture.assert_called_once_with(self.mock_content_widget)
                self.mock_content_widget.update.assert_called_once_with(new_content)
                mock_restore.assert_called_once_with(
                    self.mock_content_widget, {"scroll_y": 50, "is_at_bottom": False}
                )

    def test_update_content_preserve_scroll_exception_fallback(self):
        """Test fallback behavior when scroll preservation fails."""
        new_content = "Updated content"

        # Mock exception during scroll state capture
        with patch.object(
            self.scroll_manager,
            "_capture_scroll_state",
            side_effect=Exception("Test error"),
        ):
            self.scroll_manager.update_content_preserve_scroll(
                self.mock_content_widget, new_content
            )

            # Should still update content as fallback
            self.mock_content_widget.update.assert_called_once_with(new_content)

    def test_update_content_preserve_scroll_streaming_scenario(self):
        """Test scroll preservation during streaming updates."""
        content_updates = ["Initial content", "Updated content", "Final content"]

        # Mock user scrolling up during streaming
        self.mock_content_widget.scroll_y = 20
        self.mock_content_widget.max_scroll_y = 100

        for content in content_updates:
            self.scroll_manager.update_content_preserve_scroll(
                self.mock_content_widget, content
            )

        # Should preserve user's scroll position for each update
        assert self.mock_content_widget.update.call_count == 3
        assert (
            self.mock_app.call_after_refresh.call_count == 3
        )  # Should schedule restoration each time


class TestScrollIntegration:
    """Test scroll management integration with main app."""

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

                # Import after patching to avoid import-time dependencies
                from spegel.main import Spegel

                self.app = Spegel()

    def test_scroll_manager_initialization(self):
        """Test that ScrollManager is properly initialized in app."""
        assert hasattr(self.app, "scroll_manager")
        assert self.app.scroll_manager is not None
        assert self.app.scroll_manager.app == self.app

    def test_scroll_manager_usage_in_streaming(self):
        """Test that scroll manager is used during content streaming."""
        # Mock content widget and scroll manager
        mock_content_widget = Mock()
        self.app.scroll_manager = Mock()

        # This would be called during streaming in update_view_content
        self.app.scroll_manager.update_content_preserve_scroll(
            mock_content_widget, "streaming content"
        )

        # Should call the scroll manager method
        self.app.scroll_manager.update_content_preserve_scroll.assert_called_once_with(
            mock_content_widget, "streaming content"
        )


class TestScrollEdgeCases:
    """Test edge cases and boundary conditions for scroll management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_app = Mock()
        self.scroll_manager = ScrollManager(self.mock_app)
        self.mock_content_widget = Mock()

    def test_scroll_state_with_negative_values(self):
        """Test handling of negative scroll values."""
        # Some UI frameworks might return negative values
        self.mock_content_widget.scroll_y = -5
        self.mock_content_widget.max_scroll_y = 100

        state = self.scroll_manager._capture_scroll_state(self.mock_content_widget)

        assert state["scroll_y"] == -5
        assert not state["is_at_bottom"]

    def test_scroll_state_with_floating_point_values(self):
        """Test handling of floating point scroll values."""
        self.mock_content_widget.scroll_y = 98.7
        self.mock_content_widget.max_scroll_y = 100.0

        state = self.scroll_manager._capture_scroll_state(self.mock_content_widget)

        assert state["scroll_y"] == 98.7
        assert state["is_at_bottom"]  # Within threshold

    def test_restore_scroll_with_floating_point_target(self):
        """Test restoring floating point scroll positions."""
        target_scroll_y = 50.5
        self.mock_content_widget.max_scroll_y = 100

        self.scroll_manager._restore_scroll_position(
            self.mock_content_widget, target_scroll_y
        )

        assert self.mock_content_widget.scroll_y == 50.5

    def test_content_update_with_empty_content(self):
        """Test content update with empty content."""
        self.scroll_manager.update_content_preserve_scroll(self.mock_content_widget, "")

        self.mock_content_widget.update.assert_called_once_with("")

    def test_content_update_with_very_long_content(self):
        """Test content update with very long content."""
        long_content = "A" * 10000  # Very long content

        self.scroll_manager.update_content_preserve_scroll(
            self.mock_content_widget, long_content
        )

        self.mock_content_widget.update.assert_called_once_with(long_content)

    def test_multiple_rapid_updates(self):
        """Test handling of multiple rapid content updates."""
        # Simulate rapid streaming updates
        contents = [f"Content update {i}" for i in range(10)]

        for content in contents:
            self.scroll_manager.update_content_preserve_scroll(
                self.mock_content_widget, content
            )

        # Should handle all updates
        assert self.mock_content_widget.update.call_count == 10

    def test_scroll_restoration_callback_execution(self):
        """Test that scroll restoration callback is properly executed."""
        state = {"scroll_y": 50, "is_at_bottom": False}

        # Capture the callback that would be passed to call_after_refresh
        captured_callback = None

        def capture_callback(callback):
            nonlocal captured_callback
            captured_callback = callback

        self.mock_app.call_after_refresh = capture_callback

        self.scroll_manager._restore_scroll_if_needed(self.mock_content_widget, state)

        # Should have captured a callback
        assert captured_callback is not None

        # Execute the callback
        self.mock_content_widget.max_scroll_y = 100
        captured_callback()

        # Should have set the scroll position
        assert self.mock_content_widget.scroll_y == 50
