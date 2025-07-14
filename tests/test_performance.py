import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from spegel.main import LinkManager, ScrollManager, Spegel
from spegel.web import extract_clean_text, html_to_markdown


class TestPerformance:
    """Test performance characteristics of key components."""

    def test_large_html_processing_performance(self):
        """Test that large HTML documents are processed in reasonable time."""
        # Create large HTML document (1MB)
        large_content = "x" * (1024 * 1024)  # 1MB of content
        large_html = f"<html><head><title>Large Doc</title></head><body>{large_content}</body></html>"

        start_time = time.time()
        result = html_to_markdown(large_html)
        processing_time = time.time() - start_time

        # Should process in under 5 seconds
        assert processing_time < 5.0, (
            f"Processing took {processing_time}s, expected < 5s"
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_many_links_extraction_performance(self):
        """Test link extraction performance with many links."""
        # Create HTML with many links
        num_links = 1000
        links_html = ""
        for i in range(num_links):
            links_html += f'<a href="https://example{i}.com">Link {i}</a> '

        html_content = f"<html><body>{links_html}</body></html>"

        # Test link extraction performance
        mock_app = Mock()
        link_manager = LinkManager(mock_app)

        start_time = time.time()
        markdown = html_to_markdown(html_content)
        links = link_manager.extract_links_from_markdown(markdown)
        processing_time = time.time() - start_time

        # Should process in under 2 seconds
        assert processing_time < 2.0, (
            f"Link extraction took {processing_time}s, expected < 2s"
        )
        assert len(links) == num_links

    def test_rapid_scroll_updates_performance(self):
        """Test performance of rapid scroll position updates."""
        mock_app = Mock()
        scroll_manager = ScrollManager(mock_app)
        mock_content_widget = Mock()

        # Simulate rapid content updates (like streaming)
        num_updates = 100
        content_updates = [f"Content update {i}" for i in range(num_updates)]

        start_time = time.time()
        for content in content_updates:
            scroll_manager.update_content_preserve_scroll(mock_content_widget, content)
        processing_time = time.time() - start_time

        # Should handle rapid updates efficiently
        assert processing_time < 1.0, (
            f"Scroll updates took {processing_time}s, expected < 1s"
        )
        assert mock_content_widget.update.call_count == num_updates

    @pytest.mark.asyncio
    async def test_concurrent_view_processing_performance(self):
        """Test performance of concurrent view processing."""
        with patch("spegel.main.load_config") as mock_load_config:
            # Create multiple views
            mock_views = []
            for i in range(5):
                view = Mock()
                view.id = f"view{i}"
                view.auto_load = True
                view.enabled = True
                mock_views.append(view)

            mock_config = Mock()
            mock_config.views = mock_views
            mock_config.settings.default_view = "view0"
            mock_config.settings.app_title = "Test"
            mock_load_config.return_value = mock_config

            with patch("spegel.main.create_client") as mock_create_client:
                mock_create_client.return_value = None

                app = Spegel()
                app.views = {f"view{i}": mock_views[i] for i in range(5)}

                # Mock view processing to be instant
                async def mock_process_single_view(view_id):
                    await asyncio.sleep(0.1)  # Simulate some processing time

                app._process_single_view = mock_process_single_view

                start_time = time.time()
                await app._process_all_views_parallel()
                processing_time = time.time() - start_time

                # Should process views concurrently, not sequentially
                # If sequential: 5 * 0.1 = 0.5s, concurrent should be ~0.1s
                assert processing_time < 0.3, (
                    f"Concurrent processing took {processing_time}s, expected < 0.3s"
                )

    def test_memory_usage_with_large_content(self):
        """Test memory efficiency with large content."""
        # This is a basic test - in a real scenario you'd use memory profiling tools
        import gc

        # Force garbage collection
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Process large content multiple times
        for _ in range(10):
            large_html = "<html><body>" + ("x" * 10000) + "</body></html>"
            result = html_to_markdown(large_html)
            del result, large_html

        # Force garbage collection again
        gc.collect()
        final_objects = len(gc.get_objects())

        # Should not have significant memory leak
        # Allow some variance due to test framework overhead
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Too many objects retained: {object_growth}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_content_edge_cases(self):
        """Test handling of various empty content scenarios."""
        edge_cases = [
            "",  # Completely empty
            " ",  # Single space
            "\n",  # Single newline
            "\t",  # Single tab
            "   \n\t   ",  # Mixed whitespace
            "<html></html>",  # Empty HTML
            "<html><body></body></html>",  # Empty body
        ]

        for content in edge_cases:
            # Should not crash on any edge case
            result = html_to_markdown(content)
            assert isinstance(result, str)

    def test_unicode_and_emoji_handling(self):
        """Test handling of Unicode characters and emojis."""
        unicode_html = """
        <html><body>
            <h1>Unicode Test 🚀</h1>
            <p>Greek: αβγδε</p>
            <p>Chinese: 你好世界</p>
            <p>Arabic: مرحبا بالعالم</p>
            <p>Emojis: 😀🎉🔥💯</p>
            <p>Math: ∑∞≠≤≥∀∃</p>
        </body></html>
        """

        result = html_to_markdown(unicode_html)

        # Should preserve Unicode content
        assert "🚀" in result
        assert "你好世界" in result
        assert "مرحبا" in result
        assert "😀🎉" in result
        assert "∑∞≠" in result

    def test_malformed_markdown_links(self):
        """Test link extraction with malformed markdown."""
        mock_app = Mock()
        link_manager = LinkManager(mock_app)

        malformed_cases = [
            "[text]()",  # Empty URL
            "[](url)",  # Empty text
            "[text](url",  # Unclosed
            "text](url)",  # Missing opening
            "[nested [text]](url)",  # Nested brackets
            "[text](url with spaces)",  # URL with spaces
            "[text](<url>)",  # Angle brackets
            "[text](url) [text2](url2)",  # Multiple links
        ]

        for content in malformed_cases:
            # Should not crash on malformed markdown
            links = link_manager.extract_links_from_markdown(content)
            assert isinstance(links, list)

    def test_extremely_long_urls(self):
        """Test handling of very long URLs."""
        # Create very long URL (2KB)
        long_url = "https://example.com/" + "a" * 2000
        html_with_long_url = (
            f'<html><body><a href="{long_url}">Long URL</a></body></html>'
        )

        result = html_to_markdown(html_with_long_url)

        # Should handle long URLs without issues
        assert isinstance(result, str)
        assert "Long URL" in result

    def test_deeply_nested_html_structures(self):
        """Test handling of deeply nested HTML."""
        # Create deeply nested structure
        content = "Deep content"
        for _ in range(50):  # 50 levels deep
            content = f"<div><span><p>{content}</p></span></div>"

        deep_html = f"<html><body>{content}</body></html>"

        # Should handle deep nesting without stack overflow
        result = html_to_markdown(deep_html)
        assert "Deep content" in result

    def test_special_characters_in_attributes(self):
        """Test handling of special characters in HTML attributes."""
        special_html = """
        <html><body>
            <a href="https://example.com?param=value&other='quoted'&more=\"double\"">Link</a>
            <img src="image.png" alt="Image with & < > characters" />
            <div class="class-with-special-chars!@#$%">Content</div>
        </body></html>
        """

        result = html_to_markdown(special_html)

        # Should handle special characters gracefully
        assert isinstance(result, str)
        assert "Link" in result

    def test_mixed_content_types(self):
        """Test handling of mixed content types in HTML."""
        mixed_html = """
        <html><body>
            <h1>Title</h1>
            <p>Paragraph with <strong>bold</strong> and <em>italic</em></p>
            <ul>
                <li>List item 1</li>
                <li>List item 2</li>
            </ul>
            <table>
                <tr><td>Cell 1</td><td>Cell 2</td></tr>
            </table>
            <blockquote>Quote text</blockquote>
            <code>inline code</code>
            <pre>preformatted text</pre>
        </body></html>
        """

        result = html_to_markdown(mixed_html)

        # Should convert various HTML elements
        assert "# Title" in result or "Title" in result
        assert "bold" in result
        assert "italic" in result
        assert "List item" in result
        assert "Cell" in result


class TestConcurrency:
    """Test concurrent operations and thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_link_navigation(self):
        """Test concurrent link navigation operations."""
        mock_app = Mock()
        mock_app.current_view = "test"
        mock_app.original_content = {"test": "sample content"}  # Mock as dict
        mock_app.query_one = Mock(return_value=Mock())  # Mock UI components
        link_manager = LinkManager(mock_app)

        # Set up links
        link_manager.current_links = [
            (f"Link{i}", f"http://example{i}.com", i * 10, i * 10 + 10)
            for i in range(10)
        ]

        # Perform concurrent navigation operations
        tasks = []
        for _ in range(20):
            tasks.append(
                asyncio.create_task(asyncio.to_thread(link_manager.navigate_next_link))
            )
            tasks.append(
                asyncio.create_task(asyncio.to_thread(link_manager.navigate_prev_link))
            )

        # Should complete without errors
        await asyncio.gather(*tasks)

        # Final state should be valid
        assert 0 <= link_manager.current_link_index < len(link_manager.current_links)

    @pytest.mark.asyncio
    async def test_concurrent_content_updates(self):
        """Test concurrent content updates with scroll preservation."""
        mock_app = Mock()
        scroll_manager = ScrollManager(mock_app)
        mock_content_widget = Mock()

        # Perform concurrent content updates (synchronous function)
        def update_content(content):
            scroll_manager.update_content_preserve_scroll(mock_content_widget, content)

        tasks = [
            asyncio.create_task(asyncio.to_thread(update_content, f"Content {i}"))
            for i in range(50)
        ]

        # Should complete without errors
        await asyncio.gather(*tasks)

        # Should have processed all updates
        assert mock_content_widget.update.call_count == 50

    def test_resource_cleanup(self):
        """Test that resources are properly cleaned up."""
        # Create and destroy multiple instances
        for _ in range(10):
            with patch("spegel.main.load_config") as mock_load_config:
                mock_config = Mock()
                mock_config.views = []
                mock_config.settings.default_view = "raw"
                mock_config.settings.app_title = "Test"
                mock_load_config.return_value = mock_config

                with patch("spegel.main.create_client") as mock_create_client:
                    mock_create_client.return_value = None

                    app = Spegel()
                    # Create managers
                    link_manager = LinkManager(app)
                    scroll_manager = ScrollManager(app)

                    # Use them briefly
                    link_manager.extract_links_from_markdown("test content")
                    scroll_manager.update_content_preserve_scroll(Mock(), "test")

                    # Clean up references
                    del app, link_manager, scroll_manager

        # Test passes if no memory issues or exceptions occur


class TestStressTests:
    """Stress tests for extreme conditions."""

    def test_many_simultaneous_links(self):
        """Test handling of documents with many links."""
        # Create HTML with 500 links
        num_links = 500
        links_html = "".join(
            [
                f'<a href="https://example{i}.com">Link {i}</a> '
                for i in range(num_links)
            ]
        )

        html_content = f"<html><body>{links_html}</body></html>"
        markdown = html_to_markdown(html_content)

        mock_app = Mock()
        link_manager = LinkManager(mock_app)
        links = link_manager.extract_links_from_markdown(markdown)

        # Should handle many links efficiently
        assert len(links) == num_links

        # Test navigation through many links
        for _ in range(10):  # Navigate through some links
            link_manager.navigate_next_link()

    def test_rapid_text_truncation(self):
        """Test rapid text truncation operations."""
        # Generate large text
        large_text = "word " * 10000  # 50KB of text
        html_content = f"<html><body>{large_text}</body></html>"

        # Perform multiple truncations with different limits
        for limit in [100, 500, 1000, 5000]:
            result = extract_clean_text(html_content, max_chars=limit)
            assert len(result) <= limit + 100  # Allow for header and truncation marker

    def test_memory_pressure_simulation(self):
        """Test behavior under memory pressure."""
        # Simulate memory pressure by creating many large objects
        large_objects = []

        try:
            # Create some memory pressure
            for i in range(10):
                large_objects.append("x" * 100000)  # 100KB each

            # Perform normal operations under pressure
            html_content = "<html><body>" + ("content " * 1000) + "</body></html>"
            result = html_to_markdown(html_content)

            assert isinstance(result, str)
            assert len(result) > 0

        finally:
            # Clean up
            del large_objects
