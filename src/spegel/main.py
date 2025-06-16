#!/usr/bin/env python3
"""
Spegel - Reflect the web through AI
"""

import asyncio
import os
from typing import Optional, Dict, List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import re
from textual import on

# External modules
from .config import load_config, View
from .llm import get_default_client, LLMClient

from textual.app import App, ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.widgets import (
    Footer,
    Header,
    Input,
    Markdown,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.binding import Binding

from .web import fetch_url as fetch_url_blocking, extract_clean_text, html_to_markdown
from .views import stream_view

# Load environment variables
load_dotenv()


class HTMLContent(Markdown):
    """Widget to display parsed HTML content as Markdown."""

    def __init__(self, content: str = "", **kwargs):
        super().__init__(content, **kwargs)
        self.can_focus = True

    def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Handle link clicks to navigate within the browser instead of opening externally."""
        # Get the main app instance
        app = self.app
        if hasattr(app, "handle_internal_link_click"):
            # Prevent default behavior (opening in external browser)
            event.prevent_default()
            # Handle the link click within our browser
            app.handle_internal_link_click(event.href)
        else:
            # Fallback to default behavior if method not available
            super().on_markdown_link_clicked(event)


class URLInput(Input):
    """Custom input widget for URLs."""

    def __init__(self, **kwargs):
        super().__init__(placeholder="Enter URL (e.g., https://example.com)", **kwargs)


class PromptEditor(TextArea):
    """Text area for editing prompts."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Spegel(App):
    """A terminal-based browser with LLM capabilities."""
    
    CSS = """
    #url-input {
        dock: bottom;
        height: 3;
        border: solid $primary;
        display: none;
    }
    
    #prompt-editor-container {
        dock: bottom;
        height: 10;
        border: solid $warning;
        display: none;
    }
    
    #content-container {
        height: 1fr;
    }
    
    .url-input-visible #url-input {
        display: block;
    }
    
    .url-input-visible #content-container {
        margin-bottom: 3;
    }
    
    .prompt-editor-visible #prompt-editor-container {
        display: block;
    }
    
    .prompt-editor-visible #content-container {
        margin-bottom: 10;
    }
    
    TabbedContent {
        height: 1fr;
    }
    
    TabPane {
        padding: 1;
        height: 1fr;
    }
    
    HTMLContent {
        height: 1fr;
        scrollbar-gutter: stable;
    }
    """

    def __init__(self, initial_url: str | None = None, **kwargs):
        # Load configuration first (before super().__init__)
        self.config = load_config()

        # Load views from config (mapping view_id -> View)
        self.views: Dict[str, View] = {v.id: v for v in self.config.views if v.enabled}
        self.current_view = self.config.settings.default_view

        super().__init__(**kwargs)

        # Set app title from config
        self.title = self.config.settings.app_title

        self.current_url: Optional[str] = None
        # URL provided via CLI to open on startup
        self._startup_url: Optional[str] = initial_url
        self.raw_html: str = ""
        self.url_input_visible = False
        self.prompt_editor_visible = False
        self.views_loaded: set = set()  # Track which views have been processed
        self.views_loading: set = set()  # Track which views are currently loading
        self.current_links: List[tuple] = []  # List of (link_text, link_url) tuples
        self.current_link_index: int = -1  # Currently selected link index
        self.original_content: Dict[
            str, str
        ] = {}  # Store original content for each view
        self.url_history: List[str] = []  # History of visited URLs for back navigation

        # Initialize LLM client via abstraction layer
        self.llm_client, self.llm_available = get_default_client()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(show_clock=True)

        with Container(id="content-container"):
            # Use the configured default view instead of hardcoding "raw"
            with TabbedContent(initial=self.current_view):
                # Create tabs in the order specified in config
                sorted_views = sorted(self.views.items(), key=lambda x: x[1].order)
                
                for view_id, view_config in sorted_views:
                    with TabPane(view_config.name, id=view_id):
                        # Remove ScrollableContainer to fix double scrollbar issue
                        if view_id == "raw":
                            yield HTMLContent(
                                "# Welcome to Spegel!\n\nPress **'/'** to enter a URL and start browsing.",
                                id=f"content-{view_id}",
                            )
                        else:
                            yield HTMLContent(
                                f"## No content loaded yet\n\nEnter a URL to see **{view_config.description}**.",
                                id=f"content-{view_id}",
                            )

        # URL input (hidden by default)
        yield URLInput(id="url-input")

        # Prompt editor (hidden by default)
        with Container(id="prompt-editor-container"):
            yield Static(
                "Edit Prompt (Ctrl+S to save, Escape to cancel):",
                id="prompt-editor-label",
            )
            yield PromptEditor(id="prompt-editor")

        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        # Set up dynamic key bindings now that the app is ready
        self._setup_bindings()

        # If a startup URL was provided, kick off fetching immediately
        if self._startup_url:
            self.run_async_task(self.fetch_and_display_url(self._startup_url))

    def action_show_url_input(self) -> None:
        """Show the URL input field."""
        if not self.url_input_visible and not self.prompt_editor_visible:
            self.url_input_visible = True
            self.add_class("url-input-visible")
            url_input = self.query_one("#url-input", URLInput)
            url_input.focus()
            if self.current_url:
                url_input.value = self.current_url

    def action_hide_overlays(self) -> None:
        """Hide URL input and prompt editor."""
        if self.url_input_visible:
            self.url_input_visible = False
            self.remove_class("url-input-visible")

        if self.prompt_editor_visible:
            self.prompt_editor_visible = False
            self.remove_class("prompt-editor-visible")

        # Focus the current tab content
        try:
            content_widget = self.query_one(
                f"#content-{self.current_view}", HTMLContent
            )
            content_widget.focus()
        except Exception:
            pass

    def action_edit_prompt(self) -> None:
        """Show prompt editor for current view."""
        if self.current_view == "raw":
            self.notify("Raw view doesn't use prompts", severity="warning")
            return

        if self.current_view not in self.views:
            self.notify(f"Invalid view: {self.current_view}", severity="error")
            return

        if not self.prompt_editor_visible and not self.url_input_visible:
            self.prompt_editor_visible = True
            self.add_class("prompt-editor-visible")

            # Load current prompt
            current_prompt = self.views[self.current_view].prompt
            prompt_editor = self.query_one("#prompt-editor", PromptEditor)
            prompt_editor.text = current_prompt
            prompt_editor.focus()

    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to a specific tab."""
        if tab_id in self.views:
            self.current_view = tab_id
            tabbed_content = self.query_one(TabbedContent)
            tabbed_content.active = tab_id

    def action_go_back(self) -> None:
        """Navigate back to the previous URL in history."""
        if len(self.url_history) < 2:
            self.notify("No previous page to go back to", severity="warning", timeout=2)
            return

        # Remove current URL from history
        self.url_history.pop()

        # Get the previous URL
        previous_url = (
            self.url_history.pop()
        )  # Remove it so it doesn't duplicate when we navigate

        self.notify(f"Going back to: {previous_url}")
        # Navigate to the previous URL
        self.run_async_task(self.fetch_and_display_url(previous_url))

    @on(Input.Submitted, "#url-input")
    async def handle_url_submission(self, event: Input.Submitted) -> None:
        """Handle URL submission and fetch content."""
        url = event.value.strip()
        if not url:
            return

        # Add https:// if no protocol specified
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        # Hide input immediately to show we're processing
        self.action_hide_overlays()

        # Fetch and process content
        await self.fetch_and_display_url(url)

    @on(TextArea.Changed, "#prompt-editor")
    def handle_prompt_change(self, event: TextArea.Changed) -> None:
        """Handle prompt editor changes."""
        # Save on Ctrl+S (we'll add this binding)
        pass

    async def on_key(self, event) -> None:
        """Handle key events."""
        # Handle Tab and Shift+Tab for link navigation (override default tab behavior)
        if (
            event.key == "tab"
            and not self.url_input_visible
            and not self.prompt_editor_visible
        ):
            if self.current_links:
                self.action_next_link()
                event.prevent_default()
                return

        if (
            event.key == "shift+tab"
            and not self.url_input_visible
            and not self.prompt_editor_visible
        ):
            if self.current_links:
                self.action_prev_link()
                event.prevent_default()
                return

        # Handle Enter for opening links
        if (
            event.key == "enter"
            and not self.url_input_visible
            and not self.prompt_editor_visible
        ):
            if (
                self.current_links
                and self.current_link_index >= 0
                and self.current_link_index < len(self.current_links)
            ):
                await self.action_open_link()
                event.prevent_default()
                return

        # Handle Ctrl+S for prompt editor
        if self.prompt_editor_visible and event.key == "ctrl+s":
            # Save the prompt
            prompt_editor = self.query_one("#prompt-editor", PromptEditor)
            new_prompt = prompt_editor.text
            self.views[self.current_view].prompt = new_prompt
            self.notify(f"Prompt saved for {self.views[self.current_view].name}")

            # Refresh current view if we have content
            if self.raw_html:
                await self.update_view_content(self.current_view)

            self.action_hide_overlays()

    @on(TabbedContent.TabActivated)
    async def handle_tab_change(self, event: TabbedContent.TabActivated) -> None:
        """Handle tab changes."""
        # Extract the actual tab name from the event
        # The event.tab.id might be something like "--content-tab-actions", so we need to extract the real ID
        raw_tab_id = str(event.tab.id)
        
        # Handle the case where Textual adds prefixes to tab IDs
        if raw_tab_id.startswith("--content-tab-"):
            tab_name = raw_tab_id.replace("--content-tab-", "")
        else:
            tab_name = raw_tab_id
        
        if tab_name in self.views:
            self.current_view = tab_name

            # Check if this view needs to be loaded on-demand
            view_config = self.views[tab_name]
            
            needs_loading = (
                self.raw_html and  # We have content to process
                tab_name not in self.views_loaded and  # Not already loaded
                tab_name not in self.views_loading and  # Not currently loading
                not view_config.auto_load  # Not an auto-load view
            )
            
            if needs_loading:
                # Start loading this view on-demand
                self.views_loading.add(tab_name)
                self._update_tab_name(tab_name)
                asyncio.create_task(self._process_single_view(tab_name))

            # Update link selection for this view
            if self.current_view in self.original_content:
                # Re-extract links with position information from stored content
                self.current_links = self._extract_links_from_markdown(
                    self.original_content[self.current_view]
                )
                self.current_link_index = -1  # Reset selection
                # Show brief notification about available links in this view
                if self.current_links:
                    self.notify(
                        f"{len(self.current_links)} links available in {self.views[self.current_view].name}",
                        timeout=2,
                    )
            else:
                # No content yet, clear links
                self.current_links = []
                self.current_link_index = -1
        else:
            self.notify(f"Unknown tab: {tab_name} (available: {list(self.views.keys())})", timeout=3)

    async def fetch_and_display_url(self, url: str) -> None:
        """Fetch URL content and display it in all views."""
        # Add URL to history if it's not already the current URL
        if not self.url_history or self.url_history[-1] != url:
            self.url_history.append(url)
            # Keep history size reasonable (last 50 URLs)
            if len(self.url_history) > 50:
                self.url_history = self.url_history[-50:]

        # Show loading in current view only
        current_content_widget = self.query_one(
            f"#content-{self.current_view}", HTMLContent
        )
        current_content_widget.update(f"Loading {url}...")

        # Reset view states
        self.views_loaded = set()
        self.views_loading = set()

        # Reset tab names
        self._reset_tab_names()

        try:
            # Fetch content
            html_text = await asyncio.get_event_loop().run_in_executor(
                None, fetch_url_blocking, url
            )

            if html_text:
                self.current_url = url
                self.title = f"LLM Browser - {url}"
                self.raw_html = html_text

                # Start processing all views in parallel
                await self._process_all_views_parallel()
            else:
                current_content_widget.update(f"Failed to load {url}")

        except Exception as e:
            current_content_widget.update(f"Error loading {url}: {str(e)}")

    async def _process_all_views_parallel(self) -> None:
        """Process views in parallel, respecting auto_load settings."""
        # Only process auto-load views initially
        auto_load_views = [view_id for view_id, view in self.views.items() if view.auto_load]
        
        # Mark auto-load views as loading first
        for view_id in auto_load_views:
            self.views_loading.add(view_id)
            self._update_tab_name(view_id)
        
        # Start auto-load tasks in the background without waiting
        for view_id in auto_load_views:
            # Create task and let it run in background
            asyncio.create_task(self._process_single_view(view_id))
        
        # Return immediately - don't wait for tasks to complete

    async def _process_single_view(self, view_id: str) -> None:
        """Process a single view and update its tab name."""
        try:
            # Set immediate loading message in content
            content_widget = self.query_one(f"#content-{view_id}", HTMLContent)
            if view_id == "raw":
                content_widget.update(
                    "## Loading content...\n\n*Please wait while the page is fetched and parsed.*"
                )
            else:
                if self.llm_available:
                    content_widget.update(
                        f"## ⏳ Preparing AI Analysis\n\n**{self.views[view_id].name}** - Getting ready to stream...\n\n*AI response will appear here in real-time.*"
                    )
                else:
                    content_widget.update(
                        f"## LLM not available\n\nAdd `GEMINI_API_KEY` to .env file\n\n**View:** {self.views[view_id].description}"
                    )

            # Yield control to allow UI updates
            await asyncio.sleep(0.1)

            # Process the view (this is the potentially slow part)
            await self.update_view_content(view_id)

            # Mark as loaded and update tab name
            self.views_loading.discard(view_id)
            self.views_loaded.add(view_id)
            self._update_tab_name(view_id)

        except Exception as e:
            # Handle errors
            self.views_loading.discard(view_id)
            self.views_loaded.add(view_id)
            self._update_tab_name(view_id)

            content_widget = self.query_one(f"#content-{view_id}", HTMLContent)
            content_widget.update(
                f"## ❌ Error\n\n**Failed to process {view_id} view:**\n\n```\n{str(e)}\n```"
            )

    def _update_tab_name(self, view_id: str) -> None:
        """Update tab name with loading/loaded indicators."""
        base_name = self.views[view_id].name
        if view_id in self.views_loading:
            display_name = f"⏳ {base_name}"
        elif view_id in self.views_loaded:
            display_name = f"✓ {base_name}"
        else:
            display_name = base_name

        def update_label() -> None:
            try:
                tabbed = self.query_one(TabbedContent)
                tab = tabbed.get_tab(view_id)
                if tab:
                    tab.label = display_name
                    tab.refresh()
            except Exception:
                # Fallback: try TabPane
                try:
                    pane = self.query_one(f"#{view_id}", TabPane)
                    pane.label = display_name
                    pane.refresh()
                except Exception:
                    pass

        # schedule the label update on main thread
        self.call_later(update_label)

    def _reset_tab_names(self) -> None:
        """Reset all tab names to their base names."""
        for view_id in self.views.keys():
            try:
                tab_pane = self.query_one(f"#{view_id}", TabPane)
                tab_pane.label = self.views[view_id].name
            except Exception:
                pass  # Ignore if tab not found

    async def update_view_content(self, view_id: str) -> None:
        """Update content for a specific view."""
        if not self.raw_html:
            return

        content_widget = self.query_one(f"#content-{view_id}", HTMLContent)

        if view_id == "raw":
            # Raw view - just parse HTML normally
            formatted_content = html_to_markdown(self.raw_html, self.current_url)

            # Store original content and extract links
            self.original_content[view_id] = formatted_content
            links = self._extract_links_from_markdown(formatted_content)

            # Only update current links if we're currently viewing this tab
            if self.current_view == view_id:
                self.current_links = links
                self.current_link_index = -1  # Reset link selection

            content_widget.update(formatted_content)

            # Show link count only if we're currently viewing this tab
            if self.current_view == view_id:
                if self.current_links:
                    self.notify(
                        f"Found {len(self.current_links)} links. Use Tab/Shift+Tab to navigate, Enter to open.",
                        timeout=5,
                    )
                else:
                    self.notify("No navigable links found on this page.", timeout=3)
        else:
            # Other views – use the central processor
            header_lines = [f"## ✨ {self.views[view_id].name}", "", "*Streaming response...*", ""]
            content_widget.update("\n".join(header_lines))

            parts = [f"## ✨ {self.views[view_id].name}", ""]

            async for chunk in stream_view(
                self.views[view_id],
                self.raw_html,
                self.llm_client,
                self.current_url,
            ):
                parts.append(chunk)
                content_widget.update("\n".join(parts))

            generated = "\n".join(parts)
            self.original_content[view_id] = generated
            links = self._extract_links_from_markdown(generated)

            if self.current_view == view_id:
                self.current_links = links
                self.current_link_index = -1
                if links:
                    self.notify(
                        f"Found {len(links)} links in {self.views[view_id].name}. Use Tab/Shift+Tab to navigate.",
                        timeout=3,
                    )

    def _extract_table_content(self, table) -> List[str]:
        """Extract content from HTML table (useful for sites like HN)."""
        content = []

        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if cells:
                row_content = []
                for cell in cells:
                    # Extract links with their text
                    links = cell.find_all("a")
                    if links:
                        for link in links:
                            text = link.get_text().strip()
                            href = link.get("href", "")
                            if text and len(text) > 3:
                                if href and not href.startswith("#"):
                                    row_content.append(f"[{text}]({href})")
                                else:
                                    row_content.append(text)
                    else:
                        # Regular cell text
                        cell_text = cell.get_text().strip()
                        if cell_text and len(cell_text) > 2:
                            row_content.append(cell_text)

                if row_content:
                    # Join cell content with separators
                    combined = " | ".join(row_content)
                    if len(combined) > 10:
                        content.append(combined)

        return content

    def _extract_text_content(self, element) -> List[str]:
        """Extract text content from element (legacy method - now simplified)."""
        return [element.get_text().strip()] if element else []

    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + len(current_line) <= width:
                current_line.append(word)
                current_length += len(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def _extract_links_from_markdown(self, content: str) -> List[tuple]:
        """Extract all links from markdown content with position tracking."""
        # Regex to match markdown links: [text](url) - including angle brackets from html2text
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"

        # Find all matches with their positions
        clean_links = []
        for match in re.finditer(link_pattern, content):
            text = match.group(1).strip()
            url = match.group(2).strip()
            start_pos = match.start()
            end_pos = match.end()

            # Remove angle brackets that html2text adds
            if url.startswith("<") and url.endswith(">"):
                url = url[1:-1]

            # Skip empty URLs or URLs that are just fragments
            if not url or url.startswith("#"):
                continue

            # Skip voting links (but keep other UI links like comments, user profiles, etc.)
            if "vote?" in url:
                continue

            # Skip "from?site=" links (these are just site indicators)
            if url.startswith("from?site="):
                continue

            # Store link with position info
            clean_links.append((text, url, start_pos, end_pos))

        return clean_links

    def _escape_markup(self, text: str) -> str:
        """Escape markup characters for safe display in notifications."""
        return text.replace("[", "\\[").replace("]", "\\]").replace("!", "\\!")

    def _highlight_current_link(self, content: str) -> str:
        """Add highlighting to the currently selected link using position-based replacement."""
        if self.current_link_index < 0 or self.current_link_index >= len(
            self.current_links
        ):
            return content

        # Get the current link with position
        link_text, link_url, start_pos, end_pos = self.current_links[
            self.current_link_index
        ]

        # Create highlighted version
        highlighted_link = f"**→ [{link_text}]({link_url}) ←**"

        # Replace the content at the specific position
        # Split content into: before_link + highlighted_link + after_link
        before = content[:start_pos]
        after = content[end_pos:]

        return before + highlighted_link + after

    def action_next_link(self) -> None:
        """Navigate to the next link."""
        if not self.current_links:
            return

        self.current_link_index = (self.current_link_index + 1) % len(
            self.current_links
        )
        self._update_current_view_with_highlight()

    def action_prev_link(self) -> None:
        """Navigate to the previous link."""
        if not self.current_links:
            return

        self.current_link_index = (self.current_link_index - 1) % len(
            self.current_links
        )
        self._update_current_view_with_highlight()

    async def action_open_link(self) -> None:
        """Open the currently selected link."""
        if self.current_link_index < 0 or self.current_link_index >= len(
            self.current_links
        ):
            self.notify("No link selected", severity="warning")
            return

        link_text, link_url, _, _ = self.current_links[self.current_link_index]

        # Handle URL resolution
        from urllib.parse import urljoin

        if not link_url.startswith(("http://", "https://")):
            if self.current_url:
                # For all relative URLs (including root-relative /path), resolve against current URL
                link_url = urljoin(self.current_url, link_url)
            else:
                # If no current URL context, assume https
                if link_url.startswith("/"):
                    link_url = f"https://example.com{link_url}"  # Fallback
                else:
                    link_url = f"https://{link_url}"

        self.notify(f"Opening: {self._escape_markup(link_text)}")
        await self.fetch_and_display_url(link_url)

    def _update_current_view_with_highlight(self) -> None:
        """Update the current view content with link highlighting."""
        if self.current_view not in self.original_content:
            return

        original = self.original_content[self.current_view]
        highlighted = self._highlight_current_link(original)

        try:
            content_widget = self.query_one(
                f"#content-{self.current_view}", HTMLContent
            )
            content_widget.update(highlighted)
        except Exception:
            pass

    def handle_internal_link_click(self, href: str) -> None:
        """Handle link clicks from markdown content to navigate within the browser."""
        # Resolve the URL using the same logic as keyboard navigation
        from urllib.parse import urljoin

        # Handle URL resolution
        if not href.startswith(("http://", "https://")):
            if self.current_url:
                # For all relative URLs (including root-relative /path), resolve against current URL
                href = urljoin(self.current_url, href)
            else:
                # If no current URL context, assume https
                if href.startswith("/"):
                    href = f"https://example.com{href}"  # Fallback
                else:
                    href = f"https://{href}"

        # Special handling for mailto links
        if href.startswith("mailto:"):
            self.notify(f"Email link: {href}", timeout=3)
            return

        # Special handling for javascript links (ignore them)
        if href.startswith("javascript:"):
            self.notify(
                "JavaScript links are not supported", severity="warning", timeout=2
            )
            return

        # Navigate to the URL within our browser
        self.notify(f"Navigating to: {href}")
        # Use call_later to avoid potential event loop issues
        self.call_later(lambda: self.run_async_task(self.fetch_and_display_url(href)))

    def run_async_task(self, task):
        """Helper to run async tasks from sync context."""
        import asyncio

        if hasattr(asyncio, "create_task"):
            asyncio.create_task(task)
        else:
            # Fallback for older Python versions
            asyncio.ensure_future(task)

    # ---------------------------------
    # Dynamic Key Binding Setup
    # ---------------------------------

    def _setup_bindings(self) -> None:
        """Register dynamic key bindings based on configuration."""
        # Essential app bindings (use keyword args for description/show)
        self.bind("slash", "show_url_input", description="Open URL", show=True)
        self.bind("escape", "hide_overlays", description="Cancel", show=False)
        self.bind("e", "edit_prompt", description="Edit Prompt", show=True)
        self.bind("b", "go_back", description="Back", show=True)
        self.bind("q", "quit", description="Quit", show=True)

        # View-specific bindings from configuration
        for view_id, view_config in self.views.items():
            if view_config.enabled and view_config.hotkey:
                try:
                    self.bind(
                        view_config.hotkey,
                        f"switch_tab('{view_id}')",
                        description=view_config.name,
                        show=True,
                    )
                except ValueError:
                    # Key already bound; warn and skip
                    self.notify(
                        f"Hotkey '{view_config.hotkey}' already bound; skipping {view_config.name}",
                        severity="warning",
                        timeout=3,
                    )


def main() -> None:
    """CLI entry point for the *spegel* command.

    Usage::

        spegel                # opens browser with welcome screen
        spegel https://news.ycombinator.com  # auto-loads URL on launch
    """

    import argparse, sys

    parser = argparse.ArgumentParser(prog="spegel", description="Spegel – Reflect the web through AI (terminal browser)")
    parser.add_argument("url", nargs="?", help="URL to open immediately on launch")

    args = parser.parse_args(sys.argv[1:])

    initial_url = args.url
    if initial_url and not initial_url.startswith(("http://", "https://")):
        # Auto-prepend https if scheme is missing
        initial_url = f"https://{initial_url}"

    app = Spegel(initial_url=initial_url)
    app.run()


if __name__ == "__main__":
    main()
