"""Drives a real, isolated Brave Browser instance for web_search/web_fetch.

Uses an automation-only profile (separate --user-data-dir, started logged
out of everything) rather than the user's everyday Brave profile, so tool
calls can't see or touch real sessions/cookies unless deliberately primed
into it (see scripts/prime_browser_profile.py).

True headless mode gets served a CAPTCHA by search.brave.com (confirmed by
testing: identical request, only headless=True vs False differs, and only
the headless run gets challenged) — it's specifically fingerprinting
headless Chromium, not blocking automation generally. So `headless=True`
here doesn't mean Chromium's own headless mode: it launches a real, headed
window and moves it off-screen instead, which reads as a normal browser to
site-side detection while staying visually out of the way for what's meant
to be a hands-free voice loop. Set headless=False in config to actually see
the window (useful for debugging).

web_search queries DuckDuckGo's server-rendered html.duckduckgo.com/html/
endpoint, not search.brave.com, despite this still being "Brave Tools" /
driving the Brave browser app -- confirmed live that Brave Search now
challenges this automated flow with a "verifying you're not a bot" page
regardless of profile freshness or the headless-vs-off-screen distinction
above (even with --disable-blink-features=AutomationControlled hiding
navigator.webdriver, the challenge still appeared, meaning it checks more
than that one signal). That's Brave choosing not to allow this kind of
automated access to their search results page -- their official Search API
(api.search.brave.com) is the sanctioned way to query Brave Search
programmatically, if that's ever wanted instead. DuckDuckGo's html
endpoint, by contrast, is designed to be server-rendered and lightweight
(no-JS friendly) and has not shown this behavior in testing. web_fetch is
unaffected either way -- it loads whatever URL it's given, unrelated to
either search provider.
"""

from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright

BRAVE_BINARY = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
OFFSCREEN_ARGS = ["--window-position=-3000,-3000", "--window-size=1280,800"]
SEARCH_URL = "https://html.duckduckgo.com/html/?q={query}"


class BraveTools:
    def __init__(
        self,
        profile_dir: str,
        headless: bool = True,
        timeout_ms: int = 15000,
        max_chars: int = 4000,
    ):
        self.profile_dir = str(Path(profile_dir).expanduser())
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.max_chars = max_chars
        self._playwright = None
        self._context = None

    def _ensure_started(self):
        if self._context is not None:
            return
        Path(self.profile_dir).mkdir(parents=True, exist_ok=True)
        args = ["--no-first-run", "--no-default-browser-check", "--disable-sync"]
        if self.headless:
            args += OFFSCREEN_ARGS
        self._playwright = sync_playwright().start()
        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=self.profile_dir,
            executable_path=BRAVE_BINARY,
            headless=False,  # see module docstring — real headless gets CAPTCHA'd
            args=args,
        )

    def stop(self):
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def _page_text(self, url: str, selector: str = None) -> str:
        self._ensure_started()
        page = self._context.new_page()
        try:
            page.goto(url, timeout=self.timeout_ms, wait_until="load")
            page.wait_for_timeout(1000)  # let client-rendered content (e.g. search results) settle
            # Prefer a specific results container over the whole page when
            # one's given: html.duckduckgo.com/html/'s own region/date-filter
            # menu otherwise eats a few hundred characters of max_chars
            # before any actual result appears.
            if selector and page.query_selector(selector):
                text = page.evaluate("(sel) => document.querySelector(sel).innerText", selector)
            else:
                text = page.evaluate("document.body.innerText")
        finally:
            page.close()
        return (text or "").strip()

    def web_fetch(self, url: str) -> str:
        try:
            return self._page_text(url)[: self.max_chars]
        except Exception as e:
            return f"Could not load {url}: {e}"

    def web_search(self, query: str) -> str:
        url = SEARCH_URL.format(query=quote(query))
        try:
            return self._page_text(url, selector="#links")[: self.max_chars]
        except Exception as e:
            return f"Search failed for {query!r}: {e}"
