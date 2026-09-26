"""Web search via Firecrawl's hosted search API (api.firecrawl.dev).

An alternative to tools/browser.py's DuckDuckGo-via-browser web_search:
Firecrawl's /v1/search endpoint returns structured, often already-
scraped results (title/url/description, sometimes full markdown page
content) directly over HTTPS, no local browser involved for search
itself. Requires an API key -- read from the FIRECRAWL_API_KEY
environment variable (see .env, gitignored: never put a real key in
config.yaml, which is tracked in git).

Deliberately search-only. web_fetch (loading one specific, already-
known URL) always goes through the local Brave browser
(tools/browser.py's BraveTools) regardless of which search provider is
configured -- Firecrawl also offers a /v1/scrape fetch endpoint, but
that's not used here by design: fetching a specific page stays local
rather than being routed through a third-party service.
"""

import requests

SEARCH_URL = "https://api.firecrawl.dev/v1/search"


class FirecrawlSearch:
    def __init__(self, api_key: str, timeout_s: int = 20, result_limit: int = 5, max_chars: int = 4000):
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.result_limit = result_limit
        self.max_chars = max_chars

    def web_search(self, query: str) -> str:
        if not self.api_key:
            return "Firecrawl search is not configured -- FIRECRAWL_API_KEY is missing."

        try:
            resp = requests.post(
                SEARCH_URL,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"query": query, "limit": self.result_limit},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return f"Search failed for {query!r}: {e}"

        if not data.get("success"):
            return f"Search failed for {query!r}: {data.get('error', 'unknown error')}"

        results = data.get("data", [])
        if not results:
            return f"No results found for {query!r}."

        blocks = []
        for r in results:
            title = r.get("title", "")
            url = r.get("url", "")
            description = r.get("description", "")
            blocks.append(f"{title}\n{url}\n{description}")
        return "\n\n".join(blocks)[: self.max_chars]
