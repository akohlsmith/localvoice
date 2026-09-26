"""OpenAI-format tool schemas and dispatch, so ChatClient stays tool-agnostic
and new tools just get added here."""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web via Brave Search and return the visible text of "
                "the results page. Use this when you need current information "
                "you don't already know."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Load a URL in a real browser and return its visible text. Use this "
                "for pages that need JavaScript rendering, or to read a specific "
                "page found via web_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The URL to fetch."}},
                "required": ["url"],
            },
        },
    },
]


def build_dispatcher(browser):
    def dispatch(name: str, arguments: dict) -> str:
        if name == "web_search":
            return browser.web_search(arguments["query"])
        if name == "web_fetch":
            return browser.web_fetch(arguments["url"])
        return f"Unknown tool: {name}"

    return dispatch
