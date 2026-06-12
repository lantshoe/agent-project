import asyncio
from ddgs import DDGS
from langchain_core.tools import tool

_search_lock: asyncio.Lock | None = None

def _get_lock() -> asyncio.Lock:
    global _search_lock
    if _search_lock is None:
        _search_lock = asyncio.Lock()
    return _search_lock


def _search_sync(query: str) -> str:
    try:
        with DDGS(timeout=10) as ddgs:
            result = list(ddgs.text(query, max_results=3))
            if not result:
                return "Error: No results found."
            output = []
            for i, r in enumerate(result, 1):
                output.append(f"[{i}] {r['title']}\n{r['body']}\nSource: {r['href']}")
            return "\n\n".join(output)
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
async def web_search(query: str) -> str:
    """
    search the web and return top results
    Use this when you need current information or facts you dont know.
    Example input: "latest AI news 2026"
    """
    async with _get_lock():
        return await asyncio.to_thread(_search_sync, query)