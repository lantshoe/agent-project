from ddgs import DDGS
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """
    search the web and return top results
    Use this when you need current information or facts you dont know.
    Example input: "latest AI news 2026"
    """
    try:
        with DDGS() as ddgs:
            result = list(ddgs.text(query, max_results=3))
            if not result:
                return "Error: No results found."
            output = []
            for i, r in enumerate(result, 1):
                output.append(f"[{i}] {r['title']}\n{r['body']}\nSource: {r['href']}")

            return "\n\n".join(output)
    except Exception as e:
        return f"Search error: {str(e)}"



