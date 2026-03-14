import json
import logging
import urllib.error
import urllib.request
import urllib.parse

from langchain.tools import tool

from src.config import get_app_config

logger = logging.getLogger(__name__)


def _get_tool_extra(tool_name: str) -> dict:
    """Safely get model_extra dict for a tool config, handling None."""
    config = get_app_config().get_tool_config(tool_name)
    if config is None:
        return {}
    return config.model_extra or {}


def _get_base_url() -> str:
    extra = _get_tool_extra("web_search")
    return extra.get("base_url", "http://127.0.0.1:8080")


def _get_max_content_chars() -> int:
    """Get max_content_chars from config with validation."""
    extra = _get_tool_extra("web_fetch")
    raw = extra.get("max_content_chars", 16384)
    try:
        value = int(raw)
        return max(value, 1)
    except (TypeError, ValueError):
        logger.warning("Invalid max_content_chars=%r, using default 16384", raw)
        return 16384


@tool("web_search", parse_docstring=True)
def web_search_tool(query: str) -> str:
    """Search the web.

    Args:
        query: The query to search for.
    """
    extra = _get_tool_extra("web_search")
    try:
        max_results = int(extra.get("max_results", 5))
    except (TypeError, ValueError):
        max_results = 5

    base_url = _get_base_url()
    params = urllib.parse.urlencode({"q": query, "format": "json"})
    url = f"{base_url}/search?{params}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        return json.dumps({"error": f"Search request failed: {exc}"})

    results = data.get("results", [])[:max_results]
    normalized = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in results
    ]
    return json.dumps(normalized, indent=2, ensure_ascii=False)


def _firecrawl_fetch(url: str, max_content_chars: int) -> str | None:
    """Try to fetch via Firecrawl API. Returns markdown content or None on failure."""
    import os
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        return None
    try:
        from firecrawl import FirecrawlApp
        api_url = os.environ.get("FIRECRAWL_API_URL")
        client = FirecrawlApp(api_key=api_key, api_url=api_url) if api_url else FirecrawlApp(api_key=api_key)
        result = client.scrape(url, formats=["markdown"])
        markdown = result.markdown or ""
        if not markdown:
            return None
        title = ""
        if result.metadata and result.metadata.title:
            title = result.metadata.title
        return f"# {title}\n\n{markdown[:max_content_chars]}" if title else markdown[:max_content_chars]
    except Exception as exc:
        logger.debug("Firecrawl fetch failed for %s: %s", url, exc)
        return None


def _urllib_fetch(url: str, timeout: int, max_content_chars: int) -> str:
    """Fallback fetch via urllib."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; DeerFlow/2.0)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        return json.dumps({"error": f"Failed to fetch URL: {exc}"})

    if "text/html" in content_type:
        try:
            from src.utils.readability import ReadabilityExtractor
            extractor = ReadabilityExtractor()
            article = extractor.extract_article(raw.decode("utf-8", errors="replace"))
            return article.to_markdown()[:max_content_chars]
        except Exception as exc:
            logger.debug("Readability extraction failed for %s: %s", url, exc)

    text = raw.decode("utf-8", errors="replace")
    return text[:max_content_chars]


@tool("web_fetch", parse_docstring=True)
def web_fetch_tool(url: str) -> str:
    """Fetch the contents of a web page at a given URL.
    Only fetch EXACT URLs that have been provided directly by the user or have been returned in results from the web_search and web_fetch tools.
    This tool can NOT access content that requires authentication, such as private Google Docs or pages behind login walls.
    Do NOT add www. to URLs that do NOT have them.
    URLs must include the schema: https://example.com is a valid URL while example.com is an invalid URL.

    Args:
        url: The URL to fetch the contents of.
    """
    if not url.startswith(("http://", "https://")):
        return json.dumps({"error": "Invalid URL scheme. URLs must start with http:// or https://"})

    extra = _get_tool_extra("web_fetch")
    try:
        timeout = int(extra.get("timeout", 10))
    except (TypeError, ValueError):
        timeout = 10
    max_content_chars = _get_max_content_chars()

    # Primary: Firecrawl (handles JS rendering, anti-bot)
    result = _firecrawl_fetch(url, max_content_chars)
    if result:
        return result

    # Fallback: direct urllib fetch
    return _urllib_fetch(url, timeout, max_content_chars)
