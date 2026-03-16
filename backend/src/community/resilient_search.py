"""
Resilient search — multi-source web search with circuit breakers.

Wraps SearxNG search with fallback and circuit breaker pattern:
- Primary: SearxNG (local, no API key)
- Circuit breaker: if source fails 3x in 10min, skip it
- Result dedup across retries
- Quality scoring (prefer results with snippets)

Usage:
    results = resilient_search("DGX Spark specs", max_results=5)
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Circuit breaker state
_circuit_state: dict[str, dict] = {}
CIRCUIT_THRESHOLD = 3  # failures before opening
CIRCUIT_RESET_SECONDS = 600  # 10 min


def _is_circuit_open(source: str) -> bool:
    """Check if a source's circuit breaker is open (should skip)."""
    state = _circuit_state.get(source)
    if not state:
        return False
    if state["failures"] >= CIRCUIT_THRESHOLD:
        if time.time() - state["last_failure"] < CIRCUIT_RESET_SECONDS:
            return True
        # Reset after timeout
        _circuit_state[source] = {"failures": 0, "last_failure": 0}
    return False


def _record_failure(source: str):
    """Record a failure for circuit breaker."""
    if source not in _circuit_state:
        _circuit_state[source] = {"failures": 0, "last_failure": 0}
    _circuit_state[source]["failures"] += 1
    _circuit_state[source]["last_failure"] = time.time()
    logger.warning(f"Search source '{source}' failure #{_circuit_state[source]['failures']}")


def _record_success(source: str):
    """Reset failure count on success."""
    _circuit_state[source] = {"failures": 0, "last_failure": 0}


def _dedup_results(results: list[dict]) -> list[dict]:
    """Remove duplicate results by URL."""
    seen_urls = set()
    deduped = []
    for r in results:
        url = r.get("url", r.get("link", ""))
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(r)
    return deduped


def _score_result(result: dict) -> float:
    """Score a search result by quality (0-1)."""
    score = 0.0
    if result.get("snippet") or result.get("content"):
        score += 0.4
    if result.get("title"):
        score += 0.2
    if result.get("url"):
        score += 0.2
    # Prefer non-social-media results
    url = result.get("url", "")
    if any(d in url for d in ["reddit.com", "twitter.com", "facebook.com"]):
        score -= 0.1
    if any(d in url for d in [".gov", ".edu", ".org", "arxiv.org", "github.com"]):
        score += 0.2
    return min(1.0, max(0.0, score))


def resilient_search(query: str, max_results: int = 5,
                      searxng_url: str = "http://127.0.0.1:8080") -> list[dict]:
    """Search with circuit breaker and retry.

    Returns list of {title, url, snippet} dicts.
    """
    import requests

    results = []

    # Source 1: SearxNG (primary)
    if not _is_circuit_open("searxng"):
        try:
            resp = requests.get(f"{searxng_url}/search", params={
                "q": query,
                "format": "json",
                "categories": "general",
                "language": "en",
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for r in data.get("results", [])[:max_results * 2]:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                    "source": "searxng",
                })
            _record_success("searxng")
        except Exception as e:
            _record_failure("searxng")
            logger.warning(f"SearxNG search failed: {e}")

    # Dedup, score, and sort
    results = _dedup_results(results)
    results.sort(key=lambda r: _score_result(r), reverse=True)

    return results[:max_results]


def get_circuit_status() -> dict[str, Any]:
    """Get current circuit breaker status for all sources."""
    return {
        source: {
            "failures": state["failures"],
            "open": _is_circuit_open(source),
            "last_failure": state["last_failure"],
        }
        for source, state in _circuit_state.items()
    }
