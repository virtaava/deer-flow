"""Helpers for safely reading run-scoped values from a LangGraph Runtime.

LangGraph passes per-run information through two channels:

- ``runtime.context`` — populated only if the caller explicitly passes a
  ``context`` dict on run creation. Many standard callers (the LangGraph
  Server REST API, the LangGraph SDK, the LangGraph Studio UI) do not set
  it; in that case ``runtime.context`` is ``None``.
- ``config.configurable`` — always populated by LangGraph Server with at
  minimum ``thread_id`` (taken from the URL path). Accessed via
  ``langgraph.config.get_config()`` from inside a node or middleware.

DeerFlow's middlewares historically read ``runtime.context.get("thread_id")``
unconditionally, which raised ``AttributeError: 'NoneType' object has no
attribute 'get'`` for any caller that did not pass a context dict. These
helpers provide a single, defensive accessor that prefers ``runtime.context``
when present, falls back to the LangGraph-injected config, and raises a
meaningful error when the value really is missing.
"""

from __future__ import annotations

from typing import Any


def resolve_runtime_value(runtime: Any, key: str) -> Any | None:
    """Look up ``key`` in ``runtime.context`` first, then ``config.configurable``.

    Returns ``None`` if absent in both. Never raises.
    """
    context = getattr(runtime, "context", None) or {}
    value = context.get(key) if isinstance(context, dict) else None
    if value is not None:
        return value

    try:
        from langgraph.config import get_config

        config = get_config() or {}
    except Exception:
        return None

    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    if isinstance(configurable, dict):
        return configurable.get(key)
    return None


def require_thread_id(runtime: Any) -> str:
    """Resolve ``thread_id`` or raise ``ValueError`` with a helpful message."""
    thread_id = resolve_runtime_value(runtime, "thread_id")
    if not thread_id:
        raise ValueError(
            "Thread ID is required. Set context.thread_id when creating the run "
            "or rely on LangGraph Server to populate config.configurable.thread_id "
            "from the thread URL."
        )
    return str(thread_id)
