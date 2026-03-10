"""Scratchpad tools for shared storage between lead agent and subagents.

Three tools: save_finding, read_findings, scratchpad_stats.
Scratchpad file is per-thread at {thread_workspace}/scratchpad.json.
"""

import logging
from typing import Annotated

from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langgraph.typing import ContextT

from src.agents.thread_state import ThreadState
from src.config.paths import get_paths

from .shared_scratchpad import EntryType, FileScratchpad, ScratchpadEntry

logger = logging.getLogger(__name__)


def _get_scratchpad(runtime: ToolRuntime[ContextT, ThreadState]) -> FileScratchpad:
    """Get scratchpad instance for the current thread."""
    thread_id = runtime.context.get("thread_id")
    if not thread_id:
        raise ValueError("Thread ID not available in runtime context")
    workspace_dir = get_paths().sandbox_work_dir(thread_id)
    return FileScratchpad(workspace_dir / "scratchpad.json")


def _get_agent_identity(runtime: ToolRuntime[ContextT, ThreadState]) -> str:
    metadata = runtime.config.get("metadata", {})
    if metadata.get("is_subagent", False):
        task_id = metadata.get("task_id", "unknown")
        return f"subagent:{task_id}"
    return "lead_agent"


@tool("save_finding", parse_docstring=True)
def save_finding_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    key: str,
    value: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    entry_type: str = "finding",
    confidence: float = 1.0,
    source: str = "unknown",
) -> str:
    """Save a finding to the shared scratchpad accessible by all agents in this thread.

    Use this to store research findings, URLs, code snippets, or notes that
    subagents or the lead agent should be able to read later.

    Args:
        key: Unique identifier for this entry (e.g., "api_endpoint_url").
        value: Content to store (text, JSON, code, etc.).
        entry_type: Type: finding, note, data, url, code, error.
        confidence: Certainty score 0.0-1.0.
        source: Where this came from (e.g., "web_search", "code_analysis").
    """
    try:
        scratchpad = _get_scratchpad(runtime)
        agent_id = _get_agent_identity(runtime)

        try:
            etype = EntryType(entry_type.lower())
        except ValueError:
            valid = [t.value for t in EntryType]
            return f"Error: Invalid entry_type '{entry_type}'. Valid: {', '.join(valid)}"

        if not 0.0 <= confidence <= 1.0:
            return f"Error: confidence must be 0.0-1.0, got {confidence}"

        entry = ScratchpadEntry(key=key, value=value, entry_type=etype, confidence=confidence, source=source)
        scratchpad.save_entry(entry, agent_id)
        return f"Saved '{key}' to scratchpad (type={entry_type}, confidence={confidence})"

    except Exception as e:
        logger.exception("Failed to save finding")
        return f"Error: {e}"


@tool("read_findings", parse_docstring=True)
def read_findings_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    entry_type: str | None = None,
    min_confidence: float = 0.0,
) -> str:
    """Read findings from the shared scratchpad with optional filtering.

    Returns all entries saved by any agent in this thread's scratchpad.

    Args:
        entry_type: Optional filter (finding, note, data, url, code, error).
        min_confidence: Minimum confidence score 0.0-1.0.
    """
    try:
        scratchpad = _get_scratchpad(runtime)

        etype = None
        if entry_type:
            try:
                etype = EntryType(entry_type.lower())
            except ValueError:
                valid = [t.value for t in EntryType]
                return f"Error: Invalid entry_type '{entry_type}'. Valid: {', '.join(valid)}"

        entries = scratchpad.read_entries(entry_type=etype, min_confidence=min_confidence)

        if not entries:
            return "No findings in scratchpad matching criteria."

        lines = [f"Found {len(entries)} entries:\n"]
        for i, e in enumerate(entries, 1):
            preview = str(e.value)[:200]
            lines.append(f"{i}. [{e.entry_type.value}] **{e.key}** (conf={e.confidence:.1f}, src={e.source})")
            lines.append(f"   {preview}")
        return "\n".join(lines)

    except Exception as e:
        logger.exception("Failed to read findings")
        return f"Error: {e}"


@tool("scratchpad_stats")
def scratchpad_stats_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Get statistics about the shared scratchpad (entry count, agents, types)."""
    try:
        scratchpad = _get_scratchpad(runtime)
        stats = scratchpad.get_stats()

        lines = [
            f"Entries: {stats['total_entries']}",
            f"Agents: {stats['agents']}",
            f"Types: {stats['entry_types']}",
            f"Avg confidence: {stats['confidence_avg']:.2f}",
        ]
        return "\n".join(lines)

    except Exception as e:
        logger.exception("Failed to get scratchpad stats")
        return f"Error: {e}"
