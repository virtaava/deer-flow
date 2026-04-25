"""Harness tools — restricted ACI tool layer for DeerFlow agents.

Replaces free-form sandbox tools (bash, write_file, str_replace) with
structured, validated, sequenced tools that give inline feedback.

Each tool wraps a HarnessTools method and returns structured feedback
that the LLM sees and acts on in real time.

Tool set:
    harness_read_file   — read with line numbers, tracks files_read
    harness_search      — capped search with "refine" feedback
    harness_propose_patch — validate diff via linters + git apply --check
    harness_apply_patch — only if proposal validated
    harness_run_tests   — structured test results
    harness_commit      — blocked unless tests pass

Removed (compared to default sandbox tools):
    bash, write_file, str_replace — all writes go through propose_patch
"""

import logging
import sys
from pathlib import Path

from langchain.tools import ToolRuntime, tool
from langgraph.typing import ContextT

from src.agents.thread_state import ThreadState
from src.sandbox.tools import (
    ensure_sandbox_initialized,
    ensure_thread_directories_exist,
    get_thread_data,
    is_local_sandbox,
    replace_virtual_path,
)

logger = logging.getLogger(__name__)

# ── HarnessTools instance management ─────────────────────────────────

# Lazy import to avoid hard dependency on harness at module load
_harness_tools_class = None
_harness_linters = None


def _get_harness_tools_class():
    global _harness_tools_class, _harness_linters
    if _harness_tools_class is None:
        harness_root = str(Path.home() / "sona" / ".harness")
        if harness_root not in sys.path:
            sys.path.insert(0, harness_root)
        from tools import HarnessTools
        _harness_tools_class = HarnessTools
        try:
            from linters import get_all_linters
            _harness_linters = get_all_linters()
        except ImportError:
            _harness_linters = []
    return _harness_tools_class


def _get_or_create_harness(runtime: ToolRuntime) -> "HarnessTools":
    """Get or create a HarnessTools instance for this thread.

    Stored in runtime.state so it persists across tool calls within
    a single agent run (preserving files_read, patches, etc).
    """
    if "harness_tools" not in runtime.state:
        HarnessTools = _get_harness_tools_class()
        # Resolve repo root from thread workspace
        thread_data = get_thread_data(runtime)
        workspace = thread_data.get("workspace_path", str(Path.home() / "sona"))
        runtime.state["harness_tools"] = HarnessTools(
            repo_root=workspace,
            linters=_harness_linters,
        )
    return runtime.state["harness_tools"]


def _resolve_path(path: str, runtime: ToolRuntime) -> str:
    """Resolve virtual path to actual path if using local sandbox."""
    if is_local_sandbox(runtime):
        thread_data = get_thread_data(runtime)
        return replace_virtual_path(path, thread_data)
    return path


# ── Tools ────────────────────────────────────────────────────────────


@tool("harness_read_file", parse_docstring=True)
def harness_read_file_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    description: str,
    path: str,
    offset: int = 0,
    limit: int = 200,
) -> str:
    """Read a file with line numbers. The harness tracks which files you've read — you can only propose patches to files you've read first.

    Shows up to 200 lines at a time with line numbers prepended. Use offset to paginate through large files.

    Args:
        description: Why you are reading this file. ALWAYS PROVIDE THIS FIRST.
        path: Absolute path to the file to read.
        offset: Line offset to start reading from (0-indexed). Default: 0.
        limit: Maximum number of lines to return. Default: 200.
    """
    try:
        ensure_sandbox_initialized(runtime)
        ensure_thread_directories_exist(runtime)
        harness = _get_or_create_harness(runtime)
        actual_path = _resolve_path(path, runtime)
        result = harness.read_file(actual_path, offset=offset, limit=limit)
        if "error" in result:
            return f"Error: {result['error']}"
        return (
            f"File: {path} ({result['total_lines']} lines, showing {result['showing']})\n"
            f"Files read so far: {len(harness.state.files_read)}\n\n"
            f"{result['content']}"
        )
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@tool("harness_search", parse_docstring=True)
def harness_search_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    description: str,
    query: str,
    path: str = ".",
    max_results: int = 30,
) -> str:
    """Search the repository for a pattern. Results are CAPPED at 30 — if you get too many results, narrow your search.

    Searches .py, .yaml, .json, and .md files.

    Args:
        description: Why you are searching. ALWAYS PROVIDE THIS FIRST.
        query: The search pattern (grep-compatible).
        path: Directory to search in. Default: current directory.
        max_results: Maximum results to return. Default: 30.
    """
    try:
        ensure_sandbox_initialized(runtime)
        ensure_thread_directories_exist(runtime)
        harness = _get_or_create_harness(runtime)
        actual_path = _resolve_path(path, runtime)
        result = harness.search_repo(query, path=actual_path, max_results=max_results)
        if "error" in result:
            return f"Error: {result['error']}"
        if result.get("message"):
            return f"⚠ {result['message']} Narrow your search query."
        if not result.get("results"):
            return f"No matches for '{query}'"
        header = f"Found {result['total_matches']} matches (showing {result['showing']}):\n"
        return header + "\n".join(result["results"])
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@tool("harness_propose_patch", parse_docstring=True)
def harness_propose_patch_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    description: str,
    diff: str,
    reason: str,
) -> str:
    """Propose a code change as a unified diff. This is the ONLY way to modify files.

    The patch is validated before it can be applied:
    - All files in the diff must have been read first (use harness_read_file)
    - The diff must be valid unified diff format
    - The patch must apply cleanly (git apply --check)
    - Architectural linters check for secrets, boundary violations, style issues

    If validation fails, you'll get specific error messages. Fix the issues and propose again.

    Args:
        description: Why you are proposing this change. ALWAYS PROVIDE THIS FIRST.
        diff: The unified diff. Must use --- a/ and +++ b/ format with @@ hunk headers.
        reason: A 1-2 sentence explanation of what this change does and why.
    """
    try:
        harness = _get_or_create_harness(runtime)
        result = harness.propose_patch(diff, reason)
        if "error" in result:
            return (
                f"❌ REJECTED: {result['error']}\n"
                f"Hint: {result.get('hint', 'Check the error and fix your diff.')}"
            )
        if result.get("status") == "rejected":
            errors = "\n".join(f"  • {e}" for e in result.get("errors", []))
            return (
                f"❌ PATCH REJECTED — {len(result.get('errors', []))} validation error(s):\n"
                f"{errors}\n\n"
                f"Fix these issues and propose again. Patch ID: {result['patch_id']}"
            )
        return (
            f"✓ PATCH VALIDATED — ready to apply.\n"
            f"  Patch ID: {result['patch_id']}\n"
            f"  Files: {', '.join(result.get('files', []))}\n"
            f"  Reason: {reason}\n\n"
            f"Use harness_apply_patch with patch_id='{result['patch_id']}' to apply."
        )
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@tool("harness_apply_patch", parse_docstring=True)
def harness_apply_patch_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    description: str,
    patch_id: str,
) -> str:
    """Apply a previously validated patch. The patch must have been proposed and validated first.

    After applying, tests_passed is reset to False — you must run harness_run_tests before committing.

    Args:
        description: Why you are applying this patch. ALWAYS PROVIDE THIS FIRST.
        patch_id: The patch ID returned by harness_propose_patch (e.g. 'patch-001').
    """
    try:
        harness = _get_or_create_harness(runtime)
        result = harness.apply_patch(patch_id)
        if "error" in result:
            return f"❌ APPLY FAILED: {result['error']}"
        return (
            f"✓ PATCH APPLIED: {patch_id}\n"
            f"  Files modified: {', '.join(result.get('files', []))}\n"
            f"  ⚠ tests_passed is now False — run harness_run_tests to verify."
        )
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@tool("harness_run_tests", parse_docstring=True)
def harness_run_tests_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    description: str,
    scope: str = "all",
    timeout: int = 120,
) -> str:
    """Run tests to verify your changes. You must run tests before committing.

    Args:
        description: Why you are running tests. ALWAYS PROVIDE THIS FIRST.
        scope: Test scope — 'all' for full suite, or a specific path like 'tests/test_foo.py'.
        timeout: Timeout in seconds. Default: 120.
    """
    try:
        harness = _get_or_create_harness(runtime)
        result = harness.run_tests(scope=scope, timeout=timeout)
        if result.get("error"):
            return f"❌ TEST ERROR: {result['error']}"
        if result.get("passed"):
            return (
                f"✓ TESTS PASSED\n"
                f"  You can now use harness_commit to commit your changes.\n\n"
                f"{result.get('stdout', '')[-500:]}"
            )
        return (
            f"❌ TESTS FAILED (exit code {result.get('returncode', '?')})\n"
            f"  Fix the failing tests and run again.\n\n"
            f"STDOUT:\n{result.get('stdout', '')[-800:]}\n"
            f"STDERR:\n{result.get('stderr', '')[-400:]}"
        )
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@tool("harness_commit", parse_docstring=True)
def harness_commit_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    description: str,
    message: str,
) -> str:
    """Commit the current changes. Only allowed if tests have passed.

    Args:
        description: Why you are committing. ALWAYS PROVIDE THIS FIRST.
        message: The commit message.
    """
    try:
        harness = _get_or_create_harness(runtime)
        result = harness.commit_change(message)
        if "error" in result:
            return f"❌ COMMIT BLOCKED: {result['error']}\nHint: {result.get('hint', '')}"
        return f"✓ COMMITTED: {message}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@tool("harness_status", parse_docstring=True)
def harness_status_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    description: str,
) -> str:
    """Check your current harness state — files read, patches proposed, tests status.

    Use this to understand what you can do next in the read→propose→apply→test→commit sequence.

    Args:
        description: Why you are checking status. ALWAYS PROVIDE THIS FIRST.
    """
    try:
        harness = _get_or_create_harness(runtime)
        state = harness.state
        files = sorted(state.files_read) if state.files_read else ["(none)"]
        patches = []
        for pid, p in state.patches_proposed.items():
            status = "✓ applied" if p.applied else ("✓ validated" if p.validated else "❌ rejected")
            patches.append(f"  {pid}: {status} — {', '.join(p.files_affected)}")
        if not patches:
            patches = ["  (none)"]

        return (
            f"Harness State:\n"
            f"  Files read: {len(state.files_read)}\n"
            f"    {chr(10).join(files)}\n"
            f"  Patches:\n"
            f"    {chr(10).join(patches)}\n"
            f"  Tests passed: {state.tests_passed}\n"
            f"  Actions taken: {len(state.action_log)}\n\n"
            f"Sequence: read → propose_patch → apply_patch → run_tests → commit"
        )
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
