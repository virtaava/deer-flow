"""Middleware to enforce turn budget and force output before recursion limit.

Unlike exploration_budget prompt instructions (which models can ignore), this
middleware hard-injects warnings into the conversation at configurable
thresholds, and at the final threshold strips tool calls to force a text
response.

Designed for cheap, fast models (e.g. DeepSeek) that don't reliably self-limit.
"""

import logging
from collections import defaultdict
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# Default thresholds as fractions of max_turns
_DEFAULT_WARN_FRACTION = 0.65  # "Getting close" warning
_DEFAULT_URGENT_FRACTION = 0.85  # "Wrap up NOW" warning
_DEFAULT_FORCE_FRACTION = 0.95  # Strip tool calls, force final answer

_WARN_MSG = (
    "[BUDGET WARNING] You have used {used} of {total} allowed turns. "
    "Start wrapping up — produce your final answer soon."
)

_URGENT_MSG = (
    "[BUDGET CRITICAL] You have used {used} of {total} turns — only {remaining} left! "
    "STOP exploring. Immediately produce your final summary with everything "
    "you have found so far."
)

_FORCE_MSG = (
    "[BUDGET EXHAUSTED] Turn limit nearly reached ({used}/{total}). "
    "Producing final answer with all results collected so far."
)


def _extract_text_from_content(content: list[Any]) -> str:
    """Extract text from multimodal content list."""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content")
            if isinstance(text, str):
                parts.append(text)
        else:
            parts.append(str(item))
    return "\n".join(parts)


class BudgetEnforcementMiddleware(AgentMiddleware[AgentState]):
    """Injects budget warnings and forces output before recursion limit.

    Counts before_model invocations directly (not messages in state, which
    can decrease due to summarization).  Each model call consumes ~8 graph
    steps (model node + tools node + middleware nodes), so thresholds are
    applied against max_turns // 8.

    Args:
        max_turns: The recursion limit for this agent.
        warn_fraction: Fraction of max_turns to trigger first warning.
        urgent_fraction: Fraction to trigger urgent warning.
        force_fraction: Fraction to strip tool calls and force answer.
    """

    def __init__(
        self,
        max_turns: int = 50,
        warn_fraction: float = _DEFAULT_WARN_FRACTION,
        urgent_fraction: float = _DEFAULT_URGENT_FRACTION,
        force_fraction: float = _DEFAULT_FORCE_FRACTION,
    ):
        super().__init__()
        self.max_turns = max_turns
        # Each model call consumes ~8 graph steps (model node + tools node +
        # middleware nodes).  Estimate effective model calls from recursion_limit.
        effective_calls = max(max_turns // 8, 10)
        self.warn_at = int(effective_calls * warn_fraction)
        self.urgent_at = int(effective_calls * urgent_fraction)
        self.force_at = int(effective_calls * force_fraction)
        # Direct invocation counter per thread (immune to summarization)
        self._call_count: dict[str, int] = defaultdict(int)
        # Track which warnings we've already sent per thread to avoid spam
        self._warned: dict[str, set[str]] = defaultdict(set)
        # Track last seen message count per thread to detect new runs
        self._last_msg_count: dict[str, int] = defaultdict(int)

    def _get_thread_id(self, state: AgentState, runtime: Runtime | None = None) -> str:
        # Prefer runtime.context for thread_id if available
        if runtime and hasattr(runtime, "context") and isinstance(runtime.context, dict):
            tid = runtime.context.get("thread_id")
            if tid:
                return str(tid)
        thread_data = state.get("thread_data")
        if thread_data and isinstance(thread_data, dict):
            return thread_data.get("workspace_path", "default") or "default"
        return "default"

    def _maybe_reset_for_new_run(self, state: AgentState, thread_id: str) -> None:
        """Reset counters if message count decreased (new run on cached instance)."""
        msg_count = len(state.get("messages", []))
        last_count = self._last_msg_count.get(thread_id, 0)
        if msg_count < last_count:
            logger.info("Budget middleware: detected new run for thread %s (msgs %d < %d), resetting", thread_id, msg_count, last_count)
            self.reset(thread_id)
        self._last_msg_count[thread_id] = msg_count

    def _apply(self, state: AgentState, runtime: Runtime | None = None) -> dict | None:
        thread_id = self._get_thread_id(state, runtime)
        self._maybe_reset_for_new_run(state, thread_id)
        self._call_count[thread_id] += 1
        turns_used = self._call_count[thread_id]
        effective_total = max(self.max_turns // 8, 10)
        logger.debug("Budget check: %d/%d model calls (warn=%d, urgent=%d, force=%d)", turns_used, effective_total, self.warn_at, self.urgent_at, self.force_at)
        warned = self._warned[thread_id]

        if turns_used >= self.force_at and "force" not in warned:
            warned.add("force")
            logger.warning(
                "Budget exhausted — forcing final answer",
                extra={"thread_id": thread_id, "turns_used": turns_used, "max_turns": effective_total},
            )
            return {"messages": [SystemMessage(content=_FORCE_MSG.format(
                used=turns_used, total=effective_total,
            ))]}

        if turns_used >= self.urgent_at and "urgent" not in warned:
            warned.add("urgent")
            logger.warning(
                "Budget critical — injecting urgent warning",
                extra={"thread_id": thread_id, "turns_used": turns_used, "max_turns": effective_total},
            )
            remaining = effective_total - turns_used
            return {"messages": [SystemMessage(content=_URGENT_MSG.format(
                used=turns_used, total=effective_total, remaining=remaining,
            ))]}

        if turns_used >= self.warn_at and "warn" not in warned:
            warned.add("warn")
            logger.info(
                "Budget warning — injecting early warning",
                extra={"thread_id": thread_id, "turns_used": turns_used, "max_turns": effective_total},
            )
            return {"messages": [SystemMessage(content=_WARN_MSG.format(
                used=turns_used, total=effective_total,
            ))]}

        return None

    def _apply_after_model(self, state: AgentState, runtime: Runtime | None = None) -> dict | None:
        """After model responds: if we're past force threshold, strip tool calls."""
        thread_id = self._get_thread_id(state, runtime)
        turns_used = self._call_count.get(thread_id, 0)
        if turns_used < self.force_at:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage):
            return None

        tool_calls = getattr(last_msg, "tool_calls", None)
        if not tool_calls:
            return None

        logger.warning(
            "Budget force-stop — stripping %d tool calls from AI response",
            len(tool_calls),
            extra={"thread_id": thread_id, "turns_used": turns_used},
        )
        effective_total = max(self.max_turns // 8, 10)
        force_text = _FORCE_MSG.format(used=turns_used, total=effective_total)

        # Handle both string and list content (multimodal messages)
        content = last_msg.content
        if isinstance(content, list):
            base_content = _extract_text_from_content(content)
        else:
            base_content = content or ""
        new_content = f"{base_content}\n\n{force_text}" if base_content else force_text

        stripped_msg = last_msg.model_copy(update={
            "tool_calls": [],
            "content": new_content,
        })
        return {"messages": [stripped_msg]}

    @override
    def before_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._apply(state, runtime)

    @override
    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._apply(state, runtime)

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._apply_after_model(state, runtime)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._apply_after_model(state, runtime)

    def reset(self, thread_id: str | None = None) -> None:
        if thread_id:
            self._warned.pop(thread_id, None)
            self._call_count.pop(thread_id, None)
            self._last_msg_count.pop(thread_id, None)
        else:
            self._warned.clear()
            self._call_count.clear()
            self._last_msg_count.clear()
