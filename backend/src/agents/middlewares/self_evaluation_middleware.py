"""Middleware to evaluate final agent responses for quality and completeness.

Wraps the model call: after the model produces a final text response (no
tool calls), uses a separate LLM call to check for gaps.  If gaps are found,
appends a revision prompt and calls the model again.

Design constraints:
  - Only fires on "final" AI messages (text content, no tool_calls).
  - Skips short responses (< min_chars) — quick answers don't need review.
  - One revision per thread max — avoids infinite self-evaluation loops.
  - Uses a separate (cheap) model for evaluation by default.
  - Falls back to "approve" on any evaluation error — never blocks the agent.
  - Uses wrap_model_call (not after_model) so the revised response replaces
    the original in the same graph step.
"""

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from src.models.factory import create_chat_model

logger = logging.getLogger(__name__)

# Defaults — overridable via constructor
_DEFAULT_MIN_CHARS = 200
_DEFAULT_MAX_REVISIONS = 1
_DEFAULT_EVALUATION_MODEL = None  # None = use the same model as the agent

_EVALUATION_PROMPT = """You are a quality evaluator for an AI assistant's response.

Evaluate whether this response adequately answers the user's question.

Check for:
1. Completeness — does it address all parts of the question?
2. Accuracy — are there obvious errors or contradictions?
3. Actionability — if steps were requested, are they clear and complete?

User question:
{question}

Assistant response:
{response}

Return ONLY valid JSON (no markdown fences):
{{"verdict": "approve" or "revise", "gaps": ["list of specific gaps"], "confidence": 0.0-1.0}}"""


def _extract_ai_content(response: ModelResponse) -> str | None:
    """Extract text content from a ModelResponse, if it's a final response."""
    if not response.result:
        return None
    for msg in response.result:
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            return None  # Has tool calls — not a final response
        if msg.content and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
    return None


class SelfEvaluationMiddleware(AgentMiddleware[AgentState]):
    """Evaluates final AI responses and requests revision if quality is low.

    Uses wrap_model_call to intercept the model response. If the response
    is a final text answer with quality gaps, appends a revision prompt
    and calls the model again within the same graph step.

    Args:
        min_chars: Minimum response length to trigger evaluation.
        max_revisions: Maximum number of revision attempts per thread.
        evaluation_model: Model name for evaluation (None = same as agent).
        enabled: Whether the middleware is active.
    """

    def __init__(
        self,
        min_chars: int = _DEFAULT_MIN_CHARS,
        max_revisions: int = _DEFAULT_MAX_REVISIONS,
        evaluation_model: str | None = _DEFAULT_EVALUATION_MODEL,
        enabled: bool = True,
    ):
        super().__init__()
        self.min_chars = min_chars
        self.max_revisions = max_revisions
        self.evaluation_model = evaluation_model
        self.enabled = enabled
        # Per-thread tracking
        self._revision_count: dict[str, int] = defaultdict(int)
        self._eval_cache: dict[str, dict[str, dict]] = defaultdict(dict)

    def _get_thread_id(self, messages: list) -> str:
        """Extract thread_id heuristic from messages (used for tracking)."""
        # We don't have state in wrap_model_call, so use message hash as fallback
        # The real thread_id comes from _get_thread_id_from_state when available
        return "default"

    def _get_thread_id_from_state(self, state: AgentState) -> str:
        thread_data = state.get("thread_data")
        if thread_data and isinstance(thread_data, dict):
            return thread_data.get("workspace_path", "default") or "default"
        return "default"

    def _find_last_user_question(self, messages: list) -> str | None:
        """Find the most recent user message in the request messages."""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and msg.content:
                return msg.content[:2000]
        return None

    def _hash_response(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _evaluate(self, question: str, response: str, thread_id: str) -> dict[str, Any]:
        """Call LLM to evaluate the response. Returns verdict dict."""
        response_hash = self._hash_response(response)

        cached = self._eval_cache[thread_id].get(response_hash)
        if cached is not None:
            logger.debug("Self-eval cache hit for %s/%s", thread_id[:8], response_hash)
            return cached

        prompt = _EVALUATION_PROMPT.format(
            question=question[:2000],
            response=response[:4000],
        )

        try:
            model = create_chat_model(name=self.evaluation_model)
            result = model.invoke([HumanMessage(content=prompt)])
            content = result.content.strip()

            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            verdict = json.loads(content)

            if "verdict" not in verdict:
                verdict["verdict"] = "approve"
            if "gaps" not in verdict or not isinstance(verdict["gaps"], list):
                verdict["gaps"] = []
            if "confidence" not in verdict:
                verdict["confidence"] = 0.5

            self._eval_cache[thread_id][response_hash] = verdict

            logger.info(
                "Self-evaluation: verdict=%s confidence=%.2f gaps=%d thread=%s",
                verdict["verdict"],
                verdict["confidence"],
                len(verdict["gaps"]),
                thread_id[:8],
            )
            return verdict

        except Exception:
            logger.exception("Self-evaluation failed, defaulting to approve")
            return {"verdict": "approve", "gaps": [], "confidence": 0.0}

    def _maybe_revise(
        self,
        request: ModelRequest,
        response: ModelResponse,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Evaluate response and retry with revision prompt if needed."""
        if not self.enabled:
            return response

        ai_content = _extract_ai_content(response)
        if ai_content is None or len(ai_content) < self.min_chars:
            return response

        # Use a stable thread key from the message history
        thread_id = "default"
        for msg in request.messages:
            if isinstance(msg, SystemMessage) and "thread" in str(getattr(msg, "content", "")):
                break
        # Fallback: hash first user message for per-conversation tracking
        question = self._find_last_user_question(request.messages)
        if question:
            thread_id = self._hash_response(question)

        if self._revision_count[thread_id] >= self.max_revisions:
            logger.debug("Self-eval: revision budget exhausted for %s", thread_id[:8])
            return response

        if not question:
            return response

        verdict = self._evaluate(question, ai_content, thread_id)

        if verdict["verdict"] == "revise" and verdict["gaps"]:
            self._revision_count[thread_id] += 1
            gaps_text = "\n".join(f"- {gap}" for gap in verdict["gaps"])
            revision_msg = (
                "[SELF-EVALUATION] Your response has gaps that need addressing:\n\n"
                f"{gaps_text}\n\n"
                "Please revise your response to address these issues."
            )
            logger.info(
                "Requesting revision for %s (%d/%d): %d gaps",
                thread_id[:8],
                self._revision_count[thread_id],
                self.max_revisions,
                len(verdict["gaps"]),
            )

            # Append the original response + revision prompt, then call model again
            patched_messages = list(request.messages)
            if response.result:
                patched_messages.extend(response.result)
            patched_messages.append(SystemMessage(content=revision_msg))

            request = request.override(messages=patched_messages)
            return handler(request)

        return response

    async def _amaybe_revise(
        self,
        request: ModelRequest,
        response: ModelResponse,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Async version of _maybe_revise."""
        if not self.enabled:
            return response

        ai_content = _extract_ai_content(response)
        if ai_content is None or len(ai_content) < self.min_chars:
            return response

        question = self._find_last_user_question(request.messages)
        thread_id = self._hash_response(question) if question else "default"

        if self._revision_count[thread_id] >= self.max_revisions:
            logger.debug("Self-eval: revision budget exhausted for %s", thread_id[:8])
            return response

        if not question:
            return response

        verdict = self._evaluate(question, ai_content, thread_id)

        if verdict["verdict"] == "revise" and verdict["gaps"]:
            self._revision_count[thread_id] += 1
            gaps_text = "\n".join(f"- {gap}" for gap in verdict["gaps"])
            revision_msg = (
                "[SELF-EVALUATION] Your response has gaps that need addressing:\n\n"
                f"{gaps_text}\n\n"
                "Please revise your response to address these issues."
            )
            logger.info(
                "Requesting revision for %s (%d/%d): %d gaps",
                thread_id[:8],
                self._revision_count[thread_id],
                self.max_revisions,
                len(verdict["gaps"]),
            )

            patched_messages = list(request.messages)
            if response.result:
                patched_messages.extend(response.result)
            patched_messages.append(SystemMessage(content=revision_msg))

            request = request.override(messages=patched_messages)
            return await handler(request)

        return response

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        response = handler(request)
        return self._maybe_revise(request, response, handler)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        response = await handler(request)
        return await self._amaybe_revise(request, response, handler)

    def reset(self, thread_id: str | None = None) -> None:
        """Clear tracking state."""
        if thread_id:
            self._revision_count.pop(thread_id, None)
            self._eval_cache.pop(thread_id, None)
        else:
            self._revision_count.clear()
            self._eval_cache.clear()
