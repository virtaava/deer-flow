"""Middleware to retry LLM calls when the model produces malformed output.

Catches common failure modes with non-OpenAI models (DeepSeek, Qwen, Ollama)
where the model returns:
  - Empty responses (no content, no tool_calls)
  - Malformed JSON in tool call arguments
  - Content that the agent framework can't parse

On failure, retries with a corrective system message appended, up to max_retries.
"""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RETRIES = 2

_REPAIR_PROMPT = (
    "Your previous response was malformed or empty. "
    "Please try again. Respond with either:\n"
    "1. A clear text answer, OR\n"
    "2. Valid tool calls with properly formatted JSON arguments.\n"
    "Do not output partial JSON or empty responses."
)


def _is_valid_response(response: ModelResponse) -> bool:
    """Check if a model response is usable."""
    if not response.result:
        return False

    for msg in response.result:
        if not isinstance(msg, AIMessage):
            continue

        # Has text content — valid
        content = msg.content
        if content:
            if isinstance(content, str) and content.strip():
                return True
            if isinstance(content, list) and any(
                (isinstance(b, str) and b.strip())
                or (isinstance(b, dict) and b.get("text", "").strip())
                for b in content
            ):
                return True

        # Has tool calls — check they have valid args
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            all_valid = True
            for tc in tool_calls:
                args = tc.get("args", {})
                if args is None:
                    all_valid = False
                    break
                # Check args is a dict (not a raw string or malformed)
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                        if not isinstance(parsed, dict):
                            all_valid = False
                            break
                    except (json.JSONDecodeError, ValueError):
                        all_valid = False
                        break
            if all_valid:
                return True

    return False


class OutputRepairMiddleware(AgentMiddleware[AgentState]):
    """Retries LLM calls when the model produces malformed or empty output.

    Args:
        max_retries: Maximum number of retry attempts. Default: 2.
    """

    def __init__(self, max_retries: int = _DEFAULT_MAX_RETRIES):
        super().__init__()
        self.max_retries = max_retries

    def _call_with_retry(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        response = handler(request)

        for attempt in range(self.max_retries):
            if _is_valid_response(response):
                return response

            logger.warning(
                "Malformed model output (attempt %d/%d), retrying with repair prompt",
                attempt + 1,
                self.max_retries,
            )

            # Append the repair instruction to messages
            repair_msg = SystemMessage(content=_REPAIR_PROMPT)
            patched_messages = list(request.messages)

            # Also include the bad response so the model sees what went wrong
            if response.result:
                patched_messages.extend(response.result)
            patched_messages.append(repair_msg)

            request = request.override(messages=patched_messages)
            response = handler(request)

        if not _is_valid_response(response):
            logger.error(
                "Model output still malformed after %d retries, returning as-is",
                self.max_retries,
            )

        return response

    async def _acall_with_retry(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        response = await handler(request)

        for attempt in range(self.max_retries):
            if _is_valid_response(response):
                return response

            logger.warning(
                "Malformed model output (attempt %d/%d), retrying with repair prompt",
                attempt + 1,
                self.max_retries,
            )

            repair_msg = SystemMessage(content=_REPAIR_PROMPT)
            patched_messages = list(request.messages)
            if response.result:
                patched_messages.extend(response.result)
            patched_messages.append(repair_msg)

            request = request.override(messages=patched_messages)
            response = await handler(request)

        if not _is_valid_response(response):
            logger.error(
                "Model output still malformed after %d retries, returning as-is",
                self.max_retries,
            )

        return response

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return self._call_with_retry(request, handler)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await self._acall_with_retry(request, handler)
