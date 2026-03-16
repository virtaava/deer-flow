"""
Hermes-aware ChatOllama wrapper.

Wraps langchain_ollama.ChatOllama to parse <tool_call> tags from models
that use the Hermes/ChatML tool format (NousResearch Hermes-4, etc.).

Ollama's native API doesn't return structured tool_calls for custom GGUF
models even when they output <tool_call> tags. This wrapper intercepts
the response and extracts tool calls using the Hermes parser.

Usage in config.yaml:
    - name: hermes4
      use: src.models.hermes_chat_ollama:HermesChatOllama
      model: hermes4:14b
      ...
"""

import json
import logging
import re
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

# Regex patterns for Hermes tool call format
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

THINK_PATTERN = re.compile(
    r"<think>(.*?)</think>",
    re.DOTALL,
)


def _parse_hermes_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Parse <tool_call> tags from Hermes-format model output.

    Returns (cleaned_content, tool_calls) where tool_calls is a list of
    dicts with 'name', 'args', 'id' keys matching LangChain's format.
    """
    tool_calls = []

    for match in TOOL_CALL_PATTERN.finditer(text):
        try:
            tc_json = json.loads(match.group(1))
            name = tc_json.get("name", "")
            args = tc_json.get("arguments", tc_json.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            tool_calls.append({
                "name": name,
                "args": args,
                "id": f"hermes_{hash(name + json.dumps(args, sort_keys=True)) % 10**8:08d}",
                "type": "tool_call",
            })
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool call JSON: {match.group(1)[:100]}")

    # Clean content: remove tool call tags, keep think tags as-is for logging
    cleaned = TOOL_CALL_PATTERN.sub("", text).strip()
    # Also strip <think> content from the visible content (it's reasoning, not output)
    cleaned = THINK_PATTERN.sub("", cleaned).strip()

    return cleaned, tool_calls


class HermesChatOllama(ChatOllama):
    """ChatOllama with Hermes tool call parsing.

    Intercepts responses and parses <tool_call> tags into structured
    tool_calls that LangChain's agent framework understands.
    """

    def _generate(self, messages: List[Any], stop: Optional[List[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        """Override to parse Hermes tool calls from raw output."""
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        # Check each generation for <tool_call> tags
        for gen in result.generations:
            msg = gen.message
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                if "<tool_call>" in content_str:
                    cleaned, tool_calls = _parse_hermes_tool_calls(content_str)
                    if tool_calls:
                        msg.tool_calls = tool_calls
                        msg.content = cleaned or ""
                        logger.info(f"Hermes parser: extracted {len(tool_calls)} tool call(s)")

        return result

    async def _agenerate(self, messages: List[Any], stop: Optional[List[str]] = None,
                         run_manager: Any = None, **kwargs: Any) -> ChatResult:
        """Async override to parse Hermes tool calls."""
        result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

        for gen in result.generations:
            msg = gen.message
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                if "<tool_call>" in content_str:
                    cleaned, tool_calls = _parse_hermes_tool_calls(content_str)
                    if tool_calls:
                        msg.tool_calls = tool_calls
                        msg.content = cleaned or ""
                        logger.info(f"Hermes parser: extracted {len(tool_calls)} tool call(s)")

        return result
