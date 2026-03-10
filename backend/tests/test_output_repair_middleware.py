"""Tests for OutputRepairMiddleware."""

from unittest.mock import MagicMock

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage

from src.agents.middlewares.output_repair_middleware import (
    OutputRepairMiddleware,
    _is_valid_response,
)


def _make_response(content="", tool_calls=None):
    """Build a ModelResponse with a single AIMessage."""
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    return ModelResponse(result=[msg])


def _make_empty_response():
    return ModelResponse(result=[])


class TestIsValidResponse:
    def test_empty_result_list(self):
        assert not _is_valid_response(_make_empty_response())

    def test_valid_text(self):
        assert _is_valid_response(_make_response("Hello"))

    def test_empty_text(self):
        assert not _is_valid_response(_make_response(""))

    def test_whitespace_only(self):
        assert not _is_valid_response(_make_response("   "))

    def test_valid_tool_calls(self):
        tc = [{"name": "bash", "id": "c1", "args": {"command": "ls"}}]
        assert _is_valid_response(_make_response("", tool_calls=tc))

    def test_tool_calls_with_empty_args(self):
        """Empty dict args are valid — the tool will handle missing params."""
        tc = [{"name": "bash", "id": "c1", "args": {}}]
        assert _is_valid_response(_make_response("", tool_calls=tc))

    def test_list_content_with_text(self):
        msg = AIMessage(content=[{"type": "text", "text": "Hello"}])
        resp = ModelResponse(result=[msg])
        assert _is_valid_response(resp)

    def test_list_content_empty(self):
        msg = AIMessage(content=[])
        resp = ModelResponse(result=[msg])
        assert not _is_valid_response(resp)


class TestOutputRepairMiddleware:
    def test_valid_response_no_retry(self):
        mw = OutputRepairMiddleware(max_retries=2)
        good_response = _make_response("Hello")
        handler = MagicMock(return_value=good_response)
        request = MagicMock()

        result = mw._call_with_retry(request, handler)

        assert handler.call_count == 1
        assert result == good_response

    def test_retry_on_empty_then_succeed(self):
        mw = OutputRepairMiddleware(max_retries=2)
        bad = _make_response("")
        good = _make_response("Fixed!")
        handler = MagicMock(side_effect=[bad, good])
        request = MagicMock()
        request.messages = []
        request.override = MagicMock(return_value=request)

        result = mw._call_with_retry(request, handler)

        assert handler.call_count == 2
        assert result == good

    def test_exhaust_retries(self):
        mw = OutputRepairMiddleware(max_retries=2)
        bad = _make_response("")
        handler = MagicMock(return_value=bad)
        request = MagicMock()
        request.messages = []
        request.override = MagicMock(return_value=request)

        result = mw._call_with_retry(request, handler)

        # 1 initial + 2 retries = 3 calls
        assert handler.call_count == 3
        assert result == bad  # returns last attempt even if bad

    def test_retry_includes_bad_response_in_context(self):
        mw = OutputRepairMiddleware(max_retries=1)
        bad = _make_response("")
        good = _make_response("OK")
        handler = MagicMock(side_effect=[bad, good])
        request = MagicMock()
        request.messages = [MagicMock()]
        request.override = MagicMock(return_value=request)

        mw._call_with_retry(request, handler)

        # Check that override was called with extended messages
        request.override.assert_called_once()
        call_kwargs = request.override.call_args
        new_messages = call_kwargs[1]["messages"]
        # Original message + bad AIMessage + repair SystemMessage
        assert len(new_messages) == 3
