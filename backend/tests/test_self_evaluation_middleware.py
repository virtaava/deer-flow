"""Tests for SelfEvaluationMiddleware."""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.middlewares.self_evaluation_middleware import (
    SelfEvaluationMiddleware,
    _extract_ai_content,
)


def _long_text(text="Python is a programming language. " * 20):
    return text


def _make_response(content="", tool_calls=None):
    """Build a ModelResponse with an AIMessage."""
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    resp = MagicMock()
    resp.result = [msg]
    return resp


def _make_request(user_question="What is Python?"):
    """Build a ModelRequest with a user question."""
    req = MagicMock()
    req.messages = [HumanMessage(content=user_question)]
    req.override = MagicMock(side_effect=lambda **kwargs: MagicMock(messages=kwargs.get("messages", req.messages)))
    return req


class TestExtractAiContent:
    def test_extracts_text(self):
        resp = _make_response(content="Hello world")
        assert _extract_ai_content(resp) == "Hello world"

    def test_returns_none_for_tool_calls(self):
        resp = _make_response(content="text", tool_calls=[{"name": "bash", "id": "1", "args": {}}])
        assert _extract_ai_content(resp) is None

    def test_returns_none_for_empty(self):
        resp = _make_response(content="")
        assert _extract_ai_content(resp) is None

    def test_returns_none_for_no_result(self):
        resp = MagicMock()
        resp.result = None
        assert _extract_ai_content(resp) is None


class TestSkipConditions:
    def test_disabled_returns_response(self):
        mw = SelfEvaluationMiddleware(enabled=False)
        resp = _make_response(content=_long_text())
        req = _make_request()
        handler = MagicMock(return_value=resp)
        result = mw.wrap_model_call(req, handler)
        assert result is resp

    def test_short_response_skipped(self):
        mw = SelfEvaluationMiddleware(min_chars=200)
        resp = _make_response(content="Short.")
        req = _make_request()
        handler = MagicMock(return_value=resp)
        result = mw.wrap_model_call(req, handler)
        assert result is resp

    def test_tool_calls_skipped(self):
        mw = SelfEvaluationMiddleware()
        resp = _make_response(
            content=_long_text(),
            tool_calls=[{"name": "bash", "id": "1", "args": {"command": "ls"}}],
        )
        req = _make_request()
        handler = MagicMock(return_value=resp)
        result = mw.wrap_model_call(req, handler)
        assert result is resp

    def test_no_user_question_skipped(self):
        mw = SelfEvaluationMiddleware()
        resp = _make_response(content=_long_text())
        req = MagicMock()
        req.messages = [SystemMessage(content="system prompt")]
        handler = MagicMock(return_value=resp)
        result = mw.wrap_model_call(req, handler)
        assert result is resp


class TestEvaluationFlow:
    @patch("src.agents.middlewares.self_evaluation_middleware.create_chat_model")
    def test_approve_returns_original(self, mock_create):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(
            content=json.dumps({"verdict": "approve", "gaps": [], "confidence": 0.9})
        )
        mock_create.return_value = mock_model

        mw = SelfEvaluationMiddleware()
        resp = _make_response(content=_long_text())
        req = _make_request()
        handler = MagicMock(return_value=resp)

        result = mw.wrap_model_call(req, handler)
        assert result is resp
        # Handler called once (initial call only)
        assert handler.call_count == 1

    @patch("src.agents.middlewares.self_evaluation_middleware.create_chat_model")
    def test_revise_calls_handler_again(self, mock_create):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(
            content=json.dumps({
                "verdict": "revise",
                "gaps": ["Missing examples", "No code"],
                "confidence": 0.8,
            })
        )
        mock_create.return_value = mock_model

        mw = SelfEvaluationMiddleware()
        original_resp = _make_response(content=_long_text())
        revised_resp = _make_response(content=_long_text() + " with examples and code")
        req = _make_request()

        call_count = [0]
        def handler(r):
            call_count[0] += 1
            if call_count[0] == 1:
                return original_resp
            return revised_resp

        result = mw.wrap_model_call(req, handler)
        # Handler called twice (original + revision)
        assert call_count[0] == 2
        assert result is revised_resp

    @patch("src.agents.middlewares.self_evaluation_middleware.create_chat_model")
    def test_revision_budget_respected(self, mock_create):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(
            content=json.dumps({"verdict": "revise", "gaps": ["gap1"], "confidence": 0.8})
        )
        mock_create.return_value = mock_model

        mw = SelfEvaluationMiddleware(max_revisions=1)

        # First call: uses revision
        resp1 = _make_response(content=_long_text())
        revised1 = _make_response(content=_long_text() + " revised")
        req1 = _make_request(user_question="question one")
        calls1 = [0]
        def handler1(r):
            calls1[0] += 1
            if calls1[0] == 1:
                return resp1
            return revised1
        mw.wrap_model_call(req1, handler1)
        assert calls1[0] == 2  # Original + revision

        # Second call with same question hash: budget exhausted
        resp2 = _make_response(content=_long_text() + " second attempt")
        req2 = _make_request(user_question="question one")
        handler2 = MagicMock(return_value=resp2)
        result2 = mw.wrap_model_call(req2, handler2)
        assert result2 is resp2
        assert handler2.call_count == 1  # No revision

    @patch("src.agents.middlewares.self_evaluation_middleware.create_chat_model")
    def test_revise_with_empty_gaps_skipped(self, mock_create):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(
            content=json.dumps({"verdict": "revise", "gaps": [], "confidence": 0.5})
        )
        mock_create.return_value = mock_model

        mw = SelfEvaluationMiddleware()
        resp = _make_response(content=_long_text())
        req = _make_request()
        handler = MagicMock(return_value=resp)

        result = mw.wrap_model_call(req, handler)
        assert result is resp
        assert handler.call_count == 1

    @patch("src.agents.middlewares.self_evaluation_middleware.create_chat_model")
    def test_evaluation_error_returns_original(self, mock_create):
        mock_model = MagicMock()
        mock_model.invoke.side_effect = RuntimeError("API error")
        mock_create.return_value = mock_model

        mw = SelfEvaluationMiddleware()
        resp = _make_response(content=_long_text())
        req = _make_request()
        handler = MagicMock(return_value=resp)

        result = mw.wrap_model_call(req, handler)
        assert result is resp

    @patch("src.agents.middlewares.self_evaluation_middleware.create_chat_model")
    def test_malformed_json_returns_original(self, mock_create):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="not json")
        mock_create.return_value = mock_model

        mw = SelfEvaluationMiddleware()
        resp = _make_response(content=_long_text())
        req = _make_request()
        handler = MagicMock(return_value=resp)

        result = mw.wrap_model_call(req, handler)
        assert result is resp

    @patch("src.agents.middlewares.self_evaluation_middleware.create_chat_model")
    def test_markdown_fenced_json_parsed(self, mock_create):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(
            content='```json\n{"verdict": "revise", "gaps": ["missing detail"], "confidence": 0.7}\n```'
        )
        mock_create.return_value = mock_model

        mw = SelfEvaluationMiddleware()
        resp = _make_response(content=_long_text())
        revised = _make_response(content=_long_text() + " detail added")
        req = _make_request()

        calls = [0]
        def handler(r):
            calls[0] += 1
            if calls[0] == 1:
                return resp
            return revised

        result = mw.wrap_model_call(req, handler)
        assert calls[0] == 2


class TestCaching:
    @patch("src.agents.middlewares.self_evaluation_middleware.create_chat_model")
    def test_same_response_uses_cache(self, mock_create):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(
            content=json.dumps({"verdict": "approve", "gaps": [], "confidence": 0.9})
        )
        mock_create.return_value = mock_model

        mw = SelfEvaluationMiddleware(max_revisions=5)
        resp = _make_response(content=_long_text())
        req = _make_request()

        handler = MagicMock(return_value=resp)
        mw.wrap_model_call(req, handler)
        mw.wrap_model_call(req, handler)

        # Evaluation model called once (second call uses cache)
        assert mock_model.invoke.call_count == 1


class TestReset:
    @patch("src.agents.middlewares.self_evaluation_middleware.create_chat_model")
    def test_reset_clears_revision_count(self, mock_create):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(
            content=json.dumps({"verdict": "revise", "gaps": ["gap"], "confidence": 0.8})
        )
        mock_create.return_value = mock_model

        mw = SelfEvaluationMiddleware(max_revisions=1)
        resp = _make_response(content=_long_text())
        revised = _make_response(content=_long_text() + " fixed")
        req = _make_request()

        # Use up budget
        calls = [0]
        def handler(r):
            calls[0] += 1
            if calls[0] == 1: return resp
            return revised
        mw.wrap_model_call(req, handler)

        # Budget exhausted
        handler2 = MagicMock(return_value=_make_response(content=_long_text() + " v2"))
        mw.wrap_model_call(req, handler2)
        assert handler2.call_count == 1

        # Reset and try again
        mw.reset()
        calls2 = [0]
        def handler3(r):
            calls2[0] += 1
            if calls2[0] == 1: return _make_response(content=_long_text() + " v3")
            return _make_response(content=_long_text() + " v3 revised")
        mw.wrap_model_call(req, handler3)
        assert calls2[0] == 2  # Revised again after reset
