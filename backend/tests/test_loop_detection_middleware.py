"""Tests for LoopDetectionMiddleware."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, SystemMessage

from src.agents.middlewares.loop_detection_middleware import (
    LoopDetectionMiddleware,
    _hash_tool_calls,
    _HARD_STOP_MSG,
    _WARNING_MSG,
)


def _make_state(tool_calls=None, content=""):
    """Build a minimal AgentState dict with an AIMessage."""
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    return {"messages": [msg]}


def _make_runtime(thread_id="default"):
    runtime = MagicMock()
    runtime.context = {"thread_id": thread_id}
    return runtime


def _bash_call(cmd="ls"):
    return {"name": "bash", "id": f"call_{cmd}", "args": {"command": cmd}}


class TestHashToolCalls:
    def test_same_calls_same_hash(self):
        a = _hash_tool_calls([_bash_call("ls")])
        b = _hash_tool_calls([_bash_call("ls")])
        assert a == b

    def test_different_calls_different_hash(self):
        a = _hash_tool_calls([_bash_call("ls")])
        b = _hash_tool_calls([_bash_call("pwd")])
        assert a != b

    def test_order_independent(self):
        a = _hash_tool_calls([_bash_call("ls"), {"name": "read_file", "args": {"path": "/tmp"}}])
        b = _hash_tool_calls([{"name": "read_file", "args": {"path": "/tmp"}}, _bash_call("ls")])
        assert a == b

    def test_empty_calls(self):
        h = _hash_tool_calls([])
        assert isinstance(h, str)
        assert len(h) > 0


class TestLoopDetection:
    def test_no_tool_calls_returns_none(self):
        mw = LoopDetectionMiddleware()
        runtime = _make_runtime()
        state = {"messages": [AIMessage(content="hello")]}
        result = mw.after_model(state, runtime)
        assert result is None

    def test_below_threshold_returns_none(self):
        mw = LoopDetectionMiddleware(warn_threshold=3)
        runtime = _make_runtime()
        call = [_bash_call("ls")]

        for _ in range(2):
            result = mw.after_model(_make_state(tool_calls=call), runtime)
            assert result is None

    def test_warn_at_threshold(self):
        mw = LoopDetectionMiddleware(warn_threshold=3, hard_limit=5)
        runtime = _make_runtime()
        call = [_bash_call("ls")]

        for _ in range(2):
            mw.after_model(_make_state(tool_calls=call), runtime)

        result = mw.after_model(_make_state(tool_calls=call), runtime)
        assert result is not None
        msgs = result["messages"]
        assert len(msgs) == 1
        assert isinstance(msgs[0], SystemMessage)
        assert "LOOP DETECTED" in msgs[0].content

    def test_hard_stop_at_limit(self):
        mw = LoopDetectionMiddleware(warn_threshold=2, hard_limit=4)
        runtime = _make_runtime()
        call = [_bash_call("ls")]

        for _ in range(3):
            mw.after_model(_make_state(tool_calls=call), runtime)

        result = mw.after_model(_make_state(tool_calls=call), runtime)
        assert result is not None
        msgs = result["messages"]
        assert len(msgs) == 1
        assert isinstance(msgs[0], AIMessage)
        assert msgs[0].tool_calls == []
        assert _HARD_STOP_MSG in msgs[0].content

    def test_hard_stop_handles_list_content(self):
        """Multimodal list content should not raise TypeError on hard stop."""
        mw = LoopDetectionMiddleware(warn_threshold=2, hard_limit=4)
        runtime = _make_runtime()
        call = [_bash_call("ls")]

        for _ in range(3):
            mw.after_model(_make_state(tool_calls=call), runtime)

        state = _make_state(tool_calls=call, content="")
        # Replace with list content
        state["messages"][-1] = AIMessage(
            content=[{"type": "text", "text": "Checking files..."}],
            tool_calls=call,
        )
        result = mw.after_model(state, runtime)
        assert result is not None
        stripped = result["messages"][0]
        assert stripped.tool_calls == []
        assert "Checking files" in stripped.content
        assert _HARD_STOP_MSG in stripped.content

    def test_different_calls_dont_trigger(self):
        mw = LoopDetectionMiddleware(warn_threshold=2)
        runtime = _make_runtime()

        for i in range(10):
            result = mw.after_model(_make_state(tool_calls=[_bash_call(f"cmd_{i}")]), runtime)
            assert result is None

    def test_window_sliding(self):
        mw = LoopDetectionMiddleware(warn_threshold=3, window_size=5)
        runtime = _make_runtime()
        call = [_bash_call("ls")]

        mw.after_model(_make_state(tool_calls=call), runtime)
        mw.after_model(_make_state(tool_calls=call), runtime)

        for i in range(5):
            mw.after_model(_make_state(tool_calls=[_bash_call(f"other_{i}")]), runtime)

        result = mw.after_model(_make_state(tool_calls=call), runtime)
        assert result is None

    def test_reset_clears_state(self):
        mw = LoopDetectionMiddleware(warn_threshold=2)
        runtime = _make_runtime()
        call = [_bash_call("ls")]

        mw.after_model(_make_state(tool_calls=call), runtime)
        mw.after_model(_make_state(tool_calls=call), runtime)

        mw.reset()
        result = mw.after_model(_make_state(tool_calls=call), runtime)
        assert result is None

    def test_non_ai_message_ignored(self):
        mw = LoopDetectionMiddleware()
        runtime = _make_runtime()
        state = {"messages": [SystemMessage(content="hello")]}
        result = mw.after_model(state, runtime)
        assert result is None

    def test_empty_messages_ignored(self):
        mw = LoopDetectionMiddleware()
        runtime = _make_runtime()
        result = mw.after_model({"messages": []}, runtime)
        assert result is None

    def test_auto_reset_on_new_run(self):
        """Cached middleware should reset when message count drops (new run)."""
        mw = LoopDetectionMiddleware(warn_threshold=3)
        runtime = _make_runtime()
        call = [_bash_call("ls")]

        # Build up history with growing message lists
        for i in range(2):
            state = {"messages": [AIMessage(content=f"m{j}", tool_calls=call) for j in range(i + 1)]}
            mw.after_model(state, runtime)

        # New run: fewer messages
        fresh_state = _make_state(tool_calls=call)
        result = mw.after_model(fresh_state, runtime)
        assert result is None  # reset happened, count is 1
