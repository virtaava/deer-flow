"""Tests for the shared scratchpad storage backend and tools."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tools.builtins.shared_scratchpad import (
    EntryType,
    FileScratchpad,
    ScratchpadEntry,
)


@pytest.fixture
def scratchpad(tmp_path):
    return FileScratchpad(tmp_path / "scratchpad.json")


def _entry(key="test", value="hello", entry_type=EntryType.FINDING, confidence=0.9):
    return ScratchpadEntry(key=key, value=value, entry_type=entry_type, confidence=confidence, source="test")


class TestFileScratchpad:
    def test_save_and_read(self, scratchpad):
        scratchpad.save_entry(_entry(key="a", value="first"), "agent1")
        entries = scratchpad.read_entries()
        assert len(entries) == 1
        assert entries[0].key == "a"
        assert entries[0].value == "first"

    def test_overwrite_same_key(self, scratchpad):
        scratchpad.save_entry(_entry(key="a", value="v1"), "agent1")
        scratchpad.save_entry(_entry(key="a", value="v2"), "agent1")
        entries = scratchpad.read_entries()
        assert len(entries) == 1
        assert entries[0].value == "v2"

    def test_multiple_entries(self, scratchpad):
        scratchpad.save_entry(_entry(key="a"), "agent1")
        scratchpad.save_entry(_entry(key="b"), "agent2")
        scratchpad.save_entry(_entry(key="c"), "agent1")
        entries = scratchpad.read_entries()
        assert len(entries) == 3

    def test_filter_by_type(self, scratchpad):
        scratchpad.save_entry(_entry(key="a", entry_type=EntryType.FINDING), "a1")
        scratchpad.save_entry(_entry(key="b", entry_type=EntryType.NOTE), "a1")
        scratchpad.save_entry(_entry(key="c", entry_type=EntryType.FINDING), "a1")
        findings = scratchpad.read_entries(entry_type=EntryType.FINDING)
        assert len(findings) == 2

    def test_filter_by_confidence(self, scratchpad):
        scratchpad.save_entry(_entry(key="a", confidence=0.3), "a1")
        scratchpad.save_entry(_entry(key="b", confidence=0.8), "a1")
        scratchpad.save_entry(_entry(key="c", confidence=0.95), "a1")
        high = scratchpad.read_entries(min_confidence=0.7)
        assert len(high) == 2

    def test_filter_by_agent(self, scratchpad):
        scratchpad.save_entry(_entry(key="a"), "lead")
        scratchpad.save_entry(_entry(key="b"), "subagent:1")
        scratchpad.save_entry(_entry(key="c"), "lead")
        lead_entries = scratchpad.read_entries(agent_identity="lead")
        assert len(lead_entries) == 2

    def test_get_entry(self, scratchpad):
        scratchpad.save_entry(_entry(key="target", value="found_it"), "a1")
        entry = scratchpad.get_entry("target")
        assert entry is not None
        assert entry.value == "found_it"

    def test_get_entry_missing(self, scratchpad):
        assert scratchpad.get_entry("nonexistent") is None

    def test_delete_entry(self, scratchpad):
        scratchpad.save_entry(_entry(key="del_me"), "a1")
        assert scratchpad.delete_entry("del_me") is True
        assert scratchpad.get_entry("del_me") is None

    def test_delete_missing(self, scratchpad):
        assert scratchpad.delete_entry("nonexistent") is False

    def test_clear(self, scratchpad):
        scratchpad.save_entry(_entry(key="a"), "a1")
        scratchpad.save_entry(_entry(key="b"), "a1")
        scratchpad.clear()
        assert scratchpad.read_entries() == []

    def test_stats(self, scratchpad):
        scratchpad.save_entry(_entry(key="a", entry_type=EntryType.FINDING, confidence=0.8), "agent1")
        scratchpad.save_entry(_entry(key="b", entry_type=EntryType.NOTE, confidence=0.6), "agent2")
        stats = scratchpad.get_stats()
        assert stats["total_entries"] == 2
        assert stats["agents"] == 2
        assert stats["entry_types"]["finding"] == 1
        assert stats["entry_types"]["note"] == 1
        assert 0.6 < stats["confidence_avg"] < 0.8

    def test_concurrent_access(self, tmp_path):
        """Two scratchpad instances writing to the same file."""
        path = tmp_path / "shared.json"
        sp1 = FileScratchpad(path)
        sp2 = FileScratchpad(path)

        sp1.save_entry(_entry(key="from_sp1"), "agent1")
        sp2.save_entry(_entry(key="from_sp2"), "agent2")

        # Both entries should be present
        entries = sp1.read_entries()
        keys = {e.key for e in entries}
        assert keys == {"from_sp1", "from_sp2"}

    def test_empty_scratchpad_read(self, scratchpad):
        entries = scratchpad.read_entries()
        assert entries == []

    def test_empty_scratchpad_stats(self, scratchpad):
        stats = scratchpad.get_stats()
        assert stats["total_entries"] == 0
        assert stats["confidence_avg"] == 0.0

    def test_agent_tracking(self, scratchpad):
        scratchpad.save_entry(_entry(key="a"), "lead")
        scratchpad.save_entry(_entry(key="b"), "lead")
        scratchpad.save_entry(_entry(key="c"), "sub:1")
        stats = scratchpad.get_stats()
        assert stats["agents"] == 2


class TestScratchpadEntry:
    def test_timestamp_auto_set(self):
        entry = ScratchpadEntry(key="x", value="y", entry_type=EntryType.NOTE)
        assert entry.timestamp != ""

    def test_metadata_default_empty(self):
        entry = ScratchpadEntry(key="x", value="y", entry_type=EntryType.NOTE)
        assert entry.metadata == {}


class TestScratchpadTools:
    def test_save_finding_tool_backend(self, tmp_path):
        """Test save_finding via the backend directly (tool requires ToolRuntime injection)."""
        sp = FileScratchpad(tmp_path / "scratchpad.json")
        sp.save_entry(ScratchpadEntry(key="test_key", value="test_value", entry_type=EntryType.FINDING, confidence=0.9, source="test"), "test-agent")
        entry = sp.get_entry("test_key")
        assert entry is not None
        assert entry.value == "test_value"
        assert entry.confidence == 0.9

    @patch("src.tools.builtins.scratchpad_tools.get_paths")
    def test_read_findings_empty(self, mock_get_paths, tmp_path):
        from src.tools.builtins.scratchpad_tools import read_findings_tool

        mock_paths = MagicMock()
        mock_paths.sandbox_work_dir.return_value = tmp_path
        mock_get_paths.return_value = mock_paths

        # Just verify the backend works for empty reads
        sp = FileScratchpad(tmp_path / "scratchpad.json")
        assert sp.read_entries() == []
