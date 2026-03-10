"""Shared Scratchpad for concurrent read/write between lead agent and subagents.

File-based JSON storage with fcntl locking for thread-safe concurrent access.
All agents in the same thread share a scratchpad at:
  {thread_data_dir}/scratchpad.json

Critical: all read-modify-write operations hold the lock for the entire
duration to prevent race conditions between concurrent agents.
"""

import fcntl
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EntryType(str, Enum):
    FINDING = "finding"
    NOTE = "note"
    DATA = "data"
    URL = "url"
    CODE = "code"
    ERROR = "error"


@dataclass
class ScratchpadEntry:
    key: str
    value: Any
    entry_type: EntryType
    confidence: float = 1.0
    source: str = "unknown"
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


def _empty_data() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {"entries": {}, "agents": {}, "metadata": {"created_at": now, "last_modified": now}}


class FileScratchpad:
    """Thread-safe file-based JSON scratchpad with fcntl locking.

    All mutating operations (save, delete, clear) use _locked_update which
    holds an exclusive lock for the entire read-modify-write cycle.
    """

    def __init__(self, filepath: str | Path, lock_timeout: float = 10.0):
        self.filepath = Path(filepath)
        self.lock_timeout = lock_timeout

    def _acquire_lock(self, fd) -> bool:
        deadline = time.monotonic() + self.lock_timeout
        while time.monotonic() < deadline:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                time.sleep(0.05)
        return False

    def _locked_update(self, mutator) -> dict:
        """Atomic read-modify-write under exclusive lock.

        Opens file in r+ mode (or creates it), acquires lock, reads JSON,
        calls mutator(data), writes back, releases lock.  The lock is held
        for the entire cycle to prevent races between concurrent agents.
        """
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Create file if it doesn't exist
        if not self.filepath.exists():
            self.filepath.write_text(json.dumps(_empty_data(), indent=2))

        with open(self.filepath, "r+") as f:
            if not self._acquire_lock(f.fileno()):
                raise IOError(f"Lock timeout on {self.filepath} after {self.lock_timeout}s")
            try:
                # Read
                f.seek(0)
                raw = f.read()
                data = json.loads(raw) if raw.strip() else _empty_data()

                # Mutate
                mutator(data)

                # Write back
                data["metadata"]["last_modified"] = datetime.now(timezone.utc).isoformat()
                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2, default=str)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return data

    def _load_data(self) -> dict:
        """Read scratchpad data (shared lock for reads)."""
        if not self.filepath.exists():
            return _empty_data()
        try:
            with open(self.filepath) as f:
                if not self._acquire_lock(f.fileno()):
                    raise IOError(f"Lock timeout on {self.filepath}")
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load scratchpad %s: %s", self.filepath, e)
            return _empty_data()

    def save_entry(self, entry: ScratchpadEntry, agent_identity: str = "unknown"):
        """Save an entry atomically."""
        def mutate(data):
            data["entries"][entry.key] = {
                "key": entry.key,
                "value": entry.value,
                "entry_type": entry.entry_type.value,
                "confidence": entry.confidence,
                "source": entry.source,
                "timestamp": entry.timestamp,
                "metadata": entry.metadata,
                "agent_identity": agent_identity,
            }
            agents = data.setdefault("agents", {})
            now = datetime.now(timezone.utc).isoformat()
            if agent_identity not in agents:
                agents[agent_identity] = {"first_seen": now, "last_seen": now, "entries_count": 0}
            agents[agent_identity]["last_seen"] = now
            agents[agent_identity]["entries_count"] = agents[agent_identity].get("entries_count", 0) + 1

        self._locked_update(mutate)
        logger.debug("Saved entry '%s' to scratchpad (agent: %s)", entry.key, agent_identity)

    def read_entries(
        self,
        entry_type: EntryType | None = None,
        min_confidence: float = 0.0,
        agent_identity: str | None = None,
    ) -> list[ScratchpadEntry]:
        data = self._load_data()
        result = []
        for entry_dict in data.get("entries", {}).values():
            if entry_type and entry_dict.get("entry_type") != entry_type.value:
                continue
            if entry_dict.get("confidence", 1.0) < min_confidence:
                continue
            if agent_identity and entry_dict.get("agent_identity") != agent_identity:
                continue
            result.append(ScratchpadEntry(
                key=entry_dict["key"],
                value=entry_dict["value"],
                entry_type=EntryType(entry_dict["entry_type"]),
                confidence=entry_dict.get("confidence", 1.0),
                source=entry_dict.get("source", "unknown"),
                timestamp=entry_dict.get("timestamp", ""),
                metadata=entry_dict.get("metadata", {}),
            ))
        return result

    def get_entry(self, key: str) -> ScratchpadEntry | None:
        data = self._load_data()
        entry_dict = data.get("entries", {}).get(key)
        if entry_dict is None:
            return None
        return ScratchpadEntry(
            key=entry_dict["key"],
            value=entry_dict["value"],
            entry_type=EntryType(entry_dict["entry_type"]),
            confidence=entry_dict.get("confidence", 1.0),
            source=entry_dict.get("source", "unknown"),
            timestamp=entry_dict.get("timestamp", ""),
            metadata=entry_dict.get("metadata", {}),
        )

    def delete_entry(self, key: str) -> bool:
        deleted = [False]
        def mutate(data):
            if key in data.get("entries", {}):
                del data["entries"][key]
                deleted[0] = True
        self._locked_update(mutate)
        return deleted[0]

    def clear(self):
        def mutate(data):
            data["entries"] = {}
        self._locked_update(mutate)

    def get_stats(self) -> dict[str, Any]:
        data = self._load_data()
        entries = data.get("entries", {})
        confidences = [e.get("confidence", 1.0) for e in entries.values()]
        type_counts: dict[str, int] = {}
        for e in entries.values():
            t = e.get("entry_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "total_entries": len(entries),
            "agents": len(data.get("agents", {})),
            "created_at": data.get("metadata", {}).get("created_at"),
            "last_modified": data.get("metadata", {}).get("last_modified"),
            "entry_types": type_counts,
            "confidence_avg": sum(confidences) / len(confidences) if confidences else 0.0,
        }
