"""Durable, unredacted per-case audit trails for phase-2 model evaluations.

Each event is JSONL and flushed immediately (one independent file per case).
These traces can contain sensitive financial/person data and full model prompts.
Keep them local, private and excluded from Git.
"""
from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> str:
    return str(value)


class TraceRecorder:
    """Append-only writer. A run crash leaves every completed event on disk."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._stream = None
        self._sequence = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # O_EXCL prevents silently overwriting existing evidence or following
        # a symlink to an existing file.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._stream = os.fdopen(fd, "w", encoding="utf-8", buffering=1)
        return self

    def event(self, event: str, **data: Any) -> None:
        if self._stream is None:
            raise RuntimeError("TraceRecorder must be used as a context manager")
        self._sequence += 1
        payload = {
            "sequence": self._sequence,
            "utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        }
        self._stream.write(json.dumps(
            payload, ensure_ascii=False, default=_json_default,
        ) + "\n")
        self._stream.flush()

    def __exit__(self, exc_type, exc, traceback):
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def create_artifact_run(output_path: str | Path) -> Path:
    """Make a collision-safe, persistent directory for traces and TXT reports."""
    output = Path(output_path)
    root = output.parent / "phase2_artifacts" / output.stem
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for _ in range(10):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run = root / f"{stamp}-{secrets.token_hex(4)}"
        try:
            run.mkdir(mode=0o700)
        except FileExistsError:
            continue
        for part in ("traces", "reports"):
            (run / part).mkdir(mode=0o700)
        return run.resolve()
    raise RuntimeError("Unable to allocate a unique artifact directory")


def case_trace_path(run: str | Path, mode: str, number: int, case_id: str) -> Path:
    """Predictable case basename, unique within an evaluation execution."""
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", str(case_id))[:60] or "case"
    stage = re.sub(r"[^a-zA-Z0-9_-]", "_", mode)[:40] or "evaluation"
    return Path(run) / "traces" / f"{stage}-{number:03d}-{slug}.jsonl"
