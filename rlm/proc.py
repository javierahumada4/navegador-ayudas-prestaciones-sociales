"""Leave a process that started a vLLM engine.

vLLM keeps the interpreter alive at exit, and a bare ``os._exit`` orphans the engine process
(``VLLM::EngineCore``), which then holds the GPU and any pipe it inherited. On the DGX that hung
job 029 for two sessions. ``hard_exit`` kills the direct children first.
"""

from __future__ import annotations

import os
import signal
import sys


def _children(pid: int) -> list[int]:
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="utf-8") as stat:
                ppid = int(stat.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if ppid == pid:
            found.append(int(entry))
    return found


def hard_exit(code: int = 0) -> None:
    """Flush, SIGKILL every descendant (the vLLM engine), then exit without atexit hooks."""
    sys.stdout.flush()
    sys.stderr.flush()
    pending, seen = [os.getpid()], []
    while pending:
        for child in _children(pending.pop()):
            seen.append(child)
            pending.append(child)
    for child in reversed(seen):
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
    os._exit(code)
