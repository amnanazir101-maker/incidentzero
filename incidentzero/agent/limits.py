"""Run limits loaded from configuration (R12).

The assignment fixes a hard internal budget in ``configs/limits.json``. Loading it
here means the controller, loop guard and retry policy all derive their bounds from
one file that the grader (or a viva demonstration) can change without touching code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_LIMITS_PATH = "configs/limits.json"


@dataclass(frozen=True)
class Limits:
    max_llm_calls: int = 14
    max_tool_calls: int = 28
    max_consecutive_model_retries: int = 3
    max_same_action_repeats: int = 2
    max_runtime_seconds: int = 120
    warning_llm_calls_remaining: int = 3
    #: Not part of the supplied config; bounds unproductive re-planning so the agent
    #: converges on a terminal action instead of revising forever.
    max_plan_revisions: int = 4
    #: Tool calls held back so a terminal verify/close or escalate always fits.
    terminal_tool_reserve: int = 3

    @classmethod
    def load(cls, path: str | Path = DEFAULT_LIMITS_PATH) -> "Limits":
        """Read limits from disk, ignoring unknown keys and falling back to defaults."""
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: int(v) for k, v in raw.items() if k in known and isinstance(v, (int, float))})
