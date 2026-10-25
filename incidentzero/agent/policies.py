from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from incidentzero.domain.models import RiskLevel


class RiskPolicy:
    def __init__(self, config_path: str | Path = "configs/risk_policy.json") -> None:
        self.mapping = json.loads(Path(config_path).read_text(encoding="utf-8"))

    def risk(self, tool_name: str) -> RiskLevel:
        return RiskLevel(self.mapping.get(tool_name, "critical"))

    def requires_human_approval(self, tool_name: str) -> bool:
        return self.risk(tool_name) in {RiskLevel.HIGH, RiskLevel.CRITICAL}

    def is_consequential(self, tool_name: str) -> bool:
        """True when the tool mutates or attempts to mutate the world.

        Observation tools are low risk by policy, so a failure there is a
        corrective signal to the model rather than a failure of the plan.
        """
        return self.risk(tool_name) is not RiskLevel.LOW


class ControlDecision(str, Enum):
    """What the controller should do next after one tool observation.

    R5 forbids collapsing every failure into "retry". Each member maps to a
    distinct branch in the controller loop.
    """

    CONTINUE = "continue"   # normal progress, keep executing the current plan
    RETRY = "retry"         # transient fault; bounded re-attempt of the same operation
    CORRECT = "correct"     # the model's request was malformed; return a corrective observation
    REPLAN = "replan"       # the plan itself is wrong or blocked; revise it
    ABORT = "abort"         # unsafe or impossible to continue


#: Statuses that mean "the world changed between observation and action" (R6).
STALE_STATUSES = frozenset({"stale_precondition"})
#: Statuses that mean "a human refused this action" (R7).
DENIED_STATUSES = frozenset({"approval_denied"})
#: Statuses produced by our own validation boundary before the simulator is reached (R8).
VALIDATION_STATUSES = frozenset({
    "validation_error", "invalid_arguments", "unknown_tool",
    # A high-risk action proposed with no supporting observation (R3). The plan is
    # not necessarily wrong; the model must gather evidence before asking again.
    "insufficient_evidence",
})
#: Statuses that mean "the backend hiccuped, the request itself was fine".
TRANSIENT_STATUSES = frozenset({"transient_error", "timeout", "rate_limited"})
#: Statuses that mean "the loop guard blocked an unproductive repeat" (R9).
LOOP_STATUSES = frozenset({"loop_blocked"})


class ReplanPolicy:
    """Classifies one tool observation into a control decision.

    ``should_replan`` is the stable public contract; ``classify`` is the richer
    signal the controller uses so that retry, re-observation, correction and
    revision stay distinct code paths.
    """

    def __init__(self, risk: RiskPolicy | None = None) -> None:
        self.risk = risk

    # -- classification ---------------------------------------------------

    def classify(self, tool_result: dict[str, Any]) -> ControlDecision:
        if not isinstance(tool_result, dict):
            return ControlDecision.CORRECT

        status = str(tool_result.get("status", "")).lower()
        retryable = bool(tool_result.get("retryable", False))

        # A stale world is new information, never a reason to re-fire blindly (R6, Q20).
        if status in STALE_STATUSES:
            return ControlDecision.REPLAN

        # The human said no. The action must not run; the plan must change (R7).
        if status in DENIED_STATUSES:
            return ControlDecision.REPLAN

        # We blocked the call ourselves. The model needs a correction, not a new plan.
        if status in VALIDATION_STATUSES:
            return ControlDecision.CORRECT

        # Repetition is unproductive, but how unproductive depends on what was
        # repeated (R9). Re-reading a metric the agent already holds is redundancy:
        # the model needs telling to move on, not a new hypothesis. Repeating a
        # consequential action that did not resolve the incident really does mean
        # the active hypothesis is wrong.
        if status in LOOP_STATUSES:
            if self._is_observation(tool_result):
                return ControlDecision.CORRECT
            return ControlDecision.REPLAN

        # Backend fault with the request intact: bounded retry (R5).
        if status in TRANSIENT_STATUSES or (retryable and status != "ok"):
            return ControlDecision.RETRY

        if status == "ok":
            # An API success is not an incident success (R10, Example 4).
            if self._verification_failed(tool_result):
                return ControlDecision.REPLAN
            return ControlDecision.CONTINUE

        # Remaining case: a non-retryable failure.
        if self._is_observation(tool_result):
            # A bad observation call is a model mistake we can correct cheaply.
            return ControlDecision.CORRECT
        return ControlDecision.REPLAN

    def _is_observation(self, tool_result: dict[str, Any]) -> bool:
        """True when the tool only reads state, per the risk policy on disk.

        Without a risk policy the answer is False, so an unknown tool is treated
        as consequential and handled at plan level rather than waved through.
        """
        tool = tool_result.get("tool")
        if self.risk is None or not isinstance(tool, str):
            return False
        return not self.risk.is_consequential(tool)

    def requires_reobservation(self, tool_result: dict[str, Any]) -> bool:
        """True when the plan must be re-grounded in fresh state before acting again.

        A stale precondition means our observation is obsolete. Q20 forbids simply
        substituting the newer world_version and re-firing the same action, so the
        controller re-observes first and only then re-confirms or revises.
        """
        return str(tool_result.get("status", "")).lower() in STALE_STATUSES

    @staticmethod
    def _verification_failed(tool_result: dict[str, Any]) -> bool:
        """True when the simulator reports the recovery criteria are still unmet."""
        data = tool_result.get("data")
        if not isinstance(data, dict) or "criteria_met" not in data:
            return False
        return not bool(data["criteria_met"])

    # -- stable contract --------------------------------------------------

    def should_replan(self, tool_result: dict[str, Any]) -> bool:
        return self.classify(tool_result) is ControlDecision.REPLAN

    def replan_trigger(self, tool_result: dict[str, Any]) -> dict[str, Any]:
        """Structured reason for the trace and for Planner.revise (R13)."""
        decision = self.classify(tool_result)
        return {
            "decision": decision.value,
            "status": tool_result.get("status"),
            "tool": tool_result.get("tool"),
            "message": tool_result.get("message"),
            "world_version": tool_result.get("world_version"),
            "evidence_id": tool_result.get("evidence_id"),
            "verification_failed": self._verification_failed(tool_result),
            "requires_reobservation": self.requires_reobservation(tool_result),
        }


#: Argument keys that do not change what an action *does* to the world.
#:
#: Every action tool carries ``expected_world_version`` and a free-text ``reason``.
#: Including them in the fingerprint would make a retry of the same action under a
#: bumped version look like a brand-new action, which is precisely the loop R9 and
#: Q20 are aimed at. They are therefore excluded so repetition stays detectable.
NON_SEMANTIC_ARG_KEYS = frozenset({
    "expected_world_version",
    "reason",
    "summary",
    "evidence_ids",
})


class LoopGuard:
    """Detects unproductive repetition of the same action (R9).

    Repetition is counted per *semantic* fingerprint, not per raw argument dict,
    so re-firing an action with a refreshed world version or a reworded
    justification still counts as a repeat.
    """

    def __init__(self, max_same_action_repeats: int = 2) -> None:
        self.max_same_action_repeats = max_same_action_repeats
        self._counts: dict[str, int] = {}

    # -- fingerprinting ---------------------------------------------------

    @staticmethod
    def fingerprint(action_name: str, arguments: dict[str, Any]) -> str:
        """Stable identity for an action, independent of non-semantic arguments.

        Keys are sorted so that argument ordering never changes the fingerprint.
        """
        args = arguments if isinstance(arguments, dict) else {}
        semantic = {k: v for k, v in args.items() if k not in NON_SEMANTIC_ARG_KEYS}
        try:
            encoded = json.dumps(semantic, sort_keys=True, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            encoded = repr(sorted((str(k), str(v)) for k, v in semantic.items()))
        return f"{action_name}:{encoded}"

    # -- stable contract --------------------------------------------------

    def record(self, action_name: str, arguments: dict[str, Any]) -> bool:
        """Record one attempt. Return True when it has repeated too often."""
        key = self.fingerprint(action_name, arguments)
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key] > self.max_same_action_repeats

    # -- inspection -------------------------------------------------------

    def count(self, action_name: str, arguments: dict[str, Any]) -> int:
        """How many times this action has already been attempted."""
        return self._counts.get(self.fingerprint(action_name, arguments), 0)

    def would_block(self, action_name: str, arguments: dict[str, Any]) -> bool:
        """Peek at whether the next attempt would trip the guard, without recording."""
        return self.count(action_name, arguments) >= self.max_same_action_repeats

    # -- reinterpretation -------------------------------------------------

    def note_material_change(self, action_name: str | None = None, arguments: dict[str, Any] | None = None) -> None:
        """Forget repetition history because the situation genuinely changed.

        R9 asks that repetition be *reinterpreted* when new evidence materially
        changes the world. The controller calls this on a grounded plan revision:
        under a new hypothesis the same action may be justified for a new reason.
        Called with no arguments it clears every counter; with an action it clears
        only that one.
        """
        if action_name is None:
            self._counts.clear()
            return
        self._counts.pop(self.fingerprint(action_name, arguments or {}), None)

    def repeated_actions(self) -> dict[str, int]:
        """Snapshot of fingerprints seen more than once, for the trace (R13)."""
        return {k: v for k, v in self._counts.items() if v > 1}
