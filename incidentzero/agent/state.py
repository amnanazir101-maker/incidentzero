from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from incidentzero.domain.models import AgentPlan

#: Terminal statuses permitted by docs/CONTRACTS.md, plus the in-flight value.
RUNNING = "running"
TERMINAL_STATUSES = frozenset({"resolved", "escalated", "aborted", "budget_exhausted", "failed"})


@dataclass
class AgentState:
    """Explicit agent state that exists independently of the chat transcript (R1).

    The message list is what the model sees. This is what the controller reasons
    about: plan, evidence, world version, outcomes, loop state, budgets consumed
    and the terminal decision.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    plan: AgentPlan | None = None
    evidence_ids: list[str] = field(default_factory=list)
    latest_world_version: int | None = None
    last_tool_results: list[dict[str, Any]] = field(default_factory=list)
    repeated_actions: dict[str, int] = field(default_factory=dict)
    status: str = RUNNING

    # -- re-planning ------------------------------------------------------
    plan_revisions: int = 0
    replan_triggers: list[dict[str, Any]] = field(default_factory=list)

    # -- failure accounting -----------------------------------------------
    tool_retry_counts: dict[str, int] = field(default_factory=dict)
    model_retries: int = 0
    validation_failures: int = 0
    empty_replies: int = 0

    # -- verification (R10) -----------------------------------------------
    #: Evidence ID of the most recent verify_recovery result.
    latest_verify_evidence_id: str | None = None
    #: Whether that verification reported criteria_met.
    recovery_criteria_met: bool = False

    # -- observation ------------------------------------------------------

    def observe_result(self, result: dict[str, Any]) -> None:
        """Fold one tool observation into explicit state."""
        if not isinstance(result, dict):
            return
        evidence = result.get("evidence_id")
        if evidence:
            self.evidence_ids.append(evidence)
        version = result.get("world_version")
        if isinstance(version, int):
            self.latest_world_version = version
        if result.get("tool") == "verify_recovery" and result.get("status") == "ok":
            self._record_verification(result)
        self.last_tool_results.append(result)
        self.last_tool_results = self.last_tool_results[-8:]

    def _record_verification(self, result: dict[str, Any]) -> None:
        """Track the only evidence that can justify closing the incident (R10)."""
        data = result.get("data")
        self.latest_verify_evidence_id = result.get("evidence_id")
        self.recovery_criteria_met = bool(isinstance(data, dict) and data.get("criteria_met"))

    def invalidate_verification(self) -> None:
        """Drop verification proof after the world changes.

        A verify_recovery result describes the world at one version. Once an action
        mutates the world, that proof is stale and must not be reused to close.
        """
        self.latest_verify_evidence_id = None
        self.recovery_criteria_met = False

    # -- planning ---------------------------------------------------------

    def record_replan(self, plan: AgentPlan, trigger: dict[str, Any]) -> None:
        self.plan = plan
        self.plan_revisions = plan.revision
        self.replan_triggers.append(trigger)

    def mark_step_done(self) -> str | None:
        """Mark the first outstanding plan step as complete; return its id."""
        if self.plan is None:
            return None
        for step in self.plan.steps:
            if step.status != "done":
                step.status = "done"
                return step.step_id
        return None

    # -- reporting --------------------------------------------------------

    def summary(self) -> str:
        """Compact state description passed to the planner on a revision."""
        recent = [
            {"tool": r.get("tool"), "status": r.get("status"), "message": r.get("message")}
            for r in self.last_tool_results[-5:]
        ]
        return (
            f"world_version={self.latest_world_version}; "
            f"evidence={self.evidence_ids[-8:]}; "
            f"recovery_criteria_met={self.recovery_criteria_met}; "
            f"plan_revisions={self.plan_revisions}; "
            f"recent_results={recent}"
        )
