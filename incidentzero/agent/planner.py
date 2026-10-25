from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from incidentzero.domain.models import AgentPlan, PlanStep
from incidentzero.model.base import ModelClient
from incidentzero.model.errors import ModelError


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "rationale_summary": {"type": "string"},
        "steps": {
            "type": "array",
            "minItems": 2,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "string"},
                    "objective": {"type": "string"},
                    "success_signal": {"type": "string"},
                },
                "required": ["step_id", "objective", "success_signal"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["hypothesis", "rationale_summary", "steps"],
    "additionalProperties": False,
}

PLAN_VALIDATOR = Draft202012Validator(PLAN_SCHEMA)

STEP_DONE = "done"
STEP_PENDING = "pending"

#: Deterministic replacement hypotheses, keyed by why the previous plan stalled.
#: Used when the model is unavailable, over budget, or returns an unusable plan.
FALLBACK_HYPOTHESIS = {
    "approval_denied": (
        "The proposed remediation was refused by a human approver, so the active "
        "hypothesis cannot be acted on as planned. A lower-risk mitigation or an "
        "evidence-backed escalation is required."
    ),
    "stale_precondition": (
        "The environment changed after it was observed, so the evidence behind the "
        "active hypothesis is out of date and must be re-established before acting."
    ),
    "insufficient_evidence": (
        "A consequential action was proposed without supporting observations. The "
        "hypothesis must be grounded in simulator evidence before remediation."
    ),
    "loop_blocked": (
        "The same action has been repeated without progress, so the active hypothesis "
        "is not producing recovery and a different explanation must be tested."
    ),
    "verification_failed": (
        "Remediation reported success but verify_recovery still fails, so the active "
        "hypothesis does not explain the customer-facing impact."
    ),
    "budget_low": (
        "The remaining call budget is too small to continue investigating safely. The "
        "plan must converge on verification or escalation."
    ),
}

DEFAULT_FALLBACK_HYPOTHESIS = (
    "The previous hypothesis is not supported by the latest evidence and must be replaced."
)


class Planner:
    """Creates and revises the explicit plan (R2, R4).

    Python owns the revision counter and the record of rejected hypotheses. The
    model may propose the content of a plan; it never decides that a revision
    happened, and it never sees its own failed hypothesis presented as viable.
    """

    def __init__(self, model: ModelClient, trace: Any | None = None) -> None:
        self.model = model
        self.trace = trace
        #: Hypotheses already tried and discarded, so a revision cannot loop back.
        self.rejected_hypotheses: list[str] = []

    # -- creation ---------------------------------------------------------

    def create(self, incident_observation: dict[str, Any], context: list[dict[str, Any]] | None = None) -> AgentPlan:
        """Create an explicit initial plan grounded in a real observation (R2)."""
        messages = [
            {"role": "system", "content": (
                "Create a short SRE investigation-and-remediation plan. "
                "Do not assume the ticket's suspected root cause is correct. "
                "Include at least one investigation subgoal, one remediation subgoal, "
                "and a final verification step. Each step needs an observable success signal."
            )},
            {"role": "user", "content": f"Incident observation: {incident_observation}"},
        ]
        if context:
            messages.append({"role": "user", "content": f"Additional evidence: {context}"})

        try:
            raw = self.model.structured(messages, "incident_plan", PLAN_SCHEMA)
        except (ModelError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            self._record("plan_fallback", {"stage": "create", "reason": str(exc)})
            return self._bootstrap_plan(incident_observation)

        plan = self._plan_from_payload(raw)
        if plan is None:
            self._record("plan_fallback", {"stage": "create", "reason": "model plan failed schema validation"})
            return self._bootstrap_plan(incident_observation)
        return plan

    # -- revision ---------------------------------------------------------

    def revise(
        self,
        current: AgentPlan,
        trigger: dict[str, Any],
        state_summary: str,
        use_model: bool = True,
    ) -> AgentPlan:
        """Produce a grounded successor plan (R4, Task E).

        Guarantees, regardless of what the model does:

        * the revision number always increases;
        * completed steps are preserved, the rest are replaced;
        * the superseded hypothesis is recorded and cannot be proposed again;
        * a revision that does not actually change anything is rejected in favour
          of a deterministic replacement, so a failed plan is never repeated.

        ``use_model=False`` produces the revision locally, which the controller
        uses when the LLM budget is too low to spend a call on re-planning.
        """
        superseded = (current.hypothesis or "").strip()
        if superseded and superseded not in self.rejected_hypotheses:
            self.rejected_hypotheses.append(superseded)

        candidate: AgentPlan | None = None
        if use_model:
            try:
                raw = self.model.structured(
                    self._revision_messages(current, trigger, state_summary),
                    "incident_plan_revision",
                    PLAN_SCHEMA,
                )
                candidate = self._plan_from_payload(raw)
            except (ModelError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                # Re-planning must never be the thing that kills the run.
                self._record("plan_revision_model_failed", {"error": str(exc)})
                candidate = None

        if candidate is not None and self._is_material_change(current, candidate):
            revised = candidate
        else:
            if candidate is not None:
                self._record("plan_revision_rejected", {
                    "reason": "model repeated a rejected or unchanged plan",
                    "hypothesis": candidate.hypothesis,
                })
            revised = self._fallback_revision(current, trigger)

        revised.steps = self._merge_steps(current, revised)
        revised.revision = current.revision + 1
        self._record("plan_revised", {
            "revision": revised.revision,
            "trigger": trigger,
            "superseded_hypothesis": superseded,
            "new_hypothesis": revised.hypothesis,
            "preserved_steps": [s.step_id for s in revised.steps if s.status == STEP_DONE],
        })
        return revised

    # -- helpers ----------------------------------------------------------

    def _revision_messages(self, current: AgentPlan, trigger: dict[str, Any], state_summary: str) -> list[dict[str, Any]]:
        completed = [s.objective for s in current.steps if s.status == STEP_DONE]
        return [
            {"role": "system", "content": (
                "You are revising an SRE incident plan that has stalled. "
                "Produce a DIFFERENT working hypothesis or a materially different set of "
                "remaining steps. Do not restate a hypothesis that has already been rejected. "
                "Keep what the evidence has already established. If the evidence shows no safe "
                "remedy remains, plan for an evidence-backed escalation instead."
            )},
            {"role": "user", "content": (
                f"Current hypothesis: {current.hypothesis}\n"
                f"Revision number so far: {current.revision}\n"
                f"Already rejected hypotheses: {self.rejected_hypotheses}\n"
                f"Completed work to preserve: {completed or '(none)'}\n"
                f"Why the plan stalled: {trigger}\n"
                f"Observed state: {state_summary}"
            )},
        ]

    def _plan_from_payload(self, raw: Any) -> AgentPlan | None:
        """Validate model output against PLAN_SCHEMA before trusting it (R8)."""
        if not isinstance(raw, dict):
            return None
        if next(PLAN_VALIDATOR.iter_errors(raw), None) is not None:
            return None
        try:
            steps = [
                PlanStep(
                    step_id=str(row["step_id"]),
                    objective=str(row["objective"]),
                    success_signal=str(row["success_signal"]),
                )
                for row in raw["steps"]
            ]
        except (KeyError, TypeError, ValueError):
            return None
        return AgentPlan(
            hypothesis=str(raw["hypothesis"]),
            steps=steps,
            rationale_summary=str(raw.get("rationale_summary", "")),
        )

    def _is_material_change(self, current: AgentPlan, candidate: AgentPlan) -> bool:
        """Reject a 'revision' that repeats a failed plan (Task E)."""
        new_hypothesis = (candidate.hypothesis or "").strip().casefold()
        if not new_hypothesis:
            return False
        if new_hypothesis in {h.strip().casefold() for h in self.rejected_hypotheses}:
            return False
        if new_hypothesis != (current.hypothesis or "").strip().casefold():
            return True
        current_objectives = [s.objective.strip().casefold() for s in current.steps]
        candidate_objectives = [s.objective.strip().casefold() for s in candidate.steps]
        return candidate_objectives != current_objectives

    def _fallback_revision(self, current: AgentPlan, trigger: dict[str, Any]) -> AgentPlan:
        """Deterministic revision used when the model cannot supply a usable one."""
        key = self._trigger_key(trigger)
        hypothesis = FALLBACK_HYPOTHESIS.get(key, DEFAULT_FALLBACK_HYPOTHESIS)
        rev = current.revision + 1

        if key == "budget_low":
            steps = [
                PlanStep(f"r{rev}-1", "Run verify_recovery to establish the objective state.", "verify_recovery returns a result."),
                PlanStep(f"r{rev}-2", "Close if criteria_met is true, otherwise escalate with the evidence gathered.", "close_incident or escalate_incident returns status ok."),
            ]
        elif key == "stale_precondition":
            steps = [
                PlanStep(f"r{rev}-1", "Re-observe health and metrics for the affected service.", "Fresh observation carrying a current world_version."),
                PlanStep(f"r{rev}-2", "Decide whether the previous action is still justified by the new evidence.", "Action is either re-confirmed or withdrawn."),
                PlanStep(f"r{rev}-3", "Verify recovery once the chosen action completes.", "verify_recovery reports criteria_met."),
            ]
        elif key == "approval_denied":
            steps = [
                PlanStep(f"r{rev}-1", "Identify a lower-risk mitigation supported by the evidence already gathered.", "A medium or low risk action is selected."),
                PlanStep(f"r{rev}-2", "If no safe alternative exists, escalate citing the denied action and its evidence.", "escalate_incident returns status ok."),
            ]
        else:
            steps = [
                PlanStep(f"r{rev}-1", "Gather evidence that discriminates between the remaining plausible causes.", "New metrics, logs, deployment or dependency evidence."),
                PlanStep(f"r{rev}-2", "Apply the least risky remediation the new evidence supports.", "Action returns status ok."),
                PlanStep(f"r{rev}-3", "Verify recovery, then close or escalate with evidence.", "verify_recovery reports criteria_met, then close_incident succeeds."),
            ]

        return AgentPlan(
            hypothesis=hypothesis,
            steps=steps,
            rationale_summary=f"Locally derived revision after: {trigger.get('status') or key}.",
        )

    @staticmethod
    def _trigger_key(trigger: dict[str, Any]) -> str:
        if not isinstance(trigger, dict):
            return "unknown"
        if trigger.get("verification_failed"):
            return "verification_failed"
        if trigger.get("budget_low"):
            return "budget_low"
        return str(trigger.get("status") or trigger.get("decision") or "unknown")

    @staticmethod
    def _merge_steps(current: AgentPlan, revised: AgentPlan) -> list[PlanStep]:
        """Carry completed work forward; retire the rest (R4: preserve evidence)."""
        preserved = [
            PlanStep(s.step_id, s.objective, s.success_signal, STEP_DONE)
            for s in current.steps if s.status == STEP_DONE
        ]
        existing_ids = {s.step_id for s in preserved}
        fresh = [s for s in revised.steps if s.step_id not in existing_ids]
        return preserved + fresh

    def _bootstrap_plan(self, incident_observation: dict[str, Any]) -> AgentPlan:
        """Minimal valid plan so a bad first model reply cannot strand the run."""
        data = incident_observation.get("data") if isinstance(incident_observation, dict) else None
        suspected = (data or {}).get("suspected_service", "the affected service")
        return AgentPlan(
            hypothesis=(
                f"The incident ticket suspects {suspected}, but this is unverified and must be "
                f"tested against simulator evidence before any remediation."
            ),
            steps=[
                PlanStep("b1", f"Observe health, metrics and logs for {suspected} and its dependencies.", "Observations identify an abnormal component."),
                PlanStep("b2", "Apply the least risky remediation the evidence supports.", "Action returns status ok."),
                PlanStep("b3", "Verify recovery, then close or escalate with evidence.", "verify_recovery reports criteria_met, then close_incident succeeds."),
            ],
            rationale_summary="Bootstrap plan: the ticket is treated as a lead, not as proof.",
        )

    def _record(self, event: str, payload: dict[str, Any]) -> None:
        if self.trace is not None:
            self.trace.record(event, payload)
