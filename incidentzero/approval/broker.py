"""Python-side authorization for consequential actions (R3, R7).

The model may argue for an action. It can never grant one. Every high/critical
tool call passes through :class:`ApprovalBroker`, which decides -- from the risk
policy read live off disk -- whether a human must be asked, checks that real
simulator evidence backs the request, calls the supplied gateway, and records
both the request and its outcome.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from incidentzero.agent.policies import RiskPolicy
from incidentzero.domain.models import RiskLevel

from .gateway import ApprovalGateway

#: Outcome codes for ApprovalDecision.reason.
NOT_REQUIRED = "not_required"
GRANTED = "granted"
DENIED = "denied"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(slots=True)
class ApprovalDecision:
    tool: str
    risk: RiskLevel
    required: bool
    granted: bool
    reason: str
    justification: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    message: str = ""

    @property
    def blocked(self) -> bool:
        """True when the action must not reach the simulator."""
        return not self.granted


class ApprovalBroker:
    """Gates consequential actions behind evidence and a human decision."""

    def __init__(
        self,
        gateway: ApprovalGateway,
        risk: RiskPolicy | None = None,
        trace: Any | None = None,
        min_evidence: int = 1,
        require_evidence: bool = True,
    ) -> None:
        self.gateway = gateway
        self.risk = risk or RiskPolicy()
        self.trace = trace
        self.min_evidence = min_evidence
        self.require_evidence = require_evidence
        #: Audit log of every decision, for the report and the trace (R13).
        self.decisions: list[ApprovalDecision] = []

    # -- evidence ---------------------------------------------------------

    @staticmethod
    def relevant_evidence(
        arguments: dict[str, Any],
        observations: Sequence[dict[str, Any]],
    ) -> list[str]:
        """Evidence IDs of successful observations that bear on this action.

        R3: "a model sentence is not evidence". Only simulator results count, and
        only those about the targeted service or about the incident as a whole.
        """
        target = arguments.get("service") if isinstance(arguments, dict) else None
        found: list[str] = []
        for obs in observations:
            if not isinstance(obs, dict) or obs.get("status") != "ok":
                continue
            evidence_id = obs.get("evidence_id")
            if not evidence_id:
                continue
            data = obs.get("data")
            subject = data.get("service") if isinstance(data, dict) else None
            # Global evidence (incident ticket, recovery check, runbook) has no
            # service field and is relevant to any action.
            if subject is None or target is None or subject == target:
                found.append(evidence_id)
        return found

    def build_justification(
        self,
        tool: str,
        arguments: dict[str, Any],
        evidence_ids: Sequence[str],
    ) -> str:
        """Human-readable case for the action, shown to the approver.

        The model's own ``reason`` is included but clearly labelled as a claim,
        never as authority.
        """
        claim = arguments.get("reason") if isinstance(arguments, dict) else None
        lines = [
            f"Action: {tool} (risk={self.risk.risk(tool).value})",
            f"Arguments: {arguments}",
            f"Model's stated reason: {claim or '(none supplied)'}",
            f"Supporting simulator evidence: {', '.join(evidence_ids) or '(none)'}",
        ]
        return "\n".join(lines)

    # -- decision ---------------------------------------------------------

    def evaluate(
        self,
        tool: str,
        arguments: dict[str, Any],
        observations: Sequence[dict[str, Any]] = (),
    ) -> ApprovalDecision:
        risk = self.risk.risk(tool)

        if not self.risk.requires_human_approval(tool):
            decision = ApprovalDecision(
                tool=tool, risk=risk, required=False, granted=True, reason=NOT_REQUIRED,
            )
            self.decisions.append(decision)
            return decision

        evidence_ids = self.relevant_evidence(arguments, observations)

        if self.require_evidence and len(evidence_ids) < self.min_evidence:
            decision = ApprovalDecision(
                tool=tool, risk=risk, required=True, granted=False,
                reason=INSUFFICIENT_EVIDENCE, evidence_ids=evidence_ids,
                message=(
                    f"{tool} is {risk.value} risk and was proposed without supporting "
                    f"observations. Gather relevant evidence "
                    f"(metrics, logs, deployments, dependencies) before requesting it."
                ),
            )
            self._record("approval_skipped_no_evidence", decision)
            self.decisions.append(decision)
            return decision

        justification = self.build_justification(tool, arguments, evidence_ids)
        self._record("approval_requested", ApprovalDecision(
            tool=tool, risk=risk, required=True, granted=False, reason="pending",
            justification=justification, evidence_ids=evidence_ids,
        ))

        granted = bool(self.gateway.approve(tool, arguments, justification))

        decision = ApprovalDecision(
            tool=tool, risk=risk, required=True, granted=granted,
            reason=GRANTED if granted else DENIED,
            justification=justification, evidence_ids=evidence_ids,
            message="" if granted else (
                f"A human approver denied {tool}. The action was not executed. "
                f"Revise the plan or escalate with evidence."
            ),
        )
        self._record("approval_result", decision)
        self.decisions.append(decision)
        return decision

    # -- observation ------------------------------------------------------

    @staticmethod
    def blocked_observation(decision: ApprovalDecision, world_version: int | None) -> dict[str, Any]:
        """Tool-result envelope returned to the model when an action is blocked.

        Shape matches the simulator's envelope so the model and the replan policy
        treat it like any other observation.
        """
        status = (
            "approval_denied" if decision.reason == DENIED else "insufficient_evidence"
        )
        return {
            "status": status,
            "tool": decision.tool,
            "world_version": world_version,
            "evidence_id": None,
            "data": None,
            "retryable": False,
            "message": decision.message,
            "risk": decision.risk.value,
        }

    def _record(self, event: str, decision: ApprovalDecision) -> None:
        if self.trace is None:
            return
        self.trace.record(event, {
            "tool": decision.tool,
            "risk": decision.risk.value,
            "required": decision.required,
            "granted": decision.granted,
            "reason": decision.reason,
            "evidence_ids": decision.evidence_ids,
            "justification": decision.justification,
        })
