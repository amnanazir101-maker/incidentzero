"""Student-written tests: ten, one per failure class (Task G).

Every test is offline. Six drive the real controller against the real simulator
with a local fake model (``FakeModel`` from conftest, or the starter's own
``ScriptedModelClient``); the other four pin down the policy objects directly.
None reads an API key, loads ``.env``, or constructs a Groq client.

    #   failure class                       requirement   model used
    1   stale world version                 R6 / Q20      none (policy)
    2   transient retry exhaustion          R5 / R11      none (policy)
    3   consequential action without evidence R3          none (broker)
    4   risk policy changed in config       R7 / viva     none (broker)
    5   re-plan repeats a failed hypothesis R4 / Task E   ScriptedModelClient
    6   invented tool                       R8            FakeModel
    7   premature close                     R10           FakeModel
    8   repeated identical action           R9            FakeModel
    9   human denies a high-risk action     R7            FakeModel
    10  reduced call budget                 R12           FakeModel
"""

from __future__ import annotations

import json

import pytest

from incidentzero.agent.planner import Planner
from incidentzero.agent.policies import ControlDecision, ReplanPolicy, RiskPolicy
from incidentzero.agent.recovery import RetryPolicy
from incidentzero.approval.broker import INSUFFICIENT_EVIDENCE, ApprovalBroker
from incidentzero.approval.gateway import AlwaysDenyGateway
from incidentzero.domain.models import AgentPlan, PlanStep
from incidentzero.model.errors import TransientModelError
from incidentzero.model.scripted import ScriptedModelClient
from incidentzero.telemetry.budget import BudgetManager

from .conftest import FakeModel, blocked_results, executed_tools, repeat_tool, tool_reply

TERMINAL_STATUSES = {"resolved", "escalated", "aborted", "budget_exhausted", "failed"}


class RecordingGateway:
    """Approval gateway that records whether it was consulted at all."""

    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.requests: list[str] = []

    def approve(self, action_name, arguments, justification):
        self.requests.append(action_name)
        return self.answer


# 1 -- stale world version (R6, Q20) ---------------------------------------

def test_stale_world_replans_and_demands_reobservation():
    """A stale action must be re-planned after re-observation, never refired
    with a newer version substituted in."""
    policy = ReplanPolicy(RiskPolicy())
    result = {"status": "stale_precondition", "tool": "scale_service", "retryable": True}

    assert policy.classify(result) is ControlDecision.REPLAN
    assert policy.should_replan(result) is True
    assert policy.requires_reobservation(result) is True
    assert policy.replan_trigger(result)["requires_reobservation"] is True


# 2 -- transient retry exhaustion (R5, R11) --------------------------------

def test_retry_policy_stops_at_the_configured_bound():
    """Bounded retry with backoff, then surface the failure. Never retry forever."""
    attempts = {"n": 0}
    sleeps: list[float] = []
    retry = RetryPolicy(max_attempts=3, sleeper=sleeps.append)

    def always_transient():
        attempts["n"] += 1
        raise TransientModelError("429 rate limited")

    with pytest.raises(TransientModelError):
        retry.call_model(always_transient)

    assert attempts["n"] == 3, "exactly max_attempts provider requests"
    assert sleeps == [0.5, 1.0], "backoff doubles, and no sleep after the last attempt"


# 3 -- consequential action without evidence (R3) --------------------------

def test_unevidenced_critical_action_is_refused_without_asking_a_human():
    """A model sentence is not evidence. With no supporting observation the
    action is refused and a human decision is never spent on it."""
    gateway = RecordingGateway(answer=True)
    broker = ApprovalBroker(gateway, RiskPolicy())

    decision = broker.evaluate(
        "failover_database",
        {"service": "order-db", "expected_world_version": 3, "reason": "the database seems slow"},
        observations=[],
    )

    assert decision.reason == INSUFFICIENT_EVIDENCE
    assert decision.blocked is True
    assert gateway.requests == []


# 4 -- risk policy is configuration, not code (R7, viva) -------------------

def test_risk_level_change_in_config_changes_approval_requirement(tmp_path):
    """Raising a tool's risk level in risk_policy.json makes the gate demand
    approval for it, with no code change."""
    config = json.loads(open("configs/risk_policy.json", encoding="utf-8").read())
    config["restart_service"] = "critical"
    path = tmp_path / "risk.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    gateway = RecordingGateway(answer=True)
    broker = ApprovalBroker(gateway, RiskPolicy(path))
    evidence = [{"status": "ok", "tool": "get_metrics", "evidence_id": "EV-0002",
                 "data": {"service": "checkout-service", "error_rate": 0.4}}]

    decision = broker.evaluate(
        "restart_service",
        {"service": "checkout-service", "expected_world_version": 3, "reason": "memory pressure is high"},
        evidence,
    )

    assert decision.required is True
    assert gateway.requests == ["restart_service"]


# 5 -- re-plan must not repeat a failed hypothesis (R4, Task E) ------------

def test_model_repeating_the_failed_hypothesis_is_overridden():
    """The model that produced the failed plan is asked to revise it and hands
    back the same hypothesis. Python rejects it and still increments the
    revision, using the starter's ScriptedModelClient as the offline model."""
    current = AgentPlan(
        hypothesis="A recent deployment regressed checkout.",
        steps=[
            PlanStep("s1", "Compare deployment timing with error onset", "versions compared", "done"),
            PlanStep("s2", "Roll back checkout-service", "action returns ok"),
        ],
        revision=0,
    )
    repeated = {
        "hypothesis": "A recent deployment regressed checkout.",
        "rationale_summary": "same again",
        "steps": [
            {"step_id": "n1", "objective": "Roll back checkout-service", "success_signal": "ok"},
            {"step_id": "n2", "objective": "Verify", "success_signal": "criteria_met"},
        ],
    }
    planner = Planner(ScriptedModelClient(structured_outputs=[repeated]))

    revised = planner.revise(
        current, {"status": "verification_failed", "verification_failed": True}, "state",
    )

    assert revised.revision == 1
    assert revised.hypothesis != repeated["hypothesis"]
    assert [s.step_id for s in revised.steps if s.status == "done"] == ["s1"], "completed work is preserved"


# 6 -- invented tool (R8) --------------------------------------------------

def test_invented_tool_never_reaches_the_simulator(make_controller, read_trace):
    """An unknown tool name is rejected before execution with a corrective
    observation, and the run still ends in a valid terminal state."""
    controller = make_controller(FakeModel(repeat_tool("delete_production_database", {"confirm": True})))

    outcome = controller.run()
    rows = read_trace()

    assert "delete_production_database" not in executed_tools(rows)
    assert any(r["status"] == "validation_error" for r in blocked_results(rows))
    assert outcome.status in TERMINAL_STATUSES


# 7 -- premature close (R10) -----------------------------------------------

def test_close_without_verification_is_refused(make_controller, read_trace):
    """The model claims success; Python refuses to record it without a
    passing verify_recovery result."""
    controller = make_controller(FakeModel(repeat_tool("close_incident", {
        "summary": "Everything looks healthy to me, closing this incident now.",
        "evidence_ids": ["EV-0001"], "expected_world_version": 1,
        "reason": "the dashboards looked fine",
    })))

    outcome = controller.run()
    rows = read_trace()

    assert outcome.status != "resolved"
    assert "close_incident" not in executed_tools(rows)
    assert any("verify_recovery" in (r["message"] or "") for r in blocked_results(rows))


# 8 -- repeated identical action (R9) --------------------------------------

def test_repeated_identical_action_is_blocked_and_replans(make_controller, read_trace):
    """The same action beyond the configured repeat limit is blocked, not
    executed, and the block triggers a plan revision."""
    controller = make_controller(FakeModel(repeat_tool("restart_service", {
        "service": "checkout-service", "expected_world_version": 1,
        "reason": "restarting again because the errors have not cleared",
    })))

    controller.run()
    rows = read_trace()

    executed = [t for t in executed_tools(rows) if t == "restart_service"]
    assert len(executed) <= controller.limits.max_same_action_repeats
    assert any(r["status"] == "loop_blocked" for r in blocked_results(rows))
    assert controller.state.plan_revisions >= 1


# 9 -- human denies a high-risk action (R7, Example 3) ---------------------

def test_denied_high_risk_action_never_mutates_the_world(registry, make_controller, read_trace):
    """The human says no. The rollback does not happen, proven by reading the
    deployed version from the simulator before and after."""
    before = registry.execute("get_service_health", {"service": "checkout-service"})["data"]["version"]
    deployments = registry.execute("get_deployments", {"service": "checkout-service"})["data"]

    def replies(n, messages):
        if n == 1:
            return tool_reply("get_metrics", {"service": "checkout-service"}, "c1")
        return tool_reply("rollback_deployment", {
            "service": "checkout-service",
            "target_version": deployments["previous"]["version"],
            "expected_world_version": registry.environment.world_version,
            "reason": "checkout errors appeared right after the most recent release",
        }, f"c{n}")

    controller = make_controller(FakeModel(replies), approval=AlwaysDenyGateway())
    controller.run()

    after = registry.execute("get_service_health", {"service": "checkout-service"})["data"]["version"]
    assert after == before, "a denied rollback must not change the deployed version"
    assert "rollback_deployment" not in executed_tools(read_trace())
    assert any(r["status"] == "approval_denied" for r in blocked_results(read_trace()))


# 10 -- reduced call budget (R12) ------------------------------------------

def test_reduced_budget_terminates_safely_within_limits(make_controller, read_trace):
    """With the budget cut well below default the run converges on a terminal
    action inside the limits instead of dying mid-loop."""
    controller = make_controller(
        FakeModel(repeat_tool("get_metrics", {"service": "checkout-service"})),
        budget=BudgetManager(max_llm_calls=4, max_tool_calls=6),
    )

    outcome = controller.run()

    assert outcome.llm_calls <= 4
    assert outcome.tool_calls <= 6
    assert outcome.status in TERMINAL_STATUSES
    assert any(row["event"] == "forced_termination" for row in read_trace())
