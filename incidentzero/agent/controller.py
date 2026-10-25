from __future__ import annotations

import json
import time
from typing import Any

from incidentzero.approval.broker import ApprovalBroker
from incidentzero.approval.gateway import ApprovalGateway
from incidentzero.domain.models import AgentOutcome, ModelReply, ToolCall
from incidentzero.model.base import ModelClient
from incidentzero.model.errors import ModelError, PermanentModelError, TransientModelError
from incidentzero.telemetry.budget import BudgetExceeded, BudgetManager
from incidentzero.telemetry.trace import TraceRecorder
from incidentzero.tools.registry import ToolRegistry

from .limits import Limits
from .planner import Planner
from .policies import ControlDecision, LoopGuard, ReplanPolicy, RiskPolicy
from .prompts import SYSTEM_PROMPT
from .recovery import RetryPolicy
from .state import AgentState

#: Tools that end the run when they succeed.
TERMINAL_TOOLS = {"close_incident", "escalate_incident"}


class AgentController:
    """Bounded autonomous incident controller.

    The model proposes one tool call at a time. Every proposal passes a Python
    gauntlet -- loop guard, schema validation, world-version precondition,
    close invariant, risk/approval -- before it can reach the simulator, and the
    observation that comes back is classified into exactly one control decision:
    continue, retry, correct, re-plan or abort.
    """

    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry,
        approval: ApprovalGateway,
        budget: BudgetManager,
        trace: TraceRecorder,
        limits: Limits | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.approval = approval
        self.budget = budget
        self.trace = trace
        self.limits = limits or Limits.load()
        self.state = AgentState()
        self.risk = RiskPolicy()
        self.planner = Planner(model, trace)
        self.replan_policy = ReplanPolicy(self.risk)
        self.loop_guard = LoopGuard(self.limits.max_same_action_repeats)
        self.retry_policy = RetryPolicy(
            max_attempts=self.limits.max_consecutive_model_retries,
            on_retry=self._on_model_retry,
        )
        # The LLM proposes; this broker is the only thing that can authorize (R7).
        self.approval_broker = ApprovalBroker(approval, self.risk, trace)
        self._started_at = time.monotonic()
        self._budget_warned = False
        #: World version at which loop-detection state was last cleared, so that a
        #: model stuck on one stale version cannot keep clearing it.
        self._loop_reset_at_version: int | None = None
        #: Recent model-call durations, used to avoid starting a call that would
        #: run past the hard runtime limit.
        self._model_latencies: list[float] = []

    # -- model ------------------------------------------------------------

    def _on_model_retry(self, event: dict[str, Any]) -> None:
        self.state.model_retries += 1
        self.trace.record("model_retry", event)

    def _model_decide(self) -> ModelReply:
        """One model decision, with bounded retry and correct budget accounting.

        Each attempt consumes LLM budget because each attempt is a real provider
        request (Q23). A budget error is not transient, so exhausting the budget
        stops the retry loop instead of burning what remains (R11, R12).
        """
        def attempt() -> ModelReply:
            self.budget.consume_llm()
            return self.model.decide(self.state.messages, self.tools.groq_tools)

        started = time.monotonic()
        reply = self.retry_policy.call_model(attempt)
        self._record_model_latency(time.monotonic() - started)
        self.trace.record("model_reply", {
            "content": reply.content,
            "finish_reason": reply.finish_reason,
            "usage": reply.usage,
            "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in reply.tool_calls],
            "llm_calls_used": self.budget.llm_calls,
        })
        return reply

    # -- observation envelopes --------------------------------------------

    def _blocked(self, tool: str, status: str, message: str = "", **extra: Any) -> dict[str, Any]:
        """Envelope for a call rejected by Python before it reached the simulator."""
        result = {
            "status": status,
            "tool": tool,
            "world_version": self.state.latest_world_version,
            "evidence_id": None,
            "data": None,
            "retryable": False,
            "message": message,
        }
        result.update(extra)
        return result

    # -- validation gauntlet ----------------------------------------------

    def _precheck(self, call: ToolCall) -> dict[str, Any] | None:
        """Reject a proposal before execution, or return None to allow it (R8).

        Ordered cheapest-first so that a call we are going to refuse never costs
        tool budget.
        """
        # R9: an unproductive repeat must be redirected, not executed.
        if self.loop_guard.would_block(call.name, call.arguments):
            return self._blocked(
                call.name, "loop_blocked",
                f"{call.name} has already been attempted "
                f"{self.loop_guard.count(call.name, call.arguments)} times with the same "
                f"arguments and did not resolve the incident. Choose a different action "
                f"or gather different evidence.",
            )

        # R8: unknown tool, wrong shape, missing or out-of-range arguments.
        ok, error = self.tools.validate(call.name, call.arguments)
        if not ok:
            self.state.validation_failures += 1
            return self._blocked(
                call.name, "validation_error",
                f"{error}. Correct the arguments and call the tool again.",
            )

        # R6: the model must act on the newest version it has observed.
        stale = self._precondition_error(call)
        if stale:
            return stale

        # R10: closure requires current verification evidence.
        if call.name == "close_incident":
            invariant = self._close_invariant_error(call)
            if invariant:
                return invariant

        return None

    def _precondition_error(self, call: ToolCall) -> dict[str, Any] | None:
        """Catch a model reusing a world version it has already seen superseded.

        This is a model-discipline check, not a substitute for the simulator's own
        optimistic-concurrency guard: the world can still move between this check
        and execution, and the simulator remains the authority.
        """
        expected = call.arguments.get("expected_world_version") if isinstance(call.arguments, dict) else None
        latest = self.state.latest_world_version
        if not isinstance(expected, int) or not isinstance(latest, int) or expected >= latest:
            return None
        # Reported as stale rather than as a generic validation failure, so that it
        # routes through the re-plan/re-observe path instead of a cheap correction.
        return self._blocked(
            call.name, "stale_precondition", retryable=True, message=(
                f"expected_world_version={expected} is older than the latest observed "
                f"world_version={latest}. Re-observe the affected service and reconsider "
                f"whether this action is still justified before retrying."
            ),
        )

    def _close_invariant_error(self, call: ToolCall) -> dict[str, Any] | None:
        """Refuse a close that is not backed by a current, passing verification (R10).

        The simulator enforces this too. Enforcing it here as well means the agent
        never even proposes a false success, and the refusal is traceable.
        """
        if not self.state.recovery_criteria_met or not self.state.latest_verify_evidence_id:
            return self._blocked(
                call.name, "validation_error",
                "close_incident requires a current verify_recovery result with "
                "criteria_met=true. Call verify_recovery first.",
            )
        cited = call.arguments.get("evidence_ids") if isinstance(call.arguments, dict) else None
        if not isinstance(cited, list) or self.state.latest_verify_evidence_id not in cited:
            return self._blocked(
                call.name, "validation_error",
                f"close_incident must cite the latest verify_recovery evidence id "
                f"{self.state.latest_verify_evidence_id} in evidence_ids.",
            )
        return None

    # -- execution --------------------------------------------------------

    def _execute_tool_call(self, call: ToolCall) -> dict[str, Any]:
        """Validate, approve if required, execute once, trace, return an observation."""
        blocked = self._precheck(call)
        if blocked is not None:
            self.trace.record("tool_blocked", {
                "call": {"name": call.name, "arguments": call.arguments},
                "result": blocked,
            })
            return blocked

        # R3/R7: consequential actions need observed evidence and a human decision.
        decision = self.approval_broker.evaluate(
            call.name, call.arguments, self.state.last_tool_results,
        )
        if decision.blocked:
            result = self.approval_broker.blocked_observation(
                decision, self.state.latest_world_version,
            )
            self.trace.record("tool_blocked", {
                "call": {"name": call.name, "arguments": call.arguments},
                "result": result,
            })
            return result

        self.budget.consume_tool()
        result = self.tools.execute(call.name, call.arguments)
        self.trace.record("tool_result", {
            "call": {"name": call.name, "arguments": call.arguments},
            "result": result,
            "tool_calls_used": self.budget.tool_calls,
        })
        return result

    def _perform(self, call: ToolCall) -> dict[str, Any]:
        """Execute one proposal, retrying only transient tool faults (R5).

        A telemetry backend timeout is retried here rather than being handed back
        to the model, which would cost an LLM call to learn nothing new. The loop
        guard counts the logical action once, not once per retry.

        The count is taken *after* the attempt, because ``_precheck`` inspects the
        guard with ``would_block``. Counting first would make the guard trip one
        attempt early and contradict ``max_same_action_repeats``, which allows two
        repeats before the next one is blocked.
        """
        fingerprint = self.loop_guard.fingerprint(call.name, call.arguments)
        attempts = self.limits.max_consecutive_model_retries

        result = self._execute_tool_call(call)
        for attempt in range(1, attempts):
            if self.replan_policy.classify(result) is not ControlDecision.RETRY:
                break
            if self.budget.remaining_tools <= self.limits.terminal_tool_reserve:
                break
            self.state.tool_retry_counts[fingerprint] = self.state.tool_retry_counts.get(fingerprint, 0) + 1
            self.trace.record("tool_retry", {
                "tool": call.name,
                "attempt": attempt,
                "max_attempts": attempts,
                "message": result.get("message"),
            })
            result = self._execute_tool_call(call)

        self.loop_guard.record(call.name, call.arguments)
        self.state.repeated_actions = dict(self.loop_guard.repeated_actions())
        return result

    # -- message plumbing -------------------------------------------------

    def _append_assistant(self, reply: ModelReply) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": reply.content}
        if reply.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in reply.tool_calls
            ]
        self.state.messages.append(msg)

    def _append_tool_result(self, call: ToolCall, result: dict[str, Any]) -> None:
        self.state.messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False, default=str),
        })

    def _system(self, content: str) -> None:
        self.state.messages.append({"role": "system", "content": content})

    # -- control decisions ------------------------------------------------

    def _handle_replan(self, result: dict[str, Any]) -> None:
        """Revise the plan in response to a grounded trigger (R4)."""
        trigger = self.replan_policy.replan_trigger(result)
        if self.budget.remaining_llm <= self.limits.warning_llm_calls_remaining:
            trigger["budget_low"] = True

        # Spending an LLM call to re-plan can cost the call needed to verify and
        # close, so a low budget forces a local revision instead.
        use_model = self.budget.remaining_llm > self.limits.warning_llm_calls_remaining

        self.trace.record("replan_triggered", trigger)
        plan = self.planner.revise(self.state.plan, trigger, self.state.summary(), use_model=use_model)
        self.state.record_replan(plan, trigger)

        # R9 permits reinterpreting repetition when new evidence *materially changes
        # the world*. Only a genuine world change qualifies. Clearing the counters
        # after a loop block would immediately unblock the action that was just
        # blocked, and clearing them after a denial would let the model re-propose
        # the action a human refused.
        world_moved = (
            trigger.get("requires_reobservation")
            and self.state.latest_world_version != self._loop_reset_at_version
        )
        if world_moved:
            self.loop_guard.note_material_change()
            self._loop_reset_at_version = self.state.latest_world_version
            self.trace.record("loop_state_reset", {
                "reason": trigger.get("status"),
                "world_version": self.state.latest_world_version,
            })

        instruction = (
            f"PLAN REVISION {plan.revision}. Previous approach was abandoned because: "
            f"{trigger.get('status')} ({trigger.get('message') or 'see trace'}). "
            f"New hypothesis: {plan.hypothesis} "
            f"Remaining steps: {[s.objective for s in plan.steps if s.status != 'done']}"
        )
        if trigger.get("requires_reobservation"):
            instruction += (
                " The world changed after your observation. Re-observe the affected "
                "service before re-attempting any action; do not simply resubmit the "
                "same action with a newer expected_world_version."
            )
        self._system(instruction)

    def _advance_plan(self, call: ToolCall, result: dict[str, Any]) -> None:
        """Mark plan progress so a later revision can preserve completed work."""
        if result.get("status") != "ok":
            return
        if call.name == "verify_recovery" and not self.state.recovery_criteria_met:
            return
        if self.risk.is_consequential(call.name) or call.name == "verify_recovery":
            self.state.mark_step_done()

    # -- budget and time discipline ---------------------------------------

    def _record_model_latency(self, seconds: float) -> None:
        """Remember how long recent model calls took, to predict the next one."""
        self._model_latencies.append(seconds)
        self._model_latencies = self._model_latencies[-5:]

    def _expected_model_latency(self) -> float:
        """Conservative estimate of the next model call, from recent observations.

        The slowest of the recent calls is used rather than the mean: under-
        estimating causes the overshoot this guard exists to prevent.
        """
        if not self._model_latencies:
            return 0.0
        return max(self._model_latencies)

    def _elapsed(self) -> float:
        return time.monotonic() - self._started_at

    def _out_of_time(self) -> bool:
        return self._elapsed() >= self.limits.max_runtime_seconds

    def _no_time_for_another_model_call(self) -> bool:
        """True when the next model call would very likely cross the limit.

        R5 requires terminating safely *before* the hard limit. Checking only the
        elapsed time lets a slow call start at 116s and finish at 128s, so the
        predicted cost of the call is included in the decision.
        """
        if not self._model_latencies:
            return False
        projected = self._elapsed() + self._expected_model_latency()
        return projected >= self.limits.max_runtime_seconds

    def _must_converge(self) -> bool:
        """True when remaining resources are only enough for a terminal action (R12)."""
        return (
            self.budget.remaining_llm <= 1
            or self.budget.remaining_tools <= self.limits.terminal_tool_reserve
            or self._out_of_time()
            or self._no_time_for_another_model_call()
        )

    def _convergence_reason(self) -> str:
        if self._out_of_time():
            return "runtime limit reached"
        if self._no_time_for_another_model_call():
            return (
                f"insufficient time for another model call "
                f"(elapsed {self._elapsed():.0f}s, recent call {self._expected_model_latency():.0f}s, "
                f"limit {self.limits.max_runtime_seconds}s)"
            )
        return "call budget nearly exhausted"

    def _warn_budget_once(self) -> None:
        if self._budget_warned:
            return
        if self.budget.remaining_llm > self.limits.warning_llm_calls_remaining:
            return
        self._budget_warned = True
        self.trace.record("budget_warning", {
            "remaining_llm": self.budget.remaining_llm,
            "remaining_tools": self.budget.remaining_tools,
        })
        self._system(
            f"BUDGET WARNING: {self.budget.remaining_llm} model calls and "
            f"{self.budget.remaining_tools} tool calls remain. Stop investigating. "
            f"Run verify_recovery and then close_incident, or escalate_incident with "
            f"the evidence already gathered."
        )

    # -- terminal ---------------------------------------------------------

    def _outcome(self, status: str, summary: str) -> AgentOutcome:
        self.state.status = status
        self.trace.record("terminal", {
            "status": status,
            "summary": summary,
            "llm_calls": self.budget.llm_calls,
            "tool_calls": self.budget.tool_calls,
            "plan_revisions": self.state.plan_revisions,
            "world_version": self.state.latest_world_version,
            "evidence_ids": self.state.evidence_ids,
        })
        return AgentOutcome(
            status=status,
            summary=summary,
            llm_calls=self.budget.llm_calls,
            tool_calls=self.budget.tool_calls,
            final_world_version=self.state.latest_world_version,
            evidence_ids=self.state.evidence_ids,
            trace_path=str(self.trace.path),
        )

    def _forced_termination(self, reason: str, fallback_status: str = "budget_exhausted") -> AgentOutcome:
        """Close or escalate without the model when resources run out (R12, R10).

        Verification is attempted first: if the incident genuinely recovered, the
        run should end as resolved rather than abandoning a fixed system. Otherwise
        it escalates with whatever evidence exists, which is a correct autonomous
        outcome (Example 5).
        """
        self.trace.record("forced_termination", {
            "reason": reason,
            "remaining_llm": self.budget.remaining_llm,
            "remaining_tools": self.budget.remaining_tools,
        })
        try:
            if self.budget.remaining_tools >= 2 and not self.state.recovery_criteria_met:
                self.state.observe_result(self._direct("verify_recovery", {}))

            if self.state.recovery_criteria_met and self.state.latest_verify_evidence_id:
                closed = self._direct("close_incident", {
                    "summary": f"Incident closed after verified recovery. Terminated early: {reason}.",
                    "evidence_ids": [self.state.latest_verify_evidence_id],
                    "expected_world_version": self.state.latest_world_version,
                    "reason": f"Recovery criteria verified; {reason}.",
                })
                self.state.observe_result(closed)
                if closed.get("status") == "ok":
                    return self._outcome("resolved", "Incident closed with verified recovery evidence.")

            if self.budget.remaining_tools >= 1:
                escalated = self._direct("escalate_incident", {
                    "reason": (
                        f"Autonomous remediation stopped: {reason}. Escalating with the "
                        f"evidence gathered so far."
                    ),
                    "evidence_ids": self.state.evidence_ids[-8:] or ["none"],
                })
                self.state.observe_result(escalated)
                if escalated.get("status") == "ok":
                    return self._outcome("escalated", f"Escalated with evidence: {reason}.")
        except BudgetExceeded:
            pass
        return self._outcome(fallback_status, f"Run stopped without a terminal action: {reason}.")

    def _direct(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Controller-initiated tool call, bypassing the model but not the budget."""
        self.budget.consume_tool()
        result = self.tools.execute(name, arguments)
        self.trace.record("tool_result", {
            "call": {"name": name, "arguments": arguments},
            "result": result,
            "initiated_by": "controller",
        })
        return result

    # -- main loop --------------------------------------------------------

    def run(self) -> AgentOutcome:
        self.state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Investigate the active production incident, mitigate it safely, verify recovery, then close it; otherwise escalate with evidence."},
        ]
        try:
            # Bootstrap with one real observation so the plan rests on evidence (R2).
            self.budget.consume_tool()
            incident = self.tools.execute("get_incident", {})
            self.state.observe_result(incident)
            self.trace.record("bootstrap_incident", incident)
            self._system(f"Current incident evidence: {json.dumps(incident, default=str)}")

            self.budget.consume_llm()
            self.state.plan = self.planner.create(incident)
            self.trace.record("plan_created", {
                "revision": self.state.plan.revision,
                "hypothesis": self.state.plan.hypothesis,
                "steps": [{"id": s.step_id, "objective": s.objective, "success_signal": s.success_signal} for s in self.state.plan.steps],
            })
            self._system(
                f"Active plan (revision {self.state.plan.revision}). Hypothesis: "
                f"{self.state.plan.hypothesis} Steps: {[s.objective for s in self.state.plan.steps]}"
            )

            while self.budget.remaining_llm > 0 and self.budget.remaining_tools > 0:
                if self._must_converge():
                    return self._forced_termination(self._convergence_reason())
                self._warn_budget_once()

                try:
                    reply = self._model_decide()
                except PermanentModelError as exc:
                    # Configuration/request error: retrying cannot help (R5).
                    self.trace.record("model_permanent_error", {"error": str(exc)})
                    return self._forced_termination(f"permanent model error: {exc}", "aborted")
                except TransientModelError as exc:
                    # Bounded retries already exhausted inside the retry policy.
                    self.trace.record("model_retry_exhausted", {"error": str(exc)})
                    return self._forced_termination(f"model unavailable after bounded retries: {exc}", "aborted")

                self._append_assistant(reply)

                if not reply.tool_calls:
                    # Q25: model text is never operational success.
                    self.state.empty_replies += 1
                    self.trace.record("model_stopped_without_tool_call", {
                        "content": reply.content, "occurrence": self.state.empty_replies,
                    })
                    if self.state.empty_replies >= 2:
                        return self._forced_termination("model stopped proposing tool calls", "failed")
                    self._system(
                        "Natural-language output is not an operational result. The incident "
                        "is only resolved once verify_recovery reports criteria_met=true and "
                        "close_incident succeeds. Call a tool now, or escalate_incident."
                    )
                    continue

                call = reply.tool_calls[0]
                result = self._perform(call)
                self.state.observe_result(result)
                self._append_tool_result(call, result)

                # A successful mutation invalidates any earlier recovery proof (R10).
                if result.get("status") == "ok" and self.risk.is_consequential(call.name) and call.name not in TERMINAL_TOOLS:
                    self.state.invalidate_verification()

                if call.name in TERMINAL_TOOLS and result.get("status") == "ok":
                    if call.name == "close_incident":
                        return self._outcome("resolved", "Incident closed with verified simulator evidence.")
                    return self._outcome("escalated", "Incident escalated with evidence.")

                decision = self.replan_policy.classify(result)
                self.trace.record("control_decision", {
                    "tool": call.name, "status": result.get("status"), "decision": decision.value,
                })

                if decision is ControlDecision.CONTINUE:
                    self._advance_plan(call, result)
                elif decision is ControlDecision.CORRECT:
                    self._system(
                        f"The previous call to {call.name} was rejected before execution: "
                        f"{result.get('message')} Fix the problem and continue with the plan."
                    )
                elif decision is ControlDecision.RETRY:
                    # Bounded tool retries already happened in _perform.
                    self._handle_replan(result)
                elif decision is ControlDecision.ABORT:
                    return self._forced_termination(f"unrecoverable condition from {call.name}", "aborted")
                else:  # REPLAN
                    if self.state.plan_revisions >= self.limits.max_plan_revisions:
                        return self._forced_termination("plan revision limit reached without recovery")
                    self._handle_replan(result)

            return self._forced_termination("call budget exhausted")

        except BudgetExceeded as exc:
            self.trace.record("budget_exhausted", {"error": str(exc)})
            return self._outcome("budget_exhausted", str(exc))
        except ModelError as exc:
            self.trace.record("model_error", {"error": str(exc)})
            return self._outcome("failed", f"Model error: {exc}")
