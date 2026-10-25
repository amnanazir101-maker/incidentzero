# IncidentZero: engineering report

Scenarios: public-a, public-b and public-c for student ID 24I-8070.
Model: `openai/gpt-oss-20b` via Groq (default). Tests: 10 behaviour tests, one per failure class (`tests/student/test_agent_requirements.py`); six drive the controller with a local fake model, one uses the starter's `ScriptedModelClient`; none calls Groq.

## 1. Architecture

The runtime is a single Python control loop in `AgentController.run()`. The model proposes one tool call per turn; Python decides what happens to it. Nothing the model says is executed directly or taken as proof.

Each proposal passes a validation gauntlet in a fixed order, cheapest first, so a call the controller will refuse never costs tool budget: loop guard, JSON-schema validation through the supplied `ToolRegistry`, a world-version precondition, the close invariant, then risk-based approval. Only then does `ToolRegistry.execute()` run. The observation that comes back is classified by `ReplanPolicy.classify()` into exactly one of five control decisions - `CONTINUE`, `RETRY`, `CORRECT`, `REPLAN`, `ABORT` - and the loop branches on that. The public `should_replan()` contract is a one-line wrapper over `classify()`.

Explicit state lives in `AgentState`, separate from the transcript: the active plan and revision number, evidence IDs, latest `world_version`, the last eight tool results, retry and validation counters, and the two fields that gate closure - the latest `verify_recovery` evidence ID and whether it reported `criteria_met`.

`Planner` creates and revises plans; `ApprovalBroker` owns authorization; `RetryPolicy` owns bounded retry with injected sleep; `LoopGuard` fingerprints actions; `Limits` loads every bound from `configs/limits.json`; `TraceRecorder` writes one JSONL row per event. The simulator is reached only through the registry; agent code never touches `_scenario_spec`, `_services`, `_oracle_snapshot()` or the seed.

The Groq adapter strips harmony-format control tokens (`<|channel|>commentary`) from tool names, and when Groq rejects a tool call server-side it recovers the attempted call from `failed_generation` and hands it to the controller's own validator instead of retrying blind. An offline `HeuristicModelClient` implements the same protocol from transcript contents alone.

## 2. Planning and re-planning strategy

The first plan is created after one real observation (`get_incident`), so it is grounded in evidence. Model output is validated against `PLAN_SCHEMA` before it is trusted; if it fails, a bootstrap plan derived from the ticket is used and the fallback is traced.

A revision is requested whenever `classify()` returns `REPLAN`: a stale precondition, a denied approval, a non-retryable or repeated consequential action, or a remediation that returned `ok` while `verify_recovery` still reports `criteria_met=false`. The trigger is recorded structurally (status, tool, message, world version, whether re-observation is required) and passed to `Planner.revise()`.

`revise()` guarantees four things whatever the model returns. The revision number always increments; Python owns the counter. Steps marked done are carried forward, the rest replaced. The superseded hypothesis joins a rejected list fed into the next revision prompt as a do-not-repeat instruction. If the model returns an already-rejected hypothesis or an unchanged plan, its output is discarded for a deterministic local revision keyed on the trigger - for a stale world the local plan starts with "re-observe, then decide whether the action is still justified", so a moved world is always re-checked before acting.

Two budget rules apply. Re-planning through the model costs an LLM call, so below the warning threshold the revision is produced locally. Revisions are capped at four; past that the controller converges rather than revising forever. Loop-guard counters are cleared only when the world version has actually advanced since the last clear, never merely because a re-plan happened.

## 3. Failure handling

**Malformed model output.** An unknown tool, wrong shape, or out-of-range argument is rejected before execution and the model receives the schema error verbatim as a corrective observation, classified `CORRECT`, not `REPLAN`: the plan is not wrong, the call is. Rejected calls count toward the loop guard, so a repeated invalid call gets exactly two corrections before it is blocked. When Groq began rejecting arguments server-side, bypassing this path, the adapter was changed to recover the attempted call from the provider error so the same loop applies.

**Transient model failure.** `RetryPolicy` retries only `TransientModelError`, with capped exponential backoff and no sleep after the final attempt. Budget is consumed inside the callable, so every attempt is charged and a `BudgetExceeded` - which is not a `ModelError` - stops the loop immediately rather than burning the remainder. Permanent errors propagate on the first attempt.

**Transient tool failure.** A telemetry `transient_error` is retried inside `_perform()` without going back to the model, which would cost an LLM call to learn nothing. The loop guard counts the logical action once.

**Stale world version.** A model reusing a version it has seen superseded gets `stale_precondition`, not a generic validation error, so it routes through re-plan with `requires_reobservation=true`. The simulator's own check remains authoritative for anything that moves between pre-check and execution.

**Approval denial.** See Section 4. The denial is an observation, `status=approval_denied`; it triggers a re-plan whose local fallback ends in "find a lower-risk action, else escalate".

**Impossible tasks.** No special case. If remediation does not move `criteria_met` to true within the revision cap or the budget, the run converges on `escalate_incident` with the evidence gathered. Escalation is a correct terminal outcome.

**Budget exhaustion.** `_must_converge()` fires when LLM calls fall to one, tool calls fall to a reserve of three, the runtime limit is reached, or when elapsed time plus the slowest recent model latency would cross it. Convergence runs `verify_recovery` and closes if the criteria are met, otherwise escalates. In both live runs in Section 5 the model never verified; this path did, and closed on the evidence.

## 4. Safety and stopping

Risk levels come from `configs/risk_policy.json`; nothing in code names a tool as high-risk. Tools mapped `high` or `critical` - by default `rollback_deployment`, `shift_traffic`, `failover_database` - go through `ApprovalBroker.evaluate()`. Unknown tools default to `critical`. A test rewrites the config to make `restart_service` critical and asserts the gate adapts.

Before a human is asked, the broker requires at least one successful observation about the targeted service, or a global one, among recent results. A model sentence is not evidence; nor is a failed observation, nor metrics about a different service. Without it the action is refused with `insufficient_evidence` and the human is never bothered. With it, the model's stated reason is shown to the approver, labelled as a claim, beside the evidence IDs. Only the gateway's boolean grants execution. Denial produces `approval_denied`, executes nothing, and a test proves it by reading the deployed version before and after.

Proof of recovery is a chain: `verify_recovery` returns `criteria_met=true`; its evidence ID is retained; `close_incident` cites that ID and the latest world version; the simulator returns `ok`. The controller enforces this before execution: a close without a passing verification, or not citing the latest verification ID, is refused. Any successful consequential action invalidates prior verification, so stale proof cannot be reused.

## 5. Evaluation

Live runs against `openai/gpt-oss-20b`, default limits (14 LLM, 28 tool, 120 s). The public-a incident (`Order writes are timing out across the checkout path`) needs `failover_database`, a critical action; the live model never proposed it, so the approval gate was exercised only in tests.

| Scenario | Outcome | LLM | Tools | Elapsed | Actions | Re-planned | Notes |
|---|---|---|---|---|---|---|---|
| public-a | escalated | 9 | 10 | 108.8 s | 1 | yes (1) | ticket's suspect scaled; database never observed; two malformed calls corrected; converged before limit |
| public-b | resolved | 10 | 11 | 119.0 s | 2 | yes (1) | stale precondition; model revision rejected by schema; malformed final call; Python verified and closed |
| public-c | resolved | 8 | 11 | 111.5 s | 2 | no | telemetry timeout retried; time guard converged; Python verified and closed |

Offline, with `HeuristicModelClient` and the same controller, all three resolve in 8-10 LLM and tool calls in under 0.1 s. Live latency of 15-25 s per call against a 120 s limit permits about five substantive decisions. In public-b and public-c the model never called `verify_recovery`; the convergence path did, and closed on the evidence. In public-a it did verify, saw `false`, and re-planned with no time left to act on it.

## 6. Three failure traces

**public-b: stale world, rejected revision, malformed final call.** The model read inventory-service metrics and logs, then proposed `scale_service` with `expected_world_version=1` - but a scheduled event had already moved the world to 2. The simulator returned `stale_precondition`, classified `REPLAN` with `requires_reobservation=true`. `Planner.revise()` asked the model for a new plan; Groq rejected it against `PLAN_SCHEMA`, the planner used the local stale-world revision ("re-observe, then decide whether the action is still justified"), and the loop guard was cleared because the world had genuinely moved. The model re-observed, scaled at version 2, observed again, and restarted at version 3 - the correct remedy for the memory leak the metrics showed. Its next generation was `{"arguments": {""}"}`, unparseable JSON Groq rejected as `tool_use_failed`; the adapter could not recover it and classified it permanent. Convergence ran `verify_recovery` (`criteria_met=true`) and closed. Three failures, three distinct responses. The adapter has since been extended to recover the tool name from unparseable generations.

**public-c: transient telemetry, then out of time.** `get_logs` on inventory-service returned `transient_error`. `_perform()` retried it once without consulting the model - one tool call, zero LLM calls. The model scaled and restarted inventory-service. At 111.5 s, with a ~20 s recent call, the time guard projected past 120 s and converged: verification passed, the incident closed. An elapsed-only check would have let the next call finish at roughly 131 s.

**public-a: two malformed calls, one wrong hypothesis.** The ticket blamed payment-service; the title said order writes were timing out. The model read payment-service metrics (healthy), hit a telemetry timeout on its logs - retried without an LLM call - then proposed `get_metrics` with an extra `limit` argument. Our validator rejected it (`additionalProperties`), classified `CORRECT`, and the model recovered on the next call. It scaled payment-service at 56.7 s, treating the ticket as proof. Its next generation was unparseable JSON for `verify_recovery`; Groq rejected it as `tool_use_failed`, the adapter recovered the tool name by regex, our validator rejected the placeholder arguments with a schema message, and the model reissued `verify_recovery` correctly at 107.7 s. `criteria_met=false` triggered a re-plan. One second later the time guard projected past 120 s and converged: verified again, escalated with ten evidence IDs. Order-db was never observed and the failover it needed was never proposed. Two malformed generations cost one call each instead of aborting the run; the wrong hypothesis cost the rest.

## 7. Trade-offs and next steps

1. **Latency, not logic, bounds live performance.** The runtime guard converges before the limit but cannot buy back the 20-25 s each `gpt-oss-20b` call costs. A 120 s target leaves no room for one wrong turn.

2. **Backoff ignores `retry-after`.** A daily-quota 429 saying "try again in 13 minutes" is retried at 0.5 s and 1 s. Bounded and safe, but pointless; parsing the provider's hint would let the controller escalate at once.

3. **Plan-step completion is coarse.** A step is marked done when a consequential action or passing verification succeeds, without matching its stated success signal. Revision preserves the right steps in practice, but the mapping is heuristic.

4. **Evidence relevance is by service name only.** The broker accepts any successful observation about the target service without judging whether it supports the specific action; that judgement is left to the human approver, who sees the IDs.

5. **Loop detection is limited to argument fingerprints.** Two `scale_service` calls on different services are different fingerprints. The prompt asks the model not to repeat "similar actions"; nothing enforces it.
