"""Offline model adapter for demos and tests (R14).

This is a stand-in for Groq, not part of the agent's reasoning. It implements the
same :class:`~incidentzero.model.base.ModelClient` protocol and decides what to
propose by reading the tool messages already in the transcript -- exactly the
information a real LLM would have.

It deliberately does NOT touch the simulator, the scenario seed, or any private
field such as ``_scenario_spec`` or ``_oracle_snapshot``. Swapping this adapter
for Groq must not change the controller's behaviour; that is the whole point of
keeping the reliability logic in Python.
"""

from __future__ import annotations

import json
from typing import Any

from incidentzero.domain.models import ModelReply, ToolCall

#: Components worth surveying on the checkout path before remediating.
SURVEY_ORDER = (
    "redis-cache",
    "inventory-service",
    "checkout-service",
    "order-db",
)

CACHE_ERROR_RATE = 0.05
MEMORY_PRESSURE_PCT = 85.0
CPU_SATURATION_PCT = 80.0
DB_CONNECTION_SATURATION_PCT = 85.0
MAX_REPLICAS = 8

#: Only this component supports a failover in the simulated platform.
DATABASE_SERVICE = "order-db"


class HeuristicModelClient:
    """Proposes the least-risky action supported by observed metrics."""

    def __init__(self, survey: tuple[str, ...] = SURVEY_ORDER) -> None:
        self.survey = survey
        self.calls = 0
        self.revisions = 0

    # -- planning ---------------------------------------------------------

    def structured(self, messages: list[dict[str, Any]], schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.revisions += 1
        return {
            "hypothesis": (
                f"Revision {self.revisions}: the ticket's suspected service is unverified. "
                f"Identify the abnormal component from metrics before remediating."
            ),
            "rationale_summary": "Survey the checkout path, remediate what the evidence shows, then verify.",
            "steps": [
                {"step_id": f"p{self.revisions}-1", "objective": "Survey metrics across the checkout path.", "success_signal": "Metrics read for each component."},
                {"step_id": f"p{self.revisions}-2", "objective": "Remediate the component the evidence implicates.", "success_signal": "Action returns status ok."},
                {"step_id": f"p{self.revisions}-3", "objective": "Verify recovery, then close or escalate.", "success_signal": "verify_recovery reports criteria_met."},
            ],
        }

    # -- acting -----------------------------------------------------------

    @staticmethod
    def _observations(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Every tool result already visible in the transcript."""
        out: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") != "tool":
                continue
            try:
                parsed = json.loads(message.get("content") or "")
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
        return out

    def decide(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelReply:
        self.calls += 1
        call_id = f"call-{self.calls}"
        observations = self._observations(messages)

        world = max(
            [o["world_version"] for o in observations if isinstance(o.get("world_version"), int)] + [1]
        )
        metrics = {
            o["data"]["service"]: o["data"]
            for o in observations
            if o.get("tool") == "get_metrics" and o.get("status") == "ok"
        }
        succeeded = {
            (o.get("tool"), (o.get("data") or {}).get("service"))
            for o in observations if o.get("status") == "ok"
        }

        def reply(name: str, arguments: dict[str, Any]) -> ModelReply:
            return ModelReply(
                content=None,
                tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
            )

        # 1. Gather evidence before proposing anything consequential (R3).
        for service in self.survey:
            if service not in metrics:
                return reply("get_metrics", {"service": service})

        # 2. Remediate the component the metrics implicate, least risky first.
        cache = metrics.get("redis-cache", {})
        if cache.get("error_rate", 0.0) >= CACHE_ERROR_RATE and ("clear_cache", "redis-cache") not in succeeded:
            return reply("clear_cache", {
                "service": "redis-cache",
                "expected_world_version": world,
                "reason": f"redis-cache error rate is {cache['error_rate']}, consistent with bad cached objects",
            })

        database = metrics.get(DATABASE_SERVICE, {})
        if database.get("db_connections_pct", 0.0) >= DB_CONNECTION_SATURATION_PCT \
                and ("failover_database", DATABASE_SERVICE) not in succeeded:
            return reply("failover_database", {
                "service": DATABASE_SERVICE,
                "expected_world_version": world,
                "reason": (
                    f"{DATABASE_SERVICE} connection pool is {database['db_connections_pct']}% "
                    f"saturated with p95 {database.get('p95_ms')}ms, indicating a degraded primary"
                ),
            })

        for service, m in metrics.items():
            if m.get("memory_pct", 0.0) >= MEMORY_PRESSURE_PCT and ("restart_service", service) not in succeeded:
                return reply("restart_service", {
                    "service": service,
                    "expected_world_version": world,
                    "reason": f"{service} memory is at {m['memory_pct']}%, consistent with a leak",
                })

        for service, m in metrics.items():
            replicas = m.get("replicas", MAX_REPLICAS)
            if m.get("cpu_pct", 0.0) >= CPU_SATURATION_PCT and replicas < MAX_REPLICAS \
                    and ("scale_service", service) not in succeeded:
                return reply("scale_service", {
                    "service": service,
                    "replicas": min(MAX_REPLICAS, replicas + 3),
                    "expected_world_version": world,
                    "reason": f"{service} cpu is at {m['cpu_pct']}% with only {replicas} replicas",
                })

        # 3. Verify, then close on proof or escalate on evidence.
        verification = None
        for o in reversed(observations):
            if o.get("tool") == "verify_recovery" and o.get("status") == "ok":
                verification = o
                break

        evidence = [o["evidence_id"] for o in observations if o.get("evidence_id")]

        if verification is None:
            return reply("verify_recovery", {})

        if not (verification.get("data") or {}).get("criteria_met"):
            if observations and observations[-1] is verification:
                # Freshly verified as still broken and no further action is supported.
                return reply("escalate_incident", {
                    "reason": (
                        "Remediation did not restore the recovery objective and no further "
                        "action is supported by the evidence gathered."
                    ),
                    "evidence_ids": evidence[-5:] or ["none"],
                })
            return reply("verify_recovery", {})

        return reply("close_incident", {
            "summary": "Remediated the implicated component and verified objective recovery of the checkout path.",
            "evidence_ids": [verification["evidence_id"]],
            "expected_world_version": world,
            "reason": "verify_recovery reported criteria_met=true",
        })
