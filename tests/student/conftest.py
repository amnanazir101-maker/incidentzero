"""Shared fixtures for student-written tests.

Every fixture here is offline. No test in this package calls Groq, so the suite
costs no provider quota (R14) and runs in well under a second.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from incidentzero.agent.controller import AgentController
from incidentzero.agent.limits import Limits
from incidentzero.approval.gateway import AlwaysApproveGateway
from incidentzero.domain.models import ModelReply, ToolCall
from incidentzero.environment.engine import SimulationEnvironment
from incidentzero.telemetry.budget import BudgetManager
from incidentzero.telemetry.trace import TraceRecorder
from incidentzero.tools.registry import ToolRegistry

VALID_PLAN = {
    "hypothesis": "The ticket's suspected service is unverified and must be tested.",
    "rationale_summary": "Investigate, remediate, verify.",
    "steps": [
        {"step_id": "s1", "objective": "Gather evidence", "success_signal": "metrics read"},
        {"step_id": "s2", "objective": "Remediate", "success_signal": "action ok"},
    ],
}


class FakeModel:
    """Local model stand-in driven by a list of replies or a callable."""

    def __init__(self, replies: Any = None, plan: dict[str, Any] | None = None) -> None:
        self._replies = replies
        self._plan = plan or VALID_PLAN
        self.calls = 0
        self.seen_messages: list[list[dict[str, Any]]] = []

    def structured(self, messages, schema_name, schema):
        return dict(self._plan)

    def decide(self, messages, tools):
        self.calls += 1
        self.seen_messages.append(list(messages))
        if callable(self._replies):
            return self._replies(self.calls, messages)
        if isinstance(self._replies, list):
            index = min(self.calls - 1, len(self._replies) - 1)
            reply = self._replies[index]
            return reply(self.calls) if callable(reply) else reply
        return ModelReply(content="nothing to do")


def tool_reply(name: str, arguments: dict[str, Any], call_id: str = "c1") -> ModelReply:
    return ModelReply(tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])


def repeat_tool(name: str, arguments: dict[str, Any]) -> Any:
    """A model that proposes the same call forever."""
    return lambda n, messages: tool_reply(name, dict(arguments), f"c{n}")


@pytest.fixture
def env():
    return SimulationEnvironment("STUDENT-TEST", "public-a")


@pytest.fixture
def registry(env):
    return ToolRegistry(env)


@pytest.fixture
def trace_path(tmp_path):
    return tmp_path / "trace.jsonl"


@pytest.fixture
def make_controller(registry, trace_path):
    """Builds a controller wired to offline components."""

    def _make(model, approval=None, budget=None, limits=None):
        controller = AgentController(
            model=model,
            tools=registry,
            approval=approval or AlwaysApproveGateway(),
            budget=budget or BudgetManager(),
            trace=TraceRecorder(trace_path),
            limits=limits or Limits(),
        )
        controller.retry_policy.sleeper = lambda seconds: None
        return controller

    return _make


@pytest.fixture
def read_trace(trace_path):
    def _read():
        if not trace_path.exists():
            return []
        return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]

    return _read


def executed_tools(trace_rows) -> list[str]:
    """Names of tools that actually reached the simulator."""
    return [
        row["payload"]["call"]["name"]
        for row in trace_rows
        if row["event"] == "tool_result"
    ]


def blocked_results(trace_rows) -> list[dict[str, Any]]:
    """Observations produced by Python without touching the simulator."""
    return [row["payload"]["result"] for row in trace_rows if row["event"] == "tool_blocked"]
