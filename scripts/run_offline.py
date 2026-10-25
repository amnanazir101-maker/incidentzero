"""Run the agent end-to-end without Groq.

Useful for development, for the report's evaluation table, and for demonstrating
the controller in a viva without spending provider quota. The reliability logic
lives in the controller, so swapping the model adapter must not change how the
agent handles failures (R14).

Examples:

    python scripts/run_offline.py --student-id 24I-8070 --scenario public-a
    python scripts/run_offline.py --student-id 24I-8070 --all
    python scripts/run_offline.py --student-id 24I-8070 --scenario public-a --deny
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from incidentzero.agent.controller import AgentController
from incidentzero.approval.gateway import (
    AlwaysApproveGateway,
    AlwaysDenyGateway,
    ConsoleApprovalGateway,
)
from incidentzero.environment.engine import SimulationEnvironment
from incidentzero.model.heuristic import HeuristicModelClient
from incidentzero.telemetry.budget import BudgetManager
from incidentzero.telemetry.trace import TraceRecorder
from incidentzero.tools.registry import ToolRegistry

SCENARIOS = ("public-a", "public-b", "public-c")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_offline")
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--scenario", default="public-a")
    parser.add_argument("--all", action="store_true", help="run public-a, public-b and public-c")
    parser.add_argument("--deny", action="store_true", help="deny every high/critical action")
    parser.add_argument("--interactive", action="store_true", help="approve at the console")
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--max-tool-calls", type=int, default=None)
    parser.add_argument("--trace-dir", default="traces")
    return parser


def gateway_for(args: argparse.Namespace):
    if args.interactive:
        return ConsoleApprovalGateway()
    return AlwaysDenyGateway() if args.deny else AlwaysApproveGateway()


def run_one(student_id: str, scenario: str, args: argparse.Namespace) -> dict:
    environment = SimulationEnvironment(student_id, scenario)
    registry = ToolRegistry(environment)

    budget = BudgetManager()
    if args.max_llm_calls is not None:
        budget.max_llm_calls = args.max_llm_calls
    if args.max_tool_calls is not None:
        budget.max_tool_calls = args.max_tool_calls

    suffix = "_denied" if args.deny else ""
    trace_path = Path(args.trace_dir) / f"{student_id}_{scenario}{suffix}_offline.jsonl"

    controller = AgentController(
        model=HeuristicModelClient(),
        tools=registry,
        approval=gateway_for(args),
        budget=budget,
        trace=TraceRecorder(trace_path),
    )

    started = time.monotonic()
    outcome = controller.run()
    elapsed = time.monotonic() - started

    approvals = [
        {"tool": d.tool, "risk": d.risk.value, "outcome": d.reason}
        for d in controller.approval_broker.decisions if d.required
    ]

    print(f"\n{'=' * 62}")
    print(f"  {scenario}   student-id={student_id}")
    print(f"{'=' * 62}")
    print(f"  Terminal outcome  : {outcome.status}")
    print(f"  Summary           : {outcome.summary}")
    print(f"  LLM requests      : {outcome.llm_calls} / {budget.max_llm_calls}")
    print(f"  Tool calls        : {outcome.tool_calls} / {budget.max_tool_calls}")
    print(f"  Plan revisions    : {controller.state.plan_revisions}")
    print(f"  Replan triggers   : {[t.get('status') for t in controller.state.replan_triggers] or '-'}")
    print(f"  High-risk actions : {approvals or '-'}")
    print(f"  Recovery proof    : criteria_met={controller.state.recovery_criteria_met} "
          f"evidence={controller.state.latest_verify_evidence_id}")
    print(f"  Model retries     : {controller.state.model_retries}")
    print(f"  Validation blocks : {controller.state.validation_failures}")
    print(f"  Evidence ids      : {len(outcome.evidence_ids)}")
    print(f"  Runtime           : {elapsed:.2f}s")
    print(f"  Trace             : {trace_path}")

    return {"scenario": scenario, "elapsed_seconds": round(elapsed, 2), **asdict(outcome)}


def main() -> None:
    args = build_parser().parse_args()
    scenarios = SCENARIOS if args.all else (args.scenario,)
    results = [run_one(args.student_id, scenario, args) for scenario in scenarios]

    if len(results) > 1:
        print(f"\n{'=' * 62}")
        print("  SUMMARY")
        print(f"{'=' * 62}")
        print(f"  {'scenario':<12}{'outcome':<18}{'llm':>5}{'tools':>7}{'secs':>8}")
        for r in results:
            print(f"  {r['scenario']:<12}{r['status']:<18}{r['llm_calls']:>5}"
                  f"{r['tool_calls']:>7}{r['elapsed_seconds']:>8.2f}")
    print()
    print(json.dumps(results, indent=2, default=str)[:0] or "", end="")


if __name__ == "__main__":
    main()
