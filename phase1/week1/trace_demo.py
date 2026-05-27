"""
Week 1 — Lab: Trace an Agent End-to-End
=========================================
This script runs the ReAct agent on a multi-step task and produces
an annotated trace showing exactly where reasoning happens vs. where
tools fire.

Lab objectives (from the curriculum):
  - Run an agent end-to-end
  - Log every prompt/response
  - Identify where reasoning happens vs. where tools fire
  - Annotate the trace manually

Usage:
    python -m phase1.week1.trace_demo
"""

from __future__ import annotations

import json
import sys
import time

from phase1.week1.react_loop import ReActAgent
from phase1.week1.agent_trace import AgentTrace, StepType


def annotate_trace(trace: AgentTrace) -> None:
    """Print a heavily annotated trace for learning purposes.

    This is the 'manual annotation' part of the lab — we label each step
    with WHY it's reasoning or action, and what pattern it represents.
    """
    print("\n" + "=" * 70)
    print("ANNOTATED TRACE — Week 1 Lab")
    print("=" * 70)
    print(f"\nTask: {trace.task}")
    print(f"Total steps: {len(trace.steps)}")
    print()

    turn = 0
    for i, step in enumerate(trace.steps):
        if step.step_type == StepType.SYSTEM:
            print(f"{'─' * 60}")
            print(f"  [{i}] SYSTEM: {step.content}")
            print(f"{'─' * 60}")
            continue

        if step.step_type == StepType.REASONING:
            turn += 1
            print(f"\n{'─' * 60}")
            print(f"  TURN {turn} — REASONING (the 'Think' in ReAct)")
            print(f"{'─' * 60}")
            print(f"  Step [{i}]: LLM generates thought")
            print(f"  Content: {step.content[:300]}")
            print()
            print(f"  ANNOTATION: This is where the LLM reasons about the task.")
            print(f"  The model decides what information it needs and which tool to use.")
            print(f"  This is NOT deterministic — it's the LLM's best judgment.")
            tin = step.metadata.get("tokens_in", 0)
            tout = step.metadata.get("tokens_out", 0)
            if tin or tout:
                print(f"  Tokens: {tin} in / {tout} out")

        elif step.step_type == StepType.ACTION:
            print(f"\n  ACTION (the 'Act' in ReAct)")
            print(f"  Step [{i}]: Tool call → {step.metadata.get('tool_name', '?')}")
            print(f"  Input: {step.metadata.get('tool_input', '?')}")
            print()
            print(f"  ANNOTATION: This is deterministic execution.")
            print(f"  The LLM chose this tool and these inputs — but the tool itself")
            print(f"  is just a Python function. No LLM involved in this step.")

        elif step.step_type == StepType.OBSERVATION:
            print(f"\n  OBSERVATION (the 'Observe' in ReAct)")
            print(f"  Step [{i}]: Result from {step.metadata.get('source', '?')}")
            print(f"  Content: {step.content[:300]}")
            print()
            print(f"  ANNOTATION: This observation feeds back into the next reasoning")
            print(f"  step. The LLM uses this to decide whether it has enough info")
            print(f"  or needs to call another tool. This is the feedback loop that")
            print(f"  makes ReAct different from simple chain-of-thought.")

    # Summary analysis
    print(f"\n{'=' * 70}")
    print("TRACE ANALYSIS")
    print(f"{'=' * 70}")

    reasoning_steps = trace.reasoning_steps()
    action_steps = trace.action_steps()

    print(f"\n  Reasoning steps: {len(reasoning_steps)}")
    print(f"  Action steps:    {len(action_steps)}")
    print(f"  Ratio:           {len(reasoning_steps) / max(len(action_steps), 1):.1f} reasoning per action")
    print()

    print("  Where reasoning happens:")
    for s in reasoning_steps:
        print(f"    - {s.content[:80]}...")

    print("\n  Where tools fire:")
    for s in action_steps:
        print(f"    - {s.metadata.get('tool_name', '?')}({s.metadata.get('tool_input', '?')})")

    tin, tout = trace.total_tokens()
    print(f"\n  Token budget: {tin} input + {tout} output = {tin + tout} total")
    print(f"  Wall time: {trace.elapsed_seconds():.2f}s")

    print(f"\n{'=' * 70}")
    print("KEY TAKEAWAY")
    print(f"{'=' * 70}")
    print("""
  The ReAct pattern interleaves reasoning (LLM calls) with action (tool calls).

  Reasoning is NON-DETERMINISTIC — the LLM decides what to do next.
  Actions are DETERMINISTIC — tools execute predictably.

  The power of ReAct: the LLM can adjust its plan based on tool results.
  This is what makes it an "agent" rather than a simple prompt chain.
""")


def run_demo_task() -> None:
    """Run a multi-step task that requires both search and calculation.

    This task is designed to show:
    1. The agent decomposing a question into sub-tasks
    2. Using search to gather information
    3. Using calculator to process numbers
    4. Synthesizing results into a final answer
    """
    agent = ReActAgent(max_turns=8)

    task = (
        "I need to know: How many years has Python existed since its first release, "
        "and what is that number multiplied by the speed of light in millions of km/s? "
        "Use your tools to find the exact numbers."
    )

    print("=" * 70)
    print("WEEK 1 LAB — End-to-End Agent Trace")
    print("=" * 70)
    print(f"\nTask: {task}")
    print("\nThis task requires:")
    print("  1. Searching for Python's release year")
    print("  2. Searching for the speed of light")
    print("  3. Calculating years since release")
    print("  4. Multiplying the two values")
    print()

    answer = agent.run(task, verbose=False)

    if agent.trace:
        annotate_trace(agent.trace)
        agent.trace.export_json("phase1/week1/demo_trace.json")

    print(f"\nFinal answer: {answer}")


def run_simple_task() -> None:
    """Run a simpler task for quick testing."""
    agent = ReActAgent(max_turns=5)
    task = "What is 42 * 17 + the number of letters in the word 'extraordinary'?"
    answer = agent.run(task, verbose=True)
    if agent.trace:
        agent.trace.print_full(verbose=True)
    print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    if "--simple" in sys.argv:
        run_simple_task()
    else:
        run_demo_task()
