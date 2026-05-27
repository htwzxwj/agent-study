"""
Week 1 — Agent Trace Logger
=============================
Records every step of an agent's execution for analysis.

The key insight from Week 1's lab: "identify where reasoning happens
vs. where tools fire." This module makes that distinction visible.

A trace consists of:
  - LLM calls (reasoning): prompt → response, with token counts and timing
  - Tool calls (action): tool name → input → output
  - Observations: what the agent perceived from the environment
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepType(Enum):
    REASONING = "reasoning"   # LLM thinking / prompt-response
    ACTION = "action"         # Tool call or environment interaction
    OBSERVATION = "observation"  # Result from environment or tool
    SYSTEM = "system"         # System events (start, end, error)


@dataclass
class TraceStep:
    """A single step in an agent's execution trace."""

    step_type: StepType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration_ms: float | None = None

    def format(self, verbose: bool = False) -> str:
        icons = {
            StepType.REASONING: "THINK",
            StepType.ACTION: "ACT",
            StepType.OBSERVATION: "OBS",
            StepType.SYSTEM: "SYS",
        }
        icon = icons[self.step_type]
        duration = f" ({self.duration_ms:.0f}ms)" if self.duration_ms else ""

        lines = [f"[{icon}] {self.content}{duration}"]

        if verbose and self.metadata:
            for key, value in self.metadata.items():
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


class AgentTrace:
    """Records and analyzes an agent's full execution trace.

    Usage:
        trace = AgentTrace(task="Find the capital of France")
        trace.add_reasoning("The user asks about France's capital...")
        trace.add_action("search", {"query": "capital of France"})
        trace.add_observation("Paris is the capital of France")
        trace.print_summary()
    """

    def __init__(self, task: str):
        self.task = task
        self.steps: list[TraceStep] = []
        self.start_time = time.time()
        self._add(StepType.SYSTEM, f"Task: {task}")

    def _add(self, step_type: StepType, content: str, **metadata: Any) -> TraceStep:
        step = TraceStep(step_type=step_type, content=content, metadata=metadata)
        self.steps.append(step)
        return step

    def add_reasoning(self, thought: str, tokens_in: int = 0, tokens_out: int = 0) -> TraceStep:
        """Record an LLM reasoning step (the 'Think' in ReAct)."""
        return self._add(
            StepType.REASONING,
            thought,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    def add_action(self, tool_name: str, tool_input: dict[str, Any], duration_ms: float | None = None) -> TraceStep:
        """Record a tool/action step (the 'Act' in ReAct)."""
        return self._add(
            StepType.ACTION,
            f"Call tool: {tool_name}",
            tool_name=tool_name,
            tool_input=json.dumps(tool_input, ensure_ascii=False),
            duration_ms=duration_ms,
        )

    def add_observation(self, observation: str, source: str = "environment") -> TraceStep:
        """Record an observation from the environment or tool result."""
        return self._add(
            StepType.OBSERVATION,
            observation,
            source=source,
        )

    def add_system(self, message: str) -> TraceStep:
        """Record a system event."""
        return self._add(StepType.SYSTEM, message)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def reasoning_steps(self) -> list[TraceStep]:
        return [s for s in self.steps if s.step_type == StepType.REASONING]

    def action_steps(self) -> list[TraceStep]:
        return [s for s in self.steps if s.step_type == StepType.ACTION]

    def total_tokens(self) -> tuple[int, int]:
        """Return (total_input_tokens, total_output_tokens)."""
        tin = sum(s.metadata.get("tokens_in", 0) for s in self.steps)
        tout = sum(s.metadata.get("tokens_out", 0) for s in self.steps)
        return tin, tout

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a concise summary of the trace."""
        tin, tout = self.total_tokens()
        reasoning = len(self.reasoning_steps())
        actions = len(self.action_steps())
        elapsed = self.elapsed_seconds()

        print("\n" + "=" * 60)
        print("TRACE SUMMARY")
        print("=" * 60)
        print(f"  Task:           {self.task}")
        print(f"  Total steps:    {len(self.steps)}")
        print(f"  Reasoning:      {reasoning} steps")
        print(f"  Actions:        {actions} steps")
        print(f"  Tokens (in/out): {tin} / {tout}")
        print(f"  Wall time:      {elapsed:.2f}s")
        print("=" * 60)

    def print_full(self, verbose: bool = False) -> None:
        """Print the full annotated trace."""
        print("\n" + "=" * 60)
        print(f"FULL TRACE — {self.task}")
        print("=" * 60)
        for i, step in enumerate(self.steps):
            print(f"\nStep {i}:")
            print(step.format(verbose=verbose))
        self.print_summary()

    def to_dict(self) -> dict[str, Any]:
        """Export trace as a serializable dict (for JSON export)."""
        return {
            "task": self.task,
            "elapsed_seconds": self.elapsed_seconds(),
            "total_tokens_in": self.total_tokens()[0],
            "total_tokens_out": self.total_tokens()[1],
            "steps": [
                {
                    "type": s.step_type.value,
                    "content": s.content,
                    "metadata": s.metadata,
                    "timestamp": s.timestamp,
                    "duration_ms": s.duration_ms,
                }
                for s in self.steps
            ],
        }

    def export_json(self, path: str) -> None:
        """Write trace to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Trace exported to {path}")


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

def demo():
    """Demonstrate the trace logger with a mock agent run."""
    trace = AgentTrace(task="What is the capital of France?")

    trace.add_reasoning(
        "The user is asking about the capital of France. I should search for this information.",
        tokens_in=25,
        tokens_out=18,
    )

    trace.add_action("search", {"query": "capital of France"}, duration_ms=120)

    trace.add_observation(
        "Paris is the capital and most populous city of France.",
        source="search_results",
    )

    trace.add_reasoning(
        "Based on the search results, the capital of France is Paris. I can now answer.",
        tokens_in=45,
        tokens_out=12,
    )

    trace.add_action("respond", {"answer": "The capital of France is Paris."}, duration_ms=5)

    trace.add_observation("Response delivered to user.", source="system")

    trace.print_full(verbose=True)

    # Export to JSON
    trace.export_json("phase1/week1/sample_trace.json")


if __name__ == "__main__":
    demo()
