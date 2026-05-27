"""
Week 1 — Agent Taxonomy
========================
Concrete implementations of different agent types from Wooldridge & Jennings (1995).

Agent types:
  1. Simple Reflex Agent   — stimulus → response (no memory, no goals)
  2. Model-Based Agent     — maintains internal state of the world
  3. Goal-Based Agent      — selects actions that achieve goals
  4. Utility-Based Agent   — maximizes a utility function over outcomes
  5. ReAct Agent           — LLM-powered perceive → think → act loop (Week 1 focus)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Environment: a simple grid world for demonstrating agent perception
# ---------------------------------------------------------------------------

@dataclass
class GridWorld:
    """A tiny 5x5 grid with obstacles and a goal."""

    width: int = 5
    height: int = 5
    goal: tuple[int, int] = (4, 4)
    obstacles: set[tuple[int, int]] = field(default_factory=lambda: {(1, 1), (2, 3), (3, 1)})
    agent_pos: tuple[int, int] = (0, 0)

    def percept(self) -> dict[str, Any]:
        """Return what the agent can observe from the current position."""
        x, y = self.agent_pos
        return {
            "position": self.agent_pos,
            "goal": self.goal,
            "distance_to_goal": abs(self.goal[0] - x) + abs(self.goal[1] - y),
            "at_goal": self.agent_pos == self.goal,
            "adjacent_obstacles": [
                (dx, dy)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                if (x + dx, y + dy) in self.obstacles
            ],
        }

    def apply_action(self, action: str) -> bool:
        """Execute an action. Returns True if the action was valid."""
        x, y = self.agent_pos
        moves = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        if action not in moves:
            return False
        dx, dy = moves[action]
        nx, ny = x + dx, y + dy
        if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in self.obstacles:
            self.agent_pos = (nx, ny)
            return True
        return False

    def render(self) -> str:
        """ASCII visualization of the grid."""
        lines = []
        for row in range(self.height):
            line = ""
            for col in range(self.width):
                pos = (col, row)
                if pos == self.agent_pos:
                    line += " A "
                elif pos == self.goal:
                    line += " G "
                elif pos in self.obstacles:
                    line += " # "
                else:
                    line += " . "
            lines.append(line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Simple Reflex Agent — no memory, pure stimulus → response
# ---------------------------------------------------------------------------

class ReflexAgent:
    """The simplest possible agent: react to the current percept only.

    Rule: move toward the goal (greedy, no planning, no memory).
    This agent can get stuck in loops or behind obstacles.
    """

    def __init__(self, env: GridWorld):
        self.env = env
        self.trace: list[dict] = []

    def run(self, max_steps: int = 20) -> str:
        for step in range(max_steps):
            percept = self.env.percept()
            action = self._rule(percept)
            self.env.apply_action(action)
            self.trace.append({"step": step, "percept": percept, "action": action})
            if percept["at_goal"]:
                return f"Reached goal in {step} steps"
        return f"Failed after {max_steps} steps"

    def _rule(self, percept: dict) -> str:
        """Hardcoded rule: move horizontally first, then vertically."""
        x, y = self.env.agent_pos
        gx, gy = self.env.goal
        if x < gx:
            return "right"
        if x > gx:
            return "left"
        if y < gy:
            return "down"
        return "up"


# ---------------------------------------------------------------------------
# 2. Goal-Based Agent — has a goal, reasons about which actions achieve it
# ---------------------------------------------------------------------------

class GoalBasedAgent:
    """An agent that maintains a goal and plans a path to it.

    Uses BFS to find a path around obstacles — this is "reasoning"
    that the reflex agent cannot do.
    """

    def __init__(self, env: GridWorld):
        self.env = env
        self.goal: tuple[int, int] = env.goal
        self.trace: list[dict] = []

    def run(self, max_steps: int = 50) -> str:
        path = self._plan()
        if not path:
            return "No path found"
        for step, action in enumerate(path[:max_steps]):
            percept = self.env.percept()
            self.env.apply_action(action)
            self.trace.append({"step": step, "percept": percept, "action": action, "planned": True})
            if self.env.agent_pos == self.goal:
                return f"Reached goal in {step + 1} steps (planned path)"
        return f"Failed after {max_steps} steps"

    def _plan(self) -> list[str]:
        """BFS pathfinding — the 'thinking' part of the agent."""
        from collections import deque

        start = self.env.agent_pos
        queue: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])
        visited = {start}
        moves = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == self.goal:
                return path
            for action, (dx, dy) in moves.items():
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < self.env.width
                    and 0 <= ny < self.env.height
                    and (nx, ny) not in self.env.obstacles
                    and (nx, ny) not in visited
                ):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [action]))
        return []


# ---------------------------------------------------------------------------
# 3. Utility-Based Agent — optimizes a utility function over outcomes
# ---------------------------------------------------------------------------

class UtilityBasedAgent:
    """An agent that chooses actions by maximizing a utility function.

    Unlike goal-based (binary: reached goal or not), this agent considers
    tradeoffs: path length, obstacle proximity, energy cost, etc.
    """

    def __init__(self, env: GridWorld):
        self.env = env
        self.trace: list[dict] = []

    def run(self, max_steps: int = 50) -> str:
        for step in range(max_steps):
            percept = self.env.percept()
            if percept["at_goal"]:
                return f"Reached goal in {step} steps"
            action = self._best_action(percept)
            self.env.apply_action(action)
            self.trace.append({
                "step": step,
                "percept": percept,
                "action": action,
                "utility_scores": self._score_all_actions(percept),
            })
        return f"Failed after {max_steps} steps"

    def _utility(self, pos: tuple[int, int], percept: dict) -> float:
        """Utility function: higher is better.

        Combines: distance to goal (primary), obstacle avoidance (secondary).
        """
        gx, gy = self.env.goal
        distance = abs(gx - pos[0]) + abs(gy - pos[1])
        obstacle_penalty = sum(
            1 for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if (pos[0] + dx, pos[1] + dy) in self.env.obstacles
        )
        return -distance - 0.3 * obstacle_penalty

    def _score_all_actions(self, percept: dict) -> dict[str, float]:
        scores = {}
        x, y = self.env.agent_pos
        moves = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        for action, (dx, dy) in moves.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.env.width and 0 <= ny < self.env.height and (nx, ny) not in self.env.obstacles:
                scores[action] = self._utility((nx, ny), percept)
            else:
                scores[action] = float("-inf")
        return scores

    def _best_action(self, percept: dict) -> str:
        scores = self._score_all_actions(percept)
        return max(scores, key=scores.get)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def demo():
    """Run all agent types on the same environment and compare traces."""
    print("=" * 60)
    print("Agent Taxonomy Demo — Grid World")
    print("=" * 60)

    agents = [
        ("Reflex Agent", ReflexAgent),
        ("Goal-Based Agent", GoalBasedAgent),
        ("Utility-Based Agent", UtilityBasedAgent),
    ]

    for name, AgentClass in agents:
        env = GridWorld()
        agent = AgentClass(env)
        result = agent.run()

        print(f"\n{'─' * 40}")
        print(f"  {name}")
        print(f"{'─' * 40}")
        print(f"  Result: {result}")
        print(f"  Steps taken: {len(agent.trace)}")
        print(f"  Trace: {[t['action'] for t in agent.trace]}")

    print("\n" + "=" * 60)
    print("Key differences:")
    print("  Reflex:  no planning, can get stuck behind obstacles")
    print("  Goal:    plans ahead with BFS, finds optimal path")
    print("  Utility: plans ahead + considers tradeoffs (obstacle proximity)")
    print("=" * 60)


if __name__ == "__main__":
    demo()
