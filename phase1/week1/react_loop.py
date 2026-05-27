"""
Week 1 — ReAct Loop Implementation
====================================
A from-scratch implementation of the ReAct (Reasoning + Acting) pattern.

Paper: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2023)

The core loop:
  1. PERCEIVE  — receive user input / environment state
  2. THINK     — LLM reasons about what to do next
  3. ACT       — LLM chooses an action (tool call or final answer)
  4. OBSERVE   — execute the action, get the result
  5. GOTO 2    — feed observation back into the next thinking step

Key insight: the LLM's reasoning is interleaved with tool execution.
The model doesn't just generate an answer — it thinks step-by-step,
uses tools to gather information, and adjusts its reasoning based on results.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import anthropic
from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam

from shared.utils.config import ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, DEFAULT_MODEL, MAX_TOKENS
from phase1.week1.agent_trace import AgentTrace


# ---------------------------------------------------------------------------
# Tool registry — simple dict-based, no decorators yet (that's Week 4)
# ---------------------------------------------------------------------------

ToolFn = Callable[..., str]


def make_tool(name: str, description: str, parameters: dict[str, Any]) -> ToolParam:
    """Create a tool definition for the Anthropic API."""
    # Extract required field names and strip the "required" key from each param
    required_fields = []
    clean_params = {}
    for param_name, param_def in parameters.items():
        clean_def = {k: v for k, v in param_def.items() if k != "required"}
        clean_params[param_name] = clean_def
        if param_def.get("required", False):
            required_fields.append(param_name)
    return ToolParam(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": clean_params,
            "required": required_fields,
        },
    )


# ---------------------------------------------------------------------------
# Built-in tools for the demo agent
# ---------------------------------------------------------------------------

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return f"Error: invalid characters in expression: {expression}"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def search(query: str) -> str:
    """Mock search tool — returns hardcoded results for demo purposes."""
    knowledge = {
        "capital of france": "Paris is the capital and most populous city of France.",
        "population of tokyo": "Tokyo has a population of approximately 14 million people in the city proper.",
        "python creator": "Python was created by Guido van Rossum, first released in 1991.",
        "speed of light": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
        "water boiling point": "Water boils at 100°C (212°F) at standard atmospheric pressure.",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower or any(word in query_lower for word in key.split()):
            return value
    return f"No results found for: {query}"


def read_file(filepath: str) -> str:
    """Read a file's contents (sandboxed to project directory)."""
    import os
    safe_path = os.path.abspath(filepath)
    if not safe_path.startswith(os.path.abspath(".")):
        return "Error: access denied — path outside project directory"
    try:
        with open(safe_path) as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: file not found: {filepath}"
    except Exception as e:
        return f"Error reading file: {e}"


# Tool definitions for the Anthropic API
TOOL_DEFS: list[ToolParam] = [
    make_tool(
        "calculator",
        "Evaluate a mathematical expression. Use Python syntax.",
        {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate, e.g. '2 + 3 * 4'",
                "required": True,
            },
        },
    ),
    make_tool(
        "search",
        "Search for factual information. Returns relevant text from a knowledge base.",
        {
            "query": {
                "type": "string",
                "description": "The search query",
                "required": True,
            },
        },
    ),
    make_tool(
        "read_file",
        "Read the contents of a file in the project directory.",
        {
            "filepath": {
                "type": "string",
                "description": "Path to the file to read",
                "required": True,
            },
        },
    ),
]

# Tool name → function mapping
TOOL_REGISTRY: dict[str, ToolFn] = {
    "calculator": calculator,
    "search": search,
    "read_file": read_file,
}


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful assistant that uses tools to answer questions.

When you receive a question:
1. THINK about what information you need and how to get it
2. Use tools to gather information
3. THINK about the results and whether you need more information
4. Provide a final answer when you have enough information

Always show your reasoning. Use the tools available to you — don't guess \
when you can look something up or calculate it.
"""


class ReActAgent:
    """A ReAct (Reasoning + Acting) agent built from scratch.

    No framework — just raw API calls + a simple loop.

    Attributes:
        client: Anthropic API client
        model: Model ID to use
        tools: Tool definitions for the API
        tool_fns: Mapping of tool names to callable functions
        max_turns: Maximum reasoning turns before forced stop
        trace: Execution trace for analysis
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        tools: list[ToolParam] | None = None,
        tool_fns: dict[str, ToolFn] | None = None,
        max_turns: int = 10,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        client_kwargs: dict[str, Any] = {"api_key": ANTHROPIC_API_KEY}
        if ANTHROPIC_BASE_URL:
            client_kwargs["base_url"] = ANTHROPIC_BASE_URL
        self.client = anthropic.Anthropic(**client_kwargs)
        self.model = model
        self.tools = tools or TOOL_DEFS
        self.tool_fns = tool_fns or TOOL_REGISTRY
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self.messages: list[MessageParam] = []
        self.trace: AgentTrace | None = None

    def run(self, user_input: str, verbose: bool = False) -> str:
        """Run the ReAct loop on a user query.

        Returns the agent's final answer as a string.
        """
        self.trace = AgentTrace(task=user_input)
        self.messages = [{"role": "user", "content": user_input}]

        if verbose:
            print(f"\n{'='*60}")
            print(f"TASK: {user_input}")
            print(f"{'='*60}")

        for turn in range(self.max_turns):
            # --- THINK: Call the LLM ---
            if verbose:
                print(f"\n--- Turn {turn + 1} ---")

            llm_start = time.time()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                tools=self.tools,
                messages=self.messages,
            )
            llm_duration = (time.time() - llm_start) * 1000

            # Process response blocks
            tool_calls = []
            text_blocks = []

            for block in response.content:
                if block.type == "text":
                    text_blocks.append(block.text)
                    if verbose:
                        print(f"\n[THINK] {block.text}")
                    self.trace.add_reasoning(
                        block.text,
                        tokens_in=response.usage.input_tokens,
                        tokens_out=response.usage.output_tokens,
                    )
                elif block.type == "tool_use":
                    tool_calls.append(block)
                    if verbose:
                        print(f"\n[ACT] {block.name}({json.dumps(block.input, ensure_ascii=False)})")

            # Add assistant response to message history
            self.messages.append({"role": "assistant", "content": response.content})

            # If no tool calls, we're done — the text is the final answer
            if not tool_calls:
                final_answer = "\n".join(text_blocks) if text_blocks else ""
                self.trace.add_system("Agent provided final answer")
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"FINAL ANSWER: {final_answer}")
                    print(f"{'='*60}")
                return final_answer

            # --- ACT: Execute tool calls ---
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.name
                tool_input = tool_call.input

                # Execute the tool
                if tool_name in self.tool_fns:
                    tool_start = time.time()
                    result = self.tool_fns[tool_name](**tool_input)
                    tool_duration = (time.time() - tool_start) * 1000
                else:
                    result = f"Error: unknown tool '{tool_name}'"
                    tool_duration = 0

                self.trace.add_action(
                    tool_name,
                    tool_input,
                    duration_ms=tool_duration,
                )

                if verbose:
                    result_preview = result[:200] + "..." if len(result) > 200 else result
                    print(f"[OBS] {result_preview}")

                self.trace.add_observation(result, source=tool_name)
                tool_results.append(ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id=tool_call.id,
                    content=result,
                ))

            # Add tool results to message history
            self.messages.append({"role": "user", "content": tool_results})

        # Max turns reached
        self.trace.add_system(f"Max turns ({self.max_turns}) reached — stopping")
        return "I couldn't complete the task within the allowed number of turns."

    def print_trace(self, verbose: bool = False) -> None:
        """Print the full execution trace."""
        if self.trace:
            self.trace.print_full(verbose=verbose)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive():
    """Run the agent in interactive mode — type questions, get answers."""
    agent = ReActAgent()

    print("=" * 60)
    print("ReAct Agent — Interactive Mode")
    print("Type 'quit' to exit, 'trace' to show last trace")
    print("Tools: calculator, search, read_file")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "trace":
            agent.print_trace(verbose=True)
            continue

        answer = agent.run(user_input, verbose=True)
        print(f"\nAnswer: {answer}")


# ---------------------------------------------------------------------------
# Demo: run a few example queries
# ---------------------------------------------------------------------------

def demo():
    """Run the ReAct agent on example queries to demonstrate the loop."""
    agent = ReActAgent()

    queries = [
        "What is the capital of France?",
        "Calculate (15 * 7) + (23 / 4.6)",
        "Who created Python and what year was it first released?",
    ]

    for query in queries:
        answer = agent.run(query, verbose=True)
        agent.print_trace(verbose=True)
        print(f"\n{'─' * 60}\n")


if __name__ == "__main__":
    import sys
    if "--interactive" in sys.argv or "-i" in sys.argv:
        interactive()
    else:
        demo()
