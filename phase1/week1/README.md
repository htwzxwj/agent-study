# Phase 1 — Week 1: What is an AI Agent?

## Learning Objectives

- Understand the difference between reflex, goal-based, and utility-based agents
- Grasp the ReAct (Reasoning + Acting) pattern and why it matters
- Be able to identify where reasoning happens vs. where tools fire in an agent trace

## Key Concepts

### Agent Taxonomy (Wooldridge & Jennings, 1995)

| Type | Characteristics | Example |
|------|----------------|---------|
| Reflex | Stimulus → response, no memory | Thermostat |
| Model-based | Maintains internal world state | Roomba with mapping |
| Goal-based | Plans actions to achieve goals | GPS navigator |
| Utility-based | Maximizes utility over outcomes | Trading bot |
| ReAct | LLM reasons + acts in a loop | This week's implementation |

### The ReAct Loop (Yao et al., 2023)

```
┌──────────┐
│ Perceive │ ← User input / environment state
└────┬─────┘
     ▼
┌──────────┐
│  Think   │ ← LLM reasons about what to do (NON-DETERMINISTIC)
└────┬─────┘
     ▼
┌──────────┐
│   Act    │ ← LLM chooses a tool call (DETERMINISTIC execution)
└────┬─────┘
     ▼
┌──────────┐
│ Observe  │ ← Tool returns a result
└────┬─────┘
     │
     └──→ Back to Think (loop until done)
```

### RAG ≠ Agent

RAG (Retrieval-Augmented Generation) is a single-step process: retrieve → generate.
An agent loops: think → act → observe → think → ... until the task is complete.

## Files

| File | Description |
|------|-------------|
| `taxonomy.py` | Agent taxonomy implementations (reflex, goal-based, utility-based) |
| `agent_trace.py` | Trace logger for recording agent execution |
| `react_loop.py` | ReAct agent implementation (Anthropic API, no framework) |
| `trace_demo.py` | Annotated end-to-end trace demo |

## Running

```bash
# Demo: agent taxonomy comparison
python -m phase1.week1.taxonomy

# Demo: trace logger
python -m phase1.week1.agent_trace

# Demo: ReAct agent on example queries
python -m phase1.week1.react_loop

# Demo: annotated end-to-end trace (main lab exercise)
python -m phase1.week1.trace_demo

# Interactive mode
python -m phase1.week1.react_loop --interactive
```

## Lab Exercise

1. Run `trace_demo.py` and study the annotated output
2. For each step in the trace, identify:
   - Is it reasoning (LLM thinking) or action (tool execution)?
   - What information does the LLM use to decide the next action?
   - What would happen if the tool returned a different result?
3. Modify the task in `trace_demo.py` and observe how the trace changes
4. Add a new tool to `react_loop.py` and see how the agent uses it

## Reading List

- [ ] Wooldridge & Jennings (1995) — "Intelligent Agents: Theory and Practice"
- [ ] Yao et al. (2023) — "ReAct: Synergizing Reasoning and Acting in Language Models"
- [ ] Andrej Karpathy — "Software 2.0"
- [ ] Anthropic — Tool use documentation
