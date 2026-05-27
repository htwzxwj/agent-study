# Agent Study — Progress Tracker

## Phase 1 — Agent Foundations & Mental Models (Weeks 1-3)

### Week 1: What is an AI Agent?
- [x] Environment setup complete
- [x] Agent taxonomy examples (reflex, goal-based, utility-based)
- [x] ReAct loop implementation (from scratch, no framework)
- [x] Agent trace logger
- [x] End-to-end trace demo with annotations
- [ ] Read: Wooldridge & Jennings (1995) — agent taxonomy
- [ ] Read: ReAct paper (Yao et al., 2023)
- [ ] Watch: Andrej Karpathy — Software 2.0
- [ ] Lab: Manually annotate a trace and identify reasoning vs. action steps

### Week 2: LLMs as Reasoning Engines
- [ ] Chain-of-thought prompting
- [ ] Self-consistency decoding
- [ ] Structured output (JSON mode)
- [ ] Context window management
- [ ] Lab: Build a raw reasoning harness

### Week 3: Memory & State
- [ ] Memory taxonomy (in-context, external, episodic)
- [ ] Vector DB retrieval (FAISS, ChromaDB)
- [ ] Entity extraction & knowledge graphs
- [ ] Lab: Persistent memory store from scratch

**Milestone:** Simple ReAct agent with memory — no framework

---

## Phase 2 — Core Agent Architecture (Weeks 4-6)

### Week 4: Tool Use & Function Calling
- [ ] Tool schema design
- [ ] Parallel vs. sequential tool calls
- [ ] Error propagation & security
- [ ] Lab: General-purpose tool registry with decorators

### Week 5: Planning & Task Decomposition
- [ ] MRKL, PAL, Tree of Thought
- [ ] Task graph decomposition
- [ ] Lab: Task graph executor

### Week 6: Evaluation & Reliability
- [ ] Trajectory vs. outcome evaluation
- [ ] LLM-as-judge
- [ ] Lab: Eval harness with 50 test cases

**Milestone:** Evaluated, tested, single-agent system

---

## Phase 3 — Multi-Agent Systems (Weeks 7-9)

### Week 7: Multi-Agent Architecture Patterns
- [ ] Hub-and-spoke vs. peer mesh
- [ ] Role specialization
- [ ] Lab: Orchestrator + 3 specialist agents

### Week 8: Communication & Shared State
- [ ] Message passing vs. shared memory
- [ ] Redis-backed shared blackboard
- [ ] Lab: Async agents with shared state

### Week 9: Safety, Alignment & Control
- [ ] Prompt injection defense
- [ ] Constitutional constraints
- [ ] Lab: Red-team your own agent

**Milestone:** Secure, multi-agent system with human oversight

---

## Phase 4 — Production & Beyond (Weeks 10-12)

### Week 10: Observability & Debugging
- [ ] OpenTelemetry instrumentation
- [ ] Cost dashboards
- [ ] Lab: Full tracing with OTEL spans

### Week 11: Latency, Cost & Model Selection
- [ ] Cascade models
- [ ] Semantic caching
- [ ] Lab: Cut cost by 50% without quality loss

### Week 12: Capstone — IssueForge
- [ ] Explorer agent (AST analysis, semantic search)
- [ ] Planner agent (task DAG generation)
- [ ] Coder agent (patch generation, sandboxed execution)
- [ ] Reviewer agent (constitutional self-critique)
- [ ] Orchestrator (full pipeline)

**Milestone:** Production-ready capstone agent
