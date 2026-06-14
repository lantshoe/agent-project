# Multi-Agent AI System

A production-grade multi-agent AI system built with LangGraph, supporting parallel task execution, long-term memory, structured observability, and specialist subagents coordinated by a supervisor.

---

## Overview

This project started as a single ReAct agent and evolved through three phases into a full multi-agent orchestration system. It can decompose complex tasks, delegate to specialist subagents running in parallel, verify each step's result, and synthesize a final answer — all with persistent memory across sessions.

**Example task:**
```
"Research the top 3 AI companies by revenue in 2025,
then create an Excel file called ai_companies.xlsx
with their names and revenue figures."
```

The system automatically:
- Assigns a **research agent** to search the web in parallel
- Assigns a **data agent** to create and populate the Excel file
- Passes research results as structured context to the data agent
- Synthesizes a final answer from both agents' outputs

---

## Architecture

```
User input
    ↓
Supervisor
  ├── decomposes task into typed Tasks
  ├── dispatches independent tasks in parallel (asyncio.gather)
  ├── passes file_state + results between dependent tasks
  └── aggregates SubagentResults into final answer
        ↓
  ┌─────────────────────────────────────┐
  │         Specialist Subagents        │
  │                                     │
  │  Research Agent   Data Agent        │
  │  ─────────────    ──────────        │
  │  web_search       create_workbook   │
  │  summarize        write_sheet       │
  │                   calculator        │
  │                   read_sheet        │
  │                                     │
  │         Generalist Agent            │
  │         (all tools)                 │
  └─────────────────────────────────────┘
        ↓
  Shared Infrastructure (all agents)
  ─────────────────────────────────────
  ReAct loop       verify_step
  budget guard     run_logger
  arg_validator    long_term_memory
  tool_node        FAISS embeddings
```

Each subagent runs its own independent ReAct loop:
```
LLM → should_use_tool → tools → budget → verifier → LLM → ...
```

---

## Features

### Core Execution
| Feature | Description |
|---------|-------------|
| ReAct loop | Single clean path: think → tool → observe → repeat |
| Parallel tool execution | Independent tool calls run via `asyncio.gather()` |
| Budget guard | Hard ceiling on steps + wall-clock time, graceful exit |
| Step verifier | After every tool: continue / retry / skip / escalate |
| Retry hints | Targeted hints injected into LLM context on retry |

### Multi-Agent (Phase 3)
| Feature | Description |
|---------|-------------|
| Supervisor agent | Decomposes tasks, dispatches in parallel, aggregates results |
| Research specialist | Focused on web search and summarization |
| Data specialist | Focused on files, Excel, and calculations |
| Generalist agent | Handles tasks that don't fit a specialist |
| File state propagation | Downstream agents know what files/sheets already exist |
| Parallel subagent dispatch | Independent tasks run simultaneously across agents |
| Partial failure handling | continue / retry / skip / escalate per task |

### Tool System
| Tool | Description |
|------|-------------|
| `web_search` | DuckDuckGo search with async lock (thread-safe) |
| `calculator` | Safe math expression evaluator |
| `read_file` / `write_file` / `list_files` | Sandboxed filesystem tools |
| MCP filesystem server | Full filesystem access via Model Context Protocol |
| MCP Excel server | `create_workbook`, `write_sheet`, `read_sheet`, `filter_rows`, etc. |

### Memory System
| Feature | Description |
|---------|-------------|
| Episodic memory | Past runs stored and retrieved via FAISS semantic search |
| User profile | LLM-merged profile updated after every run |
| Learned lessons | Extracted from evaluation, stored as individually searchable records |
| Context injection | Relevant memory injected as context before each run |
| Self-evaluation | LLM scores each run: quality, efficiency, learnings |

### Observability
| Feature | Description |
|---------|-------------|
| Structured run log | JSON file per run — steps, timings, verifier decisions, tokens |
| Token tracking | Input/output tokens logged per LLM call |
| Debug logging | Structured console logs across all layers |
| Per-agent logs | Each subagent writes its own run log |

### Safety
| Feature | Description |
|---------|-------------|
| Sandbox filesystem | All file ops restricted to `agent_workspace/`, path escape blocked |
| Arg validator | Strips unknown args, coerces types, reports missing required args |
| Async safety | Sync wrapper works inside FastAPI and Jupyter event loops |
| Typed models | Pydantic models for all data — no raw dicts in core paths |

---

## Project Structure

```
agents/
├── core/                        # Pure infrastructure
│   ├── llm.py                   # LLM factory (Claude + Ollama)
│   ├── logger.py                # Structured console logging
│   ├── models.py                # Shared Pydantic models
│   ├── skill_enum.py            # REASONING / CODING / SIMPLE
│   ├── tool_node.py             # Tool execution (single + parallel)
│   ├── verifier.py              # Step verification logic
│   ├── arg_validator.py         # Tool argument validation + coercion
│   └── run_logger.py            # Structured JSON run logging
│   └── supervisor.py            # Multi-agent orchestrator
│
├── myagent/                     # Agent implementations
│   ├── base_agent.py            # Shared ReAct loop (used by all subagents)
│   ├── research_agent.py        # Web search specialist
│   ├── data_agent.py            # File/Excel/calculation specialist
│
├── models/                      # Typed contracts
│   ├── supervisor.py            # Task, SubagentResult, DelegationPlan, etc.
│   └── __init__.py
│
├── memory/                      # Long-term memory
│   ├── long_term.py             # FAISS-based semantic memory store
│   └── evaluator.py             # LLM-based execution quality evaluation
│
├── tools/                       # Tool definitions
│   ├── search.py                # web_search (async, thread-safe)
│   ├── calculator.py            # Safe math evaluator
│   ├── filesystem.py            # read_file, write_file, list_files
│   └── mcp_client.py            # MCP server configuration
│
├── execution/                   # Execution infrastructure
│   └── checkpointer.py          # LangGraph SqliteSaver checkpointing
│
└── utils/
    └── sandbox.py               # Filesystem sandbox enforcement

agent_workspace/                 # Agent's sandboxed working directory
agent_memory/                    # FAISS index + episode records
logs/runs/                       # Structured JSON run logs
checkpoints/                     # LangGraph state checkpoints
mcp_servers/
└── excel_server.py              # Custom MCP Excel server
tests/
├── conftest.py
├── test_models.py               # 30 tests
├── test_verifier.py             # 22 tests
└── test_arg_validator.py        # 22 tests
```

---

## Getting Started

### Requirements

- Python 3.11+
- Node.js (for MCP filesystem server)
- An Anthropic API key **or** a local Ollama instance

### Installation

```bash
git clone 
cd agent-project
pip install -r requirements.txt
npm install -g @modelcontextprotocol/server-filesystem
```

### Configuration

Create a `.env` file in the project root:

```bash
# LLM Provider — choose "claude" or "ollama"
LLM_PROVIDER=claude

# Anthropic (if using Claude)
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-haiku-4-5-20251001

# Ollama (if using local models)
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434

# Paths (optional — these are the defaults)
AGENT_SANDBOX_DIR=./agent_workspace
MEMORY_DIR=./agent_memory
AGENT_LOG_DIR=./logs/runs
AGENT_CHECKPOINT_DIR=./checkpoints

# Embedding model for memory (optional)
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

> ⚠️ Never commit `.env` to git. It's already in `.gitignore`.

### Run the single agent (Phase 2)

```python
from agents.core.agent import run_agent

response = run_agent("Search for the latest AI news and summarize it")
print(response)
```

### Run the multi-agent supervisor (Phase 3)

```python
response = run_multiagent_sync("""
Research the top 3 AI companies by revenue in 2025,
then create an Excel file called ai_companies.xlsx
with their names and revenue figures.
""")
print(response)
```

Or run directly:

```bash
python -m agents.myagent.supervisor
```

### Run the tests

```bash
pytest tests/ -v
```

---

## Configuration Reference

| Env var | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `claude` or `ollama` |
| `ANTHROPIC_API_KEY` | — | Required if `LLM_PROVIDER=claude` |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | Anthropic model string |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `AGENT_SANDBOX_DIR` | `./agent_workspace` | Agent's working directory |
| `MEMORY_DIR` | `./agent_memory` | FAISS memory storage |
| `AGENT_LOG_DIR` | `./logs/runs` | JSON run log location |
| `AGENT_CHECKPOINT_DIR` | `./checkpoints` | LangGraph checkpoint DB |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer for memory |

---

## How the Verifier Works

After every tool execution, the verifier makes one of four decisions:

| Decision | When | What happens |
|----------|------|-------------|
| `continue` | Tool succeeded | LLM proceeds to next step |
| `retry` | Tool failed, retries remaining | Hint injected into context, LLM retries |
| `skip` | Tool failed, retries exhausted, non-critical | Skip message injected, continue |
| `escalate` | Tool failed, retries exhausted, critical tool | Stop message injected, surface error |

Critical tools (escalate on failure): `create_workbook`, `write_sheet`, `write_file`

---

## How Memory Works

Each run goes through this cycle:

```
Before run:   retrieve relevant episodes + profile → inject as context
During run:   agent executes with memory-informed context
After run:    evaluate quality → store episode + learnings → update profile
```

Memory is stored in three layers:
- **Episodes** — summaries of past runs with quality scores
- **Learnings** — individual lessons extracted from each run
- **Plan templates** — good execution plans for reuse on similar tasks

All stored in FAISS for semantic similarity retrieval — similar future tasks get relevant past experience automatically.

---

## Reading Run Logs

After each run, a JSON file is written to `logs/runs/`:

```json
{
  "run_id": "2026-06-12_14-32-01_default",
  "duration_s": 68.4,
  "user_input": "Research the top 3 AI companies...",
  "budget": {
    "steps_used": 5,
    "max_steps": 15,
    "seconds_used": 68.4,
    "exhausted": false
  },
  "tokens": {
    "total_input": 4821,
    "total_output": 892,
    "total": 5713
  },
  "steps": [
    {"index": 1, "tool": "web_search", "status": "success", "verifier": "continue", "duration_ms": 2340},
    {"index": 2, "tool": "create_workbook", "status": "success", "verifier": "continue", "duration_ms": 890}
  ],
  "verifier_summary": {"continue": 5, "retry": 1, "skip": 0, "escalate": 0},
  "evaluation": {"quality": "good", "efficiency_score": 8}
}
```

Quick view of last run:
```bash
cat logs/runs/$(ls logs/runs/ | tail -1) | python -m json.tool
```

---

## Development History

### Phase 1 — Stabilize
Fixed 9 bugs in the original codebase: empty `StepRecord` on error, `json.loads` crash on plain-text tool output, final answer with no tool context, orphaned message history, executor wrong-tool patching, planner retry loop, raw dicts replaced with Pydantic models, `arg_validator` Pydantic schema API fix, `asyncio.run` in running event loops.

### Phase 2 — Clean ReAct Architecture
Replaced the planner/executor split with a single ReAct loop. Added step verifier (continue/retry/skip/escalate), budget guard (max steps + max time), parallel tool execution via `asyncio.gather`, memory injection before run, and structured run logging.

### Phase 3 — Multi-Agent Orchestration
Added supervisor agent with dynamic task decomposition, three specialist subagents (research, data, generalist), parallel subagent dispatch, typed `Task`/`SubagentResult`/`DelegationPlan` contracts, file state propagation between dependent tasks, per-agent run logs, and LangGraph checkpointing.

---

## Roadmap

- **Phase 4** — DAG-based workflow pipelines for repeated task patterns
- Streaming responses (token-by-token output)
- Human-in-the-loop (pause before irreversible actions)
- FastAPI interface
- Skill-based model routing (cheap model for simple tasks)
- Context window management (summarize old messages)
- Browser tool integration
- Code execution tool

---

## License

MIT