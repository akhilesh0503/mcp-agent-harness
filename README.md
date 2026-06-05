# MCP Agent Harness

A production-grade MCP server + agent harness built in Python. An LLM (Ollama/Claude) never calls tools directly — every tool call flows through a 6-layer validation and execution pipeline.

## Architecture

```
User Request
     │
  FastAPI (harness)
     │
  Claude / Ollama ──── returns tool_call JSON
     │
  ┌──▼──────────────────────────────────┐
  │  SecurityGuard      (prompt injection, path traversal, SSRF)
  │  PermissionResolver (policy.yaml risk levels + HITL for destructive)
  │  ToolRegistry       (tool exists + JSON Schema validation)
  │  BudgetTracker      (Redis: token spend + call count per session)
  │  Executor           (cache check → circuit breaker → MCP tool call)
  │  AuditLogger        (PostgreSQL write + DLQ fallback, trace_id)
  └─────────────────────────────────────┘
     │
  result injected back → loop until final answer
```

**MCP Server** runs as a separate networked process, exposing 3 tools:
- `postgres_query` — read-only SQL queries
- `http_api_call` — external HTTP requests
- `file_read` — sandboxed file access

## Stack

| Layer | Tech |
|---|---|
| LLM | Ollama (`qwen2.5:3b`) → swappable to Claude |
| API | FastAPI + asyncio |
| Tools | MCP Python SDK |
| Budget tracking | Redis |
| Audit log | PostgreSQL (asyncpg) |
| Observability | Prometheus + Grafana |
| Infra | Docker Compose |

## Build Phases

- [x] **Phase 1** — Infrastructure (docker-compose, .env, init.sql, Prometheus, Grafana, policy.yaml)
- [x] **Phase 2** — MCP Server + 3 tools
- [x] **Phase 3** — Config + Pydantic models
- [x] **Phase 4a** — SecurityGuard + PermissionResolver
- [x] **Phase 4b** — ToolRegistry + BudgetTracker
- [x] **Phase 4c** — Executor (circuit breaker + cache + MCP client)
- [x] **Phase 4d** — AuditLogger (PostgreSQL + DLQ)
- [x] **Phase 5** — LLM abstraction + Ollama client (+ Claude client)
- [x] **Phase 6a** — Pipeline orchestrator
- [x] **Phase 6b** — FastAPI entry point + agentic loop
- [ ] **Phase 7** — Prometheus metrics on all layers
- [ ] **Phase 8** — Grafana dashboard
- [ ] **Phase 9** — Tests (PermissionResolver + BudgetTracker)

## Running Locally

**Prerequisites:** Docker Desktop, Ollama, Python 3.11

```bash
# 1. Create and activate virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# 3. Pull the local model
ollama pull qwen2.5:3b

# 4. Start infrastructure
docker compose up postgres redis prometheus grafana -d

# 5. Run MCP server
python -m src.mcp_server.server

# 6. Run harness
uvicorn src.harness.main:app --reload
```

**Grafana:** http://localhost:3000 (admin / admin)  
**Prometheus:** http://localhost:9090  
**Harness API:** http://localhost:8000/docs

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Send a message; agent reasons + calls tools and returns final answer |
| `GET` | `/health` | Liveness check — shows registered tools + circuit breaker states |
| `GET` | `/budget/{session_id}` | Token and call spend for a session |
| `GET` | `/tools` | List tools registered from MCP server |
| `DELETE` | `/session/{session_id}` | Clear conversation history for a session |
| `POST` | `/hitl/{trace_id}/decide` | Approve or reject a pending destructive tool call |
| `GET` | `/audit/dlq-depth` | Number of records waiting in the audit dead-letter queue |

## Layer Summary

| Layer | File | Responsibility |
|---|---|---|
| 1 SecurityGuard | `layers/security_guard.py` | Prompt injection, path traversal, SSRF, SQL timing — pure regex, zero I/O |
| 2 PermissionResolver | `layers/permission_resolver.py` | policy.yaml risk level lookup; HITL pause + Redis poll for destructive tools |
| 3 ToolRegistry | `layers/tool_registry.py` | Validates tool exists + full JSON Schema argument check |
| 4 BudgetTracker | `layers/budget_tracker.py` | Redis atomic call/token counters per session; fail-closed on Redis down |
| 5 Executor | `layers/executor.py` | Cache → per-tool circuit breaker → MCP call → cache write |
| 6 AuditLogger | `layers/audit_logger.py` | asyncpg INSERT; DLQ fallback to Redis on DB failure; background drainer |

## Key Design Decisions

- **Session** = UUID per conversation, auto-generated at request time
- **BudgetTracker** fails closed — Redis down → reject the call, never silently allow
- **Circuit breaker** is per-tool with independent locks — one failing tool never blocks others
- **AuditLogger DLQ** — failed PostgreSQL writes go to Redis list `audit:dlq`; background drainer replays every 30s with up to 3 retries before abandoning; loss is always logged at CRITICAL
- **HITL approval** pauses the agent loop for destructive-risk tools; polls Redis every 1s up to configurable timeout; fail-closed on timeout
- **Cache key = audit input_hash** — SHA-256(tool + args) is computed once and reused for both Redis cache lookup and the audit log row
- **LLM is swappable** — `OllamaClient` and `ClaudeClient` share a common `LLMClient` interface (Phase 5)
