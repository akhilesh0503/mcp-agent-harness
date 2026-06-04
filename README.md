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
- [ ] **Phase 4b** — ToolRegistry + BudgetTracker
- [ ] **Phase 4c** — Executor (circuit breaker + cache + MCP client)
- [ ] **Phase 4d** — AuditLogger (PostgreSQL + DLQ)
- [ ] **Phase 5** — LLM abstraction + Ollama client
- [ ] **Phase 6** — Pipeline orchestrator + FastAPI agentic loop
- [ ] **Phase 7** — Prometheus metrics on all layers
- [ ] **Phase 8** — Grafana dashboard
- [ ] **Phase 9** — Tests (PermissionResolver + BudgetTracker)

## Running Locally

**Prerequisites:** Docker Desktop, Ollama

```bash
# Pull the model
ollama pull qwen2.5:3b

# Start infrastructure
docker compose up postgres redis prometheus grafana -d

# Run MCP server
python -m src.mcp_server.server

# Run harness
uvicorn src.harness.main:app --reload
```

**Grafana:** http://localhost:3000 (admin / admin)  
**Prometheus:** http://localhost:9090  
**Harness API:** http://localhost:8000/docs

## Key Design Decisions

- **Session** = UUID per conversation, auto-generated at request time
- **BudgetTracker** fails closed (Redis down → reject, not allow)
- **Circuit breaker** per tool: 3 failures → open for 30s
- **AuditLogger** uses a dead-letter queue (Redis list) when PostgreSQL is unavailable — no records are lost
- **HITL approval** pauses the agent loop for destructive-risk tools, 30s timeout
- **LLM is swappable** — `OllamaClient` and `ClaudeClient` share a common `LLMClient` interface
