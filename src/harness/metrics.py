"""
Central Prometheus metric definitions for the MCP Agent Harness.

All layers import from here so metric names are defined once.
The /metrics endpoint in main.py serves these via prometheus_client.
"""
from prometheus_client import Counter, Gauge, Histogram

# ── SecurityGuard ─────────────────────────────────────────────────────────────

SECURITY_CHECKS = Counter(
    "harness_security_checks_total",
    "Security guard checks by tool and outcome",
    ["tool", "result"],          # result: allowed | blocked
)
SECURITY_BLOCKS = Counter(
    "harness_security_blocks_total",
    "Security blocks by category",
    ["category"],                # prompt_injection | path_traversal | ssrf | sql_timing
)

# ── PermissionResolver ────────────────────────────────────────────────────────

PERMISSION_CHECKS = Counter(
    "harness_permission_checks_total",
    "Permission resolver decisions",
    ["tool", "result"],          # allowed | rejected | hitl_approved | hitl_rejected | hitl_timeout
)

# ── ToolRegistry ──────────────────────────────────────────────────────────────

REGISTRY_CHECKS = Counter(
    "harness_registry_checks_total",
    "Tool registry validation results",
    ["tool", "result"],          # valid | unknown_tool | schema_error
)

# ── BudgetTracker ─────────────────────────────────────────────────────────────

BUDGET_CHECKS = Counter(
    "harness_budget_checks_total",
    "Budget tracker gate outcomes",
    ["result"],                  # allowed | calls_exceeded | tokens_exceeded | redis_error
)

# ── Executor ──────────────────────────────────────────────────────────────────

EXECUTOR_CALLS = Counter(
    "harness_executor_calls_total",
    "Executor outcomes per tool",
    ["tool", "result"],          # success | cache_hit | circuit_open | timeout | error
)
EXECUTOR_LATENCY = Histogram(
    "harness_executor_latency_seconds",
    "MCP tool call wall time (excludes cache hits)",
    ["tool"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)
CIRCUIT_BREAKER_OPEN = Gauge(
    "harness_circuit_breaker_open",
    "1 if the circuit is OPEN for this tool, 0 otherwise",
    ["tool"],
)

# ── AuditLogger ───────────────────────────────────────────────────────────────

AUDIT_WRITES = Counter(
    "harness_audit_writes_total",
    "Audit log write outcomes",
    ["result"],                  # success | dlq | dlq_failed
)
AUDIT_DLQ_DEPTH = Gauge(
    "harness_audit_dlq_depth",
    "Records currently waiting in the audit dead-letter queue",
)

# ── Pipeline (end-to-end) ─────────────────────────────────────────────────────

PIPELINE_CALLS = Counter(
    "harness_pipeline_calls_total",
    "Total pipeline executions by tool and final status",
    ["tool", "result_status"],
)
PIPELINE_LATENCY = Histogram(
    "harness_pipeline_latency_seconds",
    "End-to-end latency for a full pipeline run (all 6 layers)",
    ["tool"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 30.0, 60.0],
)

# ── LLM / Agentic loop ────────────────────────────────────────────────────────

LLM_TURNS = Counter(
    "harness_llm_turns_total",
    "LLM conversation turns",
    ["result"],                  # tool_calls | final_answer
)
LLM_TOKENS = Counter(
    "harness_llm_tokens_total",
    "Cumulative tokens consumed (prompt + completion)",
)
CHAT_REQUESTS = Counter(
    "harness_chat_requests_total",
    "POST /chat requests",
    ["result"],                  # success | error | max_turns
)
