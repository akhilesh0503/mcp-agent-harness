from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    READ_ONLY  = "read_only"
    NETWORK    = "network"
    DESTRUCTIVE = "destructive"


class ResultStatus(str, Enum):
    PENDING         = "pending"
    SUCCESS         = "success"
    ERROR           = "error"
    REJECTED        = "rejected"
    SECURITY_BLOCKED = "security_blocked"
    BUDGET_EXCEEDED = "budget_exceeded"
    CIRCUIT_OPEN    = "circuit_open"
    HITL_TIMEOUT    = "hitl_timeout"
    TIMEOUT         = "timeout"


# ── Core pipeline models ───────────────────────────────────────────────────────

class ToolCall(BaseModel):
    """Represents a single tool invocation returned by the LLM."""
    tool_call_id: str
    tool_name: str
    arguments: dict


class ToolResult(BaseModel):
    """The content injected back into the LLM conversation after tool execution."""
    tool_call_id: str
    content: str
    is_error: bool = False


class PipelineContext(BaseModel):
    """
    Carrier object that flows through all 6 harness layers.
    Each layer reads from it and writes its findings back into it.
    The AuditLogger reads the final state to write one audit_log row.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    trace_id: UUID = Field(default_factory=uuid4)
    session_id: str
    tool_call: ToolCall

    # Set by PermissionResolver
    risk_level: RiskLevel | None = None
    permission_granted: bool = False

    # Set by Executor
    cache_hit: bool = False

    # Set by whichever layer finalises the call
    result: ToolResult | None = None
    result_status: ResultStatus = ResultStatus.PENDING
    rejected_at_layer: str | None = None
    error_message: str | None = None

    # Timing — latency_ms filled in by pipeline orchestrator
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    latency_ms: int | None = None

    @property
    def input_hash(self) -> str:
        """SHA-256 of tool name + sorted arguments — stored in audit_log."""
        raw = json.dumps(
            {"tool": self.tool_call.tool_name, "args": self.tool_call.arguments},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    @property
    def trace_id_str(self) -> str:
        return str(self.trace_id)


# ── Budget model ───────────────────────────────────────────────────────────────

class BudgetStatus(BaseModel):
    session_id: str
    token_count: int
    call_count: int
    token_limit: int
    call_limit: int

    @property
    def tokens_remaining(self) -> int:
        return self.token_limit - self.token_count

    @property
    def calls_remaining(self) -> int:
        return self.call_limit - self.call_count

    @property
    def is_exhausted(self) -> bool:
        return self.token_count >= self.token_limit or self.call_count >= self.call_limit


# ── Agentic loop models ────────────────────────────────────────────────────────

class ConversationMessage(BaseModel):
    """One turn in the multi-turn conversation sent to the LLM."""
    role: str                         # user | assistant | tool
    content: str | list | None = None
    tool_calls: list[dict] | None = None   # raw OpenAI-format tool call objects
    tool_call_id: str | None = None        # set when role == "tool"


# ── FastAPI request / response ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None     # auto-generated UUID if omitted
    max_turns: int = Field(default=10, ge=1, le=50)


class ChatResponse(BaseModel):
    session_id: str
    response: str
    trace_ids: list[str]
    tool_calls_made: int
    total_tokens: int


# ── Human-in-the-loop ─────────────────────────────────────────────────────────

class HITLApprovalRequest(BaseModel):
    trace_id: str
    session_id: str
    tool_name: str
    tool_input: dict
    risk_level: str


class HITLDecision(BaseModel):
    decision: str          # approved | rejected
    decided_by: str = "system"
