from __future__ import annotations

import json
import logging
import logging.config
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.config import settings
from src.harness.metrics import CHAT_REQUESTS, LLM_TOKENS, LLM_TURNS
from src.harness.models import (
    ChatRequest,
    ChatResponse,
    ConversationMessage,
    HITLDecision,
    ToolCall,
)
from src.harness.pipeline import Pipeline

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MCP Agent Harness...")
    app.state.pipeline = await Pipeline.create()
    logger.info("Harness ready — visit /docs for the API")
    yield
    await app.state.pipeline.shutdown()


app = FastAPI(
    title="MCP Agent Harness",
    description=(
        "Production LLM agent harness. Every tool call from the LLM passes through "
        "a 6-layer pipeline: SecurityGuard → PermissionResolver → ToolRegistry → "
        "BudgetTracker → Executor → AuditLogger."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _pipeline(req: Request) -> Pipeline:
    return req.app.state.pipeline


_SESSION_TTL = 86_400  # 24 h


async def _load_history(pipeline: Pipeline, session_id: str) -> list[ConversationMessage]:
    raw = await pipeline._redis.get(f"session:messages:{session_id}")
    if not raw:
        return []
    data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    return [ConversationMessage(**m) for m in data]


async def _save_history(
    pipeline: Pipeline,
    session_id: str,
    messages: list[ConversationMessage],
) -> None:
    payload = json.dumps([m.model_dump() for m in messages])
    await pipeline._redis.setex(f"session:messages:{session_id}", _SESSION_TTL, payload)


# ── Agentic loop ──────────────────────────────────────────────────────────────

async def _agentic_loop(
    pipeline: Pipeline,
    session_id: str,
    user_message: str,
    max_turns: int,
) -> ChatResponse:
    """
    Core loop:
      1. Call LLM with full message history + tool definitions
      2. If LLM returns tool_calls → run each through the pipeline, inject results, repeat
      3. If LLM returns text → done, save history, return response
      4. If max_turns exhausted → return with a warning message
    """
    messages = await _load_history(pipeline, session_id)
    messages.append(ConversationMessage(role="user", content=user_message))

    tool_defs       = pipeline.get_tool_definitions()
    trace_ids:  list[str] = []
    total_tokens    = 0
    tool_calls_made = 0

    for turn in range(max_turns):
        logger.debug("Agentic loop turn %d/%d session=%s", turn + 1, max_turns, session_id)

        llm_response = await pipeline.chat(messages, tool_defs)
        total_tokens += llm_response.token_count

        # ── Final answer ──────────────────────────────────────────────────────
        if llm_response.is_final:
            LLM_TURNS.labels(result="final_answer").inc()
            LLM_TOKENS.inc(llm_response.token_count)
            messages.append(
                ConversationMessage(role="assistant", content=llm_response.text)
            )
            await _save_history(pipeline, session_id, messages)
            await pipeline.budget_tracker.increment_tokens(
                session_id, llm_response.token_count
            )
            return ChatResponse(
                session_id=session_id,
                response=llm_response.text or "",
                trace_ids=trace_ids,
                tool_calls_made=tool_calls_made,
                total_tokens=total_tokens,
            )

        # ── Tool calls requested ──────────────────────────────────────────────
        LLM_TURNS.labels(result="tool_calls").inc()
        LLM_TOKENS.inc(llm_response.token_count)
        # Store the assistant's tool-call turn in OpenAI wire format
        raw_tcs = [
            {
                "id":   tc.tool_call_id,
                "type": "function",
                "function": {
                    "name":      tc.tool_name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in llm_response.tool_calls
        ]
        messages.append(
            ConversationMessage(role="assistant", content=None, tool_calls=raw_tcs)
        )

        # Run each tool through the full 6-layer pipeline
        for tc in llm_response.tool_calls:
            tool_call = ToolCall(
                tool_call_id=tc.tool_call_id,
                tool_name=tc.tool_name,
                arguments=tc.arguments,
            )
            result, trace_id = await pipeline.run_tool_call(session_id, tool_call)
            trace_ids.append(trace_id)
            tool_calls_made += 1

            logger.info(
                "Tool call: tool=%s trace=%s error=%s session=%s",
                tc.tool_name, trace_id, result.is_error, session_id,
            )

            # Inject the tool result back so the LLM can reason about it
            messages.append(
                ConversationMessage(
                    role="tool",
                    content=result.content,
                    tool_call_id=tc.tool_call_id,
                )
            )

        await pipeline.budget_tracker.increment_tokens(
            session_id, llm_response.token_count
        )

    # ── Max turns reached ─────────────────────────────────────────────────────
    await _save_history(pipeline, session_id, messages)
    return ChatResponse(
        session_id=session_id,
        response=(
            f"Reached the maximum of {max_turns} reasoning turns. "
            "The agent was unable to produce a final answer. "
            "Try rephrasing your request or increasing max_turns."
        ),
        trace_ids=trace_ids,
        tool_calls_made=tool_calls_made,
        total_tokens=total_tokens,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(body: ChatRequest, req: Request) -> ChatResponse:
    """
    Send a message to the agent. The agent reasons, calls tools through the
    harness pipeline, and returns a final answer.

    If `session_id` is omitted a new session is created. Pass the returned
    `session_id` in subsequent requests to continue the conversation.
    """
    pipeline   = _pipeline(req)
    session_id = body.session_id or str(uuid.uuid4())

    try:
        result = await _agentic_loop(
            pipeline, session_id, body.message, body.max_turns
        )
        label = "max_turns" if result.tool_calls_made > 0 and not result.response.startswith("Reached") else "success"
        CHAT_REQUESTS.labels(result=label).inc()
        return result
    except Exception as exc:
        logger.exception("Unhandled error in /chat session=%s", session_id)
        CHAT_REQUESTS.labels(result="error").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health", tags=["System"])
async def health(req: Request):
    """Liveness check. Returns tool list and circuit breaker states."""
    pipeline = _pipeline(req)
    return {
        "status":          "ok",
        "tools":           pipeline.tool_registry.registered_tools,
        "circuit_breakers": pipeline.executor.breaker_states(),
    }


@app.get("/budget/{session_id}", tags=["Session"])
async def budget(session_id: str, req: Request):
    """Current token and call spend for a session."""
    pipeline = _pipeline(req)
    status   = await pipeline.get_budget_status(session_id)
    return status.model_dump()


@app.delete("/session/{session_id}", tags=["Session"])
async def clear_session(session_id: str, req: Request):
    """Clear conversation history for a session (budget counters are unaffected)."""
    pipeline = _pipeline(req)
    await pipeline._redis.delete(f"session:messages:{session_id}")
    return {"session_id": session_id, "cleared": True}


@app.get("/tools", tags=["System"])
async def list_tools(req: Request):
    """List all tools registered from the MCP server."""
    pipeline = _pipeline(req)
    return {"tools": pipeline.tool_registry.registered_tools}


@app.post("/hitl/{trace_id}/decide", tags=["HITL"])
async def hitl_decide(trace_id: str, decision: HITLDecision, req: Request):
    """
    Approve or reject a pending human-in-the-loop tool call.
    The PermissionResolver polls Redis for this key with a 1s interval.
    POST {"decision": "approved"} or {"decision": "rejected"}.
    """
    if decision.decision not in ("approved", "rejected"):
        raise HTTPException(
            status_code=422,
            detail="decision must be 'approved' or 'rejected'",
        )
    pipeline = _pipeline(req)
    key      = f"hitl:approval:{trace_id}"
    exists   = await pipeline._redis.exists(key)
    if not exists:
        raise HTTPException(
            status_code=404,
            detail=f"No pending HITL request found for trace_id={trace_id}",
        )
    await pipeline._redis.set(key, decision.decision)
    logger.info("HITL decision: trace=%s decision=%s by=%s",
                trace_id, decision.decision, decision.decided_by)
    return {
        "trace_id":   trace_id,
        "decision":   decision.decision,
        "decided_by": decision.decided_by,
    }


@app.get("/audit/dlq-depth", tags=["System"])
async def dlq_depth(req: Request):
    """Number of audit records currently waiting in the dead-letter queue."""
    pipeline = _pipeline(req)
    depth    = await pipeline.audit_logger.dlq_depth()
    return {"dlq_depth": depth}


@app.get("/metrics", tags=["System"], include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint — scraped by Prometheus on this same port."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Streaming agentic loop ────────────────────────────────────────────────────

async def _stream_agentic_loop(pipeline, session_id: str, user_message: str, max_turns: int):
    """
    Async generator that runs the agentic loop and yields SSE events
    at each meaningful moment: LLM thinking, tool start, tool end, final answer.
    The frontend listens to these to update the UI in real time.
    """
    def evt(type_: str, **kwargs) -> str:
        return f"data: {json.dumps({'type': type_, **kwargs})}\n\n"

    yield evt("session", session_id=session_id)

    messages = await _load_history(pipeline, session_id)
    messages.append(ConversationMessage(role="user", content=user_message))
    tool_defs       = pipeline.get_tool_definitions()
    total_tokens    = 0
    tool_calls_made = 0
    trace_ids: list[str] = []

    try:
        for turn in range(max_turns):
            yield evt("thinking", turn=turn + 1)

            llm_response = await pipeline.chat(messages, tool_defs)
            total_tokens += llm_response.token_count

            # ── Final answer ──────────────────────────────────────────────────
            if llm_response.is_final:
                messages.append(ConversationMessage(role="assistant", content=llm_response.text))
                await _save_history(pipeline, session_id, messages)
                await pipeline.budget_tracker.increment_tokens(session_id, llm_response.token_count)
                budget = await pipeline.get_budget_status(session_id)
                yield evt("done",
                    response=llm_response.text or "",
                    tool_calls_made=tool_calls_made,
                    total_tokens=total_tokens,
                    trace_ids=trace_ids,
                    call_count=budget.call_count,
                    token_count=budget.token_count,
                    call_limit=budget.call_limit,
                    token_limit=budget.token_limit,
                )
                return

            # ── Tool calls requested ──────────────────────────────────────────
            raw_tcs = [
                {"id": tc.tool_call_id, "type": "function",
                 "function": {"name": tc.tool_name, "arguments": json.dumps(tc.arguments)}}
                for tc in llm_response.tool_calls
            ]
            messages.append(ConversationMessage(role="assistant", content=None, tool_calls=raw_tcs))

            for tc in llm_response.tool_calls:
                yield evt("tool_start",
                    tool=tc.tool_name,
                    args=tc.arguments,
                    call_id=tc.tool_call_id,
                )

                tool_call = ToolCall(
                    tool_call_id=tc.tool_call_id,
                    tool_name=tc.tool_name,
                    arguments=tc.arguments,
                )
                result, trace_id = await pipeline.run_tool_call(session_id, tool_call)
                trace_ids.append(trace_id)
                tool_calls_made += 1

                yield evt("tool_end",
                    tool=tc.tool_name,
                    call_id=tc.tool_call_id,
                    trace_id=trace_id,
                    is_error=result.is_error,
                    preview=result.content[:300] if result.content else "",
                )

                messages.append(ConversationMessage(
                    role="tool",
                    content=result.content,
                    tool_call_id=tc.tool_call_id,
                ))

            await pipeline.budget_tracker.increment_tokens(session_id, llm_response.token_count)

    except Exception as exc:
        logger.exception("Error in streaming agentic loop session=%s", session_id)
        yield evt("error", message=str(exc))
        return

    # Max turns exhausted
    await _save_history(pipeline, session_id, messages)
    yield evt("done",
        response="Maximum reasoning turns reached without a final answer.",
        tool_calls_made=tool_calls_made,
        total_tokens=total_tokens,
        trace_ids=trace_ids,
        call_count=0,
        token_count=total_tokens,
        call_limit=settings.session_budget_calls,
        token_limit=settings.session_budget_tokens,
    )


@app.post("/chat/stream", tags=["Agent"])
async def chat_stream(body: ChatRequest, req: Request):
    """
    Streaming version of /chat using Server-Sent Events.
    The frontend connects here and receives events in real time:
    thinking → tool_start → tool_end → done (or error).
    """
    pipeline   = _pipeline(req)
    session_id = body.session_id or str(uuid.uuid4())

    async def generate():
        async for chunk in _stream_agentic_loop(
            pipeline, session_id, body.message, body.max_turns
        ):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


@app.get("/", include_in_schema=False)
async def frontend():
    """Serve the single-page frontend at the root URL."""
    try:
        with open("src/static/index.html", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>Frontend not found. Run from project root.</h1>", status_code=404)
