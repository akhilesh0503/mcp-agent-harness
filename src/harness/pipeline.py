from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import asyncpg
import redis.asyncio as aioredis
import yaml

from src.config import settings
from src.harness.layers.audit_logger import AuditLogger
from src.harness.layers.budget_tracker import BudgetTracker
from src.harness.layers.executor import Executor
from src.harness.layers.permission_resolver import PermissionResolver
from src.harness.layers.security_guard import SecurityGuard
from src.harness.layers.tool_registry import ToolRegistry
from src.harness.models import (
    BudgetStatus,
    PipelineContext,
    ResultStatus,
    ToolCall,
    ToolResult,
)
from src.llm.base import LLMClient, LLMResponse
from src.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

_POLICY_PATH      = "policy/policy.yaml"
_MCP_FETCH_RETRIES = 3
_MCP_RETRY_DELAY   = 2.0   # seconds; multiplied by attempt number


class Pipeline:
    """
    Wires all six harness layers into a single callable unit.

    Lifecycle:
      startup  → Pipeline.create()     (called from FastAPI lifespan)
      per-call → run_tool_call()
      shutdown → shutdown()

    Layer execution order per tool call:
      SecurityGuard → PermissionResolver → ToolRegistry →
      BudgetTracker → Executor → AuditLogger (always, in finally)
    """

    def __init__(
        self,
        *,
        db_pool: asyncpg.Pool,
        redis,
        mcp_tools: list[dict],
        security_guard: SecurityGuard,
        permission_resolver: PermissionResolver,
        tool_registry: ToolRegistry,
        budget_tracker: BudgetTracker,
        executor: Executor,
        audit_logger: AuditLogger,
        llm: LLMClient,
    ) -> None:
        self._db_pool   = db_pool
        self._redis     = redis
        self._mcp_tools = mcp_tools

        self.security_guard      = security_guard
        self.permission_resolver = permission_resolver
        self.tool_registry       = tool_registry
        self.budget_tracker      = budget_tracker
        self.executor            = executor
        self.audit_logger        = audit_logger
        self.llm                 = llm

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    async def create(cls, llm: LLMClient | None = None) -> "Pipeline":
        """
        Connect to all external services, build every layer, and return a
        fully initialised Pipeline. Call once in the FastAPI lifespan startup.

        Pass an explicit llm to swap providers (default: OllamaClient).
        """
        logger.info("Pipeline startup: connecting to PostgreSQL...")
        db_pool = await asyncpg.create_pool(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            min_size=2,
            max_size=20,
        )

        logger.info("Pipeline startup: connecting to Redis...")
        redis_client = aioredis.from_url(settings.redis_url, decode_responses=False)

        logger.info("Pipeline startup: fetching tools from MCP server...")
        mcp_tools = await cls._fetch_mcp_tools_with_retry()
        logger.info(
            "Pipeline startup: registered %d tools — %s",
            len(mcp_tools),
            [t["name"] for t in mcp_tools],
        )

        policy = yaml.safe_load(Path(_POLICY_PATH).read_text())

        # Build layers
        security_guard      = SecurityGuard()
        permission_resolver = PermissionResolver(_POLICY_PATH, redis=redis_client)
        tool_registry       = ToolRegistry()
        tool_registry.register_many(mcp_tools)
        budget_tracker      = BudgetTracker(redis=redis_client)
        executor            = Executor(redis=redis_client, policy=policy)
        audit_logger        = AuditLogger(db_pool=db_pool, redis=redis_client)

        await audit_logger.start_drainer()

        return cls(
            db_pool=db_pool,
            redis=redis_client,
            mcp_tools=mcp_tools,
            security_guard=security_guard,
            permission_resolver=permission_resolver,
            tool_registry=tool_registry,
            budget_tracker=budget_tracker,
            executor=executor,
            audit_logger=audit_logger,
            llm=llm or OllamaClient(),
        )

    # ── Core execution ────────────────────────────────────────────────────────

    async def run_tool_call(
        self, session_id: str, tool_call: ToolCall
    ) -> tuple[ToolResult, str]:
        """
        Run one LLM-requested tool call through all six layers.
        Returns (ToolResult, trace_id_str).
        AuditLogger always fires in the finally block — no call goes unlogged.
        """
        ctx     = PipelineContext(session_id=session_id, tool_call=tool_call)
        started = time.monotonic()

        try:
            # Layer 1 — SecurityGuard (injection / traversal / SSRF)
            if not await self.security_guard.check(ctx):
                return self._safe_result(ctx), ctx.trace_id_str

            # Layer 2 — PermissionResolver (policy + HITL)
            if not await self.permission_resolver.check(ctx):
                return self._safe_result(ctx), ctx.trace_id_str

            # Layer 3 — ToolRegistry (exists + schema valid)
            if not await self.tool_registry.check(ctx):
                return self._safe_result(ctx), ctx.trace_id_str

            # Layer 4 — BudgetTracker (spend gate)
            if not await self.budget_tracker.check(ctx):
                return self._safe_result(ctx), ctx.trace_id_str

            # Budget approved — increment call counter before execution
            await self.budget_tracker.increment_calls(session_id)

            # Layer 5 — Executor (cache → circuit breaker → MCP call)
            await self.executor.check(ctx)
            return self._safe_result(ctx), ctx.trace_id_str

        except Exception as exc:
            logger.exception(
                "Pipeline: unhandled error session=%s tool=%s",
                session_id, tool_call.tool_name,
            )
            if ctx.result is None:
                ctx.result_status     = ResultStatus.ERROR
                ctx.error_message     = str(exc)
                ctx.rejected_at_layer = "Pipeline"
                ctx.result = ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    content=f"Internal pipeline error: {exc}",
                    is_error=True,
                )
            return self._safe_result(ctx), ctx.trace_id_str

        finally:
            ctx.latency_ms = int((time.monotonic() - started) * 1000)
            # Layer 6 — AuditLogger (always, even on exception)
            await self.audit_logger.record(ctx)

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def get_tool_definitions(self) -> list[dict]:
        """Provider-formatted definitions to pass directly into llm.chat()."""
        return self.llm.tool_definitions(self._mcp_tools)

    async def chat(self, messages, tools: list[dict] | None = None) -> LLMResponse:
        """Thin wrapper so the agentic loop doesn't import LLMClient directly."""
        return await self.llm.chat(messages, tools or self.get_tool_definitions())

    async def get_budget_status(self, session_id: str) -> BudgetStatus:
        return await self.budget_tracker.get_status(session_id)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        logger.info("Pipeline: shutting down...")
        await self.audit_logger.stop_drainer()
        await self._db_pool.close()
        await self._redis.aclose()
        logger.info("Pipeline: shutdown complete")

    # ── MCP bootstrap ─────────────────────────────────────────────────────────

    @staticmethod
    async def _fetch_mcp_tools_with_retry() -> list[dict]:
        last_exc: Exception | None = None
        for attempt in range(1, _MCP_FETCH_RETRIES + 1):
            try:
                return await Pipeline._fetch_mcp_tools()
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "MCP tool fetch attempt %d/%d failed: %s",
                    attempt, _MCP_FETCH_RETRIES, exc,
                )
                if attempt < _MCP_FETCH_RETRIES:
                    await asyncio.sleep(_MCP_RETRY_DELAY * attempt)
        raise RuntimeError(
            f"Cannot reach MCP server after {_MCP_FETCH_RETRIES} attempts: {last_exc}"
        )

    @staticmethod
    async def _fetch_mcp_tools() -> list[dict]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        url = f"{settings.mcp_server_url}/mcp"
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()

        tools = []
        for tool in result.tools:
            schema = tool.inputSchema
            # MCP SDK may return a Pydantic model or a plain dict
            if hasattr(schema, "model_dump"):
                schema = schema.model_dump()
            elif not isinstance(schema, dict):
                schema = dict(schema)
            tools.append({
                "name":        tool.name,
                "description": tool.description or "",
                "inputSchema": schema,
            })
        return tools

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_result(ctx: PipelineContext) -> ToolResult:
        """Return ctx.result, creating a fallback if a layer forgot to set it."""
        if ctx.result is not None:
            return ctx.result
        return ToolResult(
            tool_call_id=ctx.tool_call.tool_call_id,
            content="No result produced (pipeline bug)",
            is_error=True,
        )
