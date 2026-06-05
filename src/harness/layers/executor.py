from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.config import settings
from src.harness.metrics import (
    CIRCUIT_BREAKER_OPEN,
    EXECUTOR_CALLS,
    EXECUTOR_LATENCY,
)
from src.harness.models import PipelineContext, ResultStatus, ToolResult

logger = logging.getLogger(__name__)


# ── Circuit breaker state ─────────────────────────────────────────────────────

@dataclass
class _Breaker:
    """
    Per-tool in-memory circuit breaker.
    Resets on process restart — acceptable for single-process FastAPI.
    Use Redis-backed state for multi-instance deployments.
    """
    state: str = "closed"    # closed | open | half_open
    failures: int = 0
    opened_at: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ── Executor layer ────────────────────────────────────────────────────────────

class Executor:
    """
    Layer 5. The only layer that performs I/O against an external system.
    Runs after all gates (Security, Permission, Registry, Budget) have passed.

    Sub-steps in order:
      1. Cache check   — return immediately on Redis hit (no MCP call)
      2. Breaker check — reject immediately if tool circuit is OPEN
      3. MCP call      — call the tool on the MCP server (60s timeout)
      4. Breaker update — record success or failure; may trip/reset the breaker
      5. Cache write   — persist result for cacheable tools (TTL from policy)
    """

    def __init__(self, redis, policy: dict) -> None:
        self._redis = redis
        self._policy = policy
        self._breakers: dict[str, _Breaker] = {}

    # ── Public ────────────────────────────────────────────────────────────────

    async def check(self, ctx: PipelineContext) -> bool:
        tool = ctx.tool_call.tool_name

        # 1. Cache
        cached = await self._cache_get(ctx)
        if cached is not None:
            ctx.cache_hit = True
            ctx.result = cached
            ctx.result_status = ResultStatus.SUCCESS
            EXECUTOR_CALLS.labels(tool=tool, result="cache_hit").inc()
            logger.debug("Cache HIT for tool '%s'", tool)
            return True

        # 2. Circuit breaker
        breaker = self._get_breaker(tool)
        async with breaker.lock:
            allowed, rejection_msg = self._breaker_check(breaker, tool)
        if not allowed:
            EXECUTOR_CALLS.labels(tool=tool, result="circuit_open").inc()
            return self._fail(ctx, ResultStatus.CIRCUIT_OPEN, rejection_msg)

        # 3. MCP call (timed)
        try:
            with EXECUTOR_LATENCY.labels(tool=tool).time():
                content = await asyncio.wait_for(
                    self._mcp_call(tool, ctx.tool_call.arguments),
                    timeout=60.0,
                )
        except asyncio.TimeoutError:
            async with breaker.lock:
                self._breaker_record_failure(breaker, tool)
            EXECUTOR_CALLS.labels(tool=tool, result="timeout").inc()
            return self._fail(ctx, ResultStatus.TIMEOUT, f"Tool '{tool}' timed out after 60s")
        except Exception as exc:
            logger.error("Executor MCP call failed — tool='%s' error='%s'", tool, exc)
            async with breaker.lock:
                self._breaker_record_failure(breaker, tool)
            EXECUTOR_CALLS.labels(tool=tool, result="error").inc()
            return self._fail(ctx, ResultStatus.ERROR, str(exc))

        # 4. Record success + update circuit breaker gauge
        async with breaker.lock:
            self._breaker_record_success(breaker, tool)

        # 5. Cache write
        await self._cache_set(ctx, content)

        EXECUTOR_CALLS.labels(tool=tool, result="success").inc()
        ctx.result = ToolResult(
            tool_call_id=ctx.tool_call.tool_call_id,
            content=content,
            is_error=False,
        )
        ctx.result_status = ResultStatus.SUCCESS
        return True

    def breaker_states(self) -> dict[str, str]:
        """Expose per-tool circuit breaker states for metrics/observability."""
        return {tool: b.state for tool, b in self._breakers.items()}

    # ── Circuit breaker ───────────────────────────────────────────────────────

    def _get_breaker(self, tool: str) -> _Breaker:
        if tool not in self._breakers:
            self._breakers[tool] = _Breaker()
        return self._breakers[tool]

    def _breaker_check(self, breaker: _Breaker, tool: str) -> tuple[bool, str]:
        """
        Evaluate current breaker state. Caller must hold breaker.lock.
        Returns (allowed, rejection_message).
        """
        threshold = settings.circuit_breaker_failure_threshold
        recovery  = settings.circuit_breaker_recovery_seconds

        if breaker.state == "closed":
            return True, ""

        if breaker.state == "open":
            elapsed = time.monotonic() - breaker.opened_at
            if elapsed >= recovery:
                logger.info("Circuit breaker '%s': OPEN → HALF_OPEN (%.0fs elapsed)", tool, elapsed)
                breaker.state = "half_open"
                return True, ""
            remaining = recovery - elapsed
            return False, (
                f"Tool '{tool}' circuit breaker is OPEN — "
                f"too many recent failures. Retries in {remaining:.0f}s."
            )

        # half_open: let exactly one probe request through
        return True, ""

    def _breaker_record_success(self, breaker: _Breaker, tool: str) -> None:
        if breaker.state == "half_open":
            logger.info("Circuit breaker '%s': HALF_OPEN → CLOSED (probe succeeded)", tool)
        breaker.failures = 0
        breaker.state = "closed"
        CIRCUIT_BREAKER_OPEN.labels(tool=tool).set(0)

    def _breaker_record_failure(self, breaker: _Breaker, tool: str) -> None:
        threshold = settings.circuit_breaker_failure_threshold

        if breaker.state == "half_open":
            logger.warning("Circuit breaker '%s': HALF_OPEN → OPEN (probe failed)", tool)
            breaker.state = "open"
            breaker.opened_at = time.monotonic()
            CIRCUIT_BREAKER_OPEN.labels(tool=tool).set(1)
            return

        breaker.failures += 1
        if breaker.failures >= threshold:
            logger.warning(
                "Circuit breaker '%s': CLOSED → OPEN (%d/%d failures)",
                tool, breaker.failures, threshold,
            )
            breaker.state = "open"
            breaker.opened_at = time.monotonic()
            CIRCUIT_BREAKER_OPEN.labels(tool=tool).set(1)

    # ── MCP call ──────────────────────────────────────────────────────────────

    async def _mcp_call(self, tool_name: str, arguments: dict) -> str:
        """
        Opens a fresh MCP session per call via streamable-http transport.
        For multi-instance prod, replace with a shared connection pool.
        """
        url = f"{settings.mcp_server_url}/mcp"
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)

        texts = [
            block.text
            for block in result.content
            if hasattr(block, "text") and block.text
        ]
        return "\n".join(texts)

    # ── Cache ─────────────────────────────────────────────────────────────────

    async def _cache_get(self, ctx: PipelineContext) -> ToolResult | None:
        if not self._is_cacheable(ctx):
            return None
        raw = await self._redis.get(f"cache:{ctx.input_hash}")
        if raw is None:
            return None
        return ToolResult(
            tool_call_id=ctx.tool_call.tool_call_id,
            content=raw.decode() if isinstance(raw, bytes) else raw,
            is_error=False,
        )

    async def _cache_set(self, ctx: PipelineContext, content: str) -> None:
        if not self._is_cacheable(ctx):
            return
        ttl = self._cache_ttl(ctx)
        await self._redis.setex(f"cache:{ctx.input_hash}", ttl, content)
        logger.debug("Cache SET tool='%s' ttl=%ds hash=%s", ctx.tool_call.tool_name, ttl, ctx.input_hash[:8])

    def _is_cacheable(self, ctx: PipelineContext) -> bool:
        if ctx.risk_level is None:
            return False
        cfg = self._policy.get("risk_levels", {}).get(ctx.risk_level.value, {})
        return bool(cfg.get("cache_results", False))

    def _cache_ttl(self, ctx: PipelineContext) -> int:
        cfg = self._policy.get("risk_levels", {}).get(ctx.risk_level.value, {})
        return int(cfg.get("cache_ttl_seconds", 60))

    # ── Failure helper ────────────────────────────────────────────────────────

    def _fail(self, ctx: PipelineContext, status: ResultStatus, message: str) -> bool:
        ctx.result_status = status
        ctx.rejected_at_layer = "Executor"
        ctx.error_message = message
        ctx.result = ToolResult(
            tool_call_id=ctx.tool_call.tool_call_id,
            content=message,
            is_error=True,
        )
        return False
