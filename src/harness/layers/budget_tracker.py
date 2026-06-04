from __future__ import annotations

import logging

from src.config import settings as _settings
from src.harness.models import BudgetStatus, PipelineContext, ResultStatus, ToolResult

logger = logging.getLogger(__name__)

_SESSION_TTL = 86_400  # 24 h — keys expire after a day of inactivity


class BudgetTracker:
    """
    Layer 4. Tracks per-session token spend and tool call count in Redis.

    Fail-closed policy: if Redis is unreachable, the call is rejected rather
    than silently allowed — an untracked session could exceed limits without
    any audit trail.

    Two separate responsibilities:
      check()            — gate before tool execution (call count + token count)
      increment_calls()  — called immediately after a tool is approved
      increment_tokens() — called by the pipeline after each LLM response turn
    """

    def __init__(self, redis) -> None:
        self._redis = redis

    # ── Pipeline check ────────────────────────────────────────────────────────

    async def check(self, ctx: PipelineContext) -> bool:
        try:
            calls  = await self._get(ctx.session_id, "calls")
            tokens = await self._get(ctx.session_id, "tokens")
        except Exception as exc:
            logger.error("BudgetTracker Redis error during check: %s", exc)
            return self._reject(
                ctx,
                "Budget check failed — Redis unavailable. Rejecting to prevent untracked spend.",
                ResultStatus.ERROR,
            )

        if calls >= _settings.session_budget_calls:
            return self._reject(
                ctx,
                f"Session call limit reached ({calls}/{_settings.session_budget_calls}). "
                "Start a new session to continue.",
            )

        if tokens >= _settings.session_budget_tokens:
            return self._reject(
                ctx,
                f"Session token limit reached ({tokens}/{_settings.session_budget_tokens}). "
                "Start a new session to continue.",
            )

        return True

    # ── Mutation helpers (called by pipeline, not part of the check chain) ────

    async def increment_calls(self, session_id: str) -> int:
        """Atomically increment tool call counter. Returns new total."""
        key = _key(session_id, "calls")
        count = await self._redis.incr(key)
        await self._redis.expire(key, _SESSION_TTL)
        logger.debug("Session %s calls: %d/%d", session_id, count, _settings.session_budget_calls)
        return count

    async def increment_tokens(self, session_id: str, count: int) -> int:
        """Atomically add token count. Returns new total. Call after each LLM turn."""
        key = _key(session_id, "tokens")
        total = await self._redis.incrby(key, count)
        await self._redis.expire(key, _SESSION_TTL)
        logger.debug("Session %s tokens: %d/%d", session_id, total, _settings.session_budget_tokens)
        return total

    async def get_status(self, session_id: str) -> BudgetStatus:
        """Returns the current spend snapshot for a session."""
        calls  = await self._get(session_id, "calls")
        tokens = await self._get(session_id, "tokens")
        return BudgetStatus(
            session_id=session_id,
            call_count=calls,
            token_count=tokens,
            call_limit=_settings.session_budget_calls,
            token_limit=_settings.session_budget_tokens,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _get(self, session_id: str, metric: str) -> int:
        raw = await self._redis.get(_key(session_id, metric))
        return int(raw) if raw else 0

    def _reject(
        self,
        ctx: PipelineContext,
        reason: str,
        status: ResultStatus = ResultStatus.BUDGET_EXCEEDED,
    ) -> bool:
        ctx.result_status = status
        ctx.rejected_at_layer = "BudgetTracker"
        ctx.error_message = reason
        ctx.result = ToolResult(
            tool_call_id=ctx.tool_call.tool_call_id,
            content=reason,
            is_error=True,
        )
        return False


def _key(session_id: str, metric: str) -> str:
    return f"budget:{session_id}:{metric}"
