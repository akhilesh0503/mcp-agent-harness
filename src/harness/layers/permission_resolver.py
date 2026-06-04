from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import httpx
import yaml

from src.config import settings
from src.harness.models import PipelineContext, RiskLevel, ResultStatus, ToolResult

logger = logging.getLogger(__name__)


class PermissionResolver:
    """
    Layer 2. Looks up the tool in policy.yaml to get its risk level and
    decides whether to allow, reject, or pause for human approval (HITL).

    HITL flow for destructive-risk tools:
      1. Write approval request to Redis key hitl:approval:{trace_id} = "pending"
      2. Optionally POST to HITL_WEBHOOK_URL with request details
      3. Poll Redis every second until a decision arrives or timeout expires
      4. Timeout → reject (fail closed)
    """

    def __init__(self, policy_path: str = "policy/policy.yaml", redis=None):
        self._redis = redis   # injected by pipeline; may be None in tests
        self._policy = self._load_policy(policy_path)

    # ── Public ────────────────────────────────────────────────────────────────

    async def check(self, ctx: PipelineContext) -> bool:
        tool_name = ctx.tool_call.tool_name
        tool_cfg = self._policy.get("tools", {}).get(tool_name)

        if not tool_cfg:
            return self._reject(ctx, f"Tool '{tool_name}' not found in policy")

        if not tool_cfg.get("allowed", False):
            return self._reject(ctx, f"Tool '{tool_name}' is disabled by policy")

        risk_str = tool_cfg.get("risk_level", "read_only")
        try:
            ctx.risk_level = RiskLevel(risk_str)
        except ValueError:
            return self._reject(ctx, f"Unknown risk level '{risk_str}' in policy")

        risk_cfg = self._policy.get("risk_levels", {}).get(risk_str, {})
        if risk_cfg.get("requires_approval", False):
            approved = await self._hitl(ctx, risk_cfg)
            if not approved:
                ctx.result_status = ResultStatus.HITL_TIMEOUT
                ctx.rejected_at_layer = "PermissionResolver"
                ctx.error_message = "Human approval was not granted within the timeout"
                ctx.result = ToolResult(
                    tool_call_id=ctx.tool_call.tool_call_id,
                    content="This operation requires human approval. Request timed out or was rejected.",
                    is_error=True,
                )
                return False

        ctx.permission_granted = True
        return True

    # ── HITL ──────────────────────────────────────────────────────────────────

    async def _hitl(self, ctx: PipelineContext, risk_cfg: dict) -> bool:
        timeout = risk_cfg.get(
            "approval_timeout_seconds",
            settings.hitl_approval_timeout_seconds,
        )
        redis_key = f"hitl:approval:{ctx.trace_id_str}"

        if self._redis is None:
            logger.warning(
                "HITL required for %s but no Redis client — auto-rejecting",
                ctx.tool_call.tool_name,
            )
            return False

        # Store pending request with a slightly longer TTL than the poll window
        payload = json.dumps({
            "trace_id": ctx.trace_id_str,
            "session_id": ctx.session_id,
            "tool_name": ctx.tool_call.tool_name,
            "tool_input": ctx.tool_call.arguments,
            "risk_level": ctx.risk_level.value if ctx.risk_level else "unknown",
        })
        await self._redis.setex(redis_key, timeout + 10, "pending")

        await self._notify_webhook(ctx, payload)
        logger.info(
            "HITL approval requested for trace=%s tool=%s timeout=%ss",
            ctx.trace_id_str, ctx.tool_call.tool_name, timeout,
        )

        # Poll for a decision
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            raw = await self._redis.get(redis_key)
            if raw:
                decision = raw.decode() if isinstance(raw, bytes) else raw
                if decision != "pending":
                    logger.info(
                        "HITL decision for trace=%s: %s",
                        ctx.trace_id_str, decision,
                    )
                    return decision == "approved"
            await asyncio.sleep(1.0)

        logger.warning("HITL timed out for trace=%s", ctx.trace_id_str)
        return False

    async def _notify_webhook(self, ctx: PipelineContext, payload: str) -> None:
        if not settings.hitl_webhook_url:
            return
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    settings.hitl_webhook_url,
                    content=payload,
                    headers={"Content-Type": "application/json"},
                )
        except Exception as exc:
            logger.warning("HITL webhook failed: %s", exc)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _reject(self, ctx: PipelineContext, reason: str) -> bool:
        ctx.result_status = ResultStatus.REJECTED
        ctx.rejected_at_layer = "PermissionResolver"
        ctx.error_message = reason
        ctx.result = ToolResult(
            tool_call_id=ctx.tool_call.tool_call_id,
            content=reason,
            is_error=True,
        )
        return False

    @staticmethod
    def _load_policy(path: str) -> dict:
        policy_file = Path(path)
        if not policy_file.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")
        with policy_file.open() as fh:
            return yaml.safe_load(fh)
