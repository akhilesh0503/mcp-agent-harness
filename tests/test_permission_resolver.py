"""
Tests for PermissionResolver (Layer 2).

Covers: policy lookup, risk level assignment, HITL approval flow,
HITL timeout, fail-closed when Redis is absent, disabled/unknown tools.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.harness.layers.permission_resolver import PermissionResolver
from src.harness.models import ResultStatus, RiskLevel


# ── Happy path ────────────────────────────────────────────────────────────────

async def test_allowed_read_only_tool(policy_path, make_ctx):
    """A read_only tool in policy should be approved with no HITL."""
    resolver = PermissionResolver(policy_path=policy_path)
    ctx = make_ctx("postgres_query")

    result = await resolver.check(ctx)

    assert result is True
    assert ctx.permission_granted is True
    assert ctx.risk_level == RiskLevel.READ_ONLY
    assert ctx.result is None          # no error result set


async def test_network_tool_approved(policy_path, make_ctx):
    """A network-risk tool requires no HITL and should be approved."""
    resolver = PermissionResolver(policy_path=policy_path)
    ctx = make_ctx("http_api_call", {"url": "https://api.example.com"})

    result = await resolver.check(ctx)

    assert result is True
    assert ctx.risk_level == RiskLevel.NETWORK


# ── Rejection cases ───────────────────────────────────────────────────────────

async def test_unknown_tool_rejected(policy_path, make_ctx):
    """A tool not in the policy must be rejected."""
    resolver = PermissionResolver(policy_path=policy_path)
    ctx = make_ctx("unknown_tool", {})

    result = await resolver.check(ctx)

    assert result is False
    assert ctx.result_status == ResultStatus.REJECTED
    assert ctx.rejected_at_layer == "PermissionResolver"
    assert ctx.result is not None
    assert ctx.result.is_error is True


async def test_disabled_tool_rejected(policy_path, make_ctx):
    """A tool present in policy but marked allowed=False must be rejected."""
    resolver = PermissionResolver(policy_path=policy_path)
    ctx = make_ctx("disabled_tool", {})

    result = await resolver.check(ctx)

    assert result is False
    assert ctx.result_status == ResultStatus.REJECTED
    assert "disabled" in ctx.error_message.lower()


# ── HITL — no Redis (fail closed) ─────────────────────────────────────────────

async def test_hitl_no_redis_fails_closed(policy_path, make_ctx):
    """
    Destructive tool with redis=None must fail closed immediately.
    No approval is possible without Redis.
    """
    resolver = PermissionResolver(policy_path=policy_path, redis=None)
    ctx = make_ctx("risky_tool", {"target": "production-db"})

    result = await resolver.check(ctx)

    assert result is False
    assert ctx.result_status == ResultStatus.HITL_TIMEOUT


# ── HITL — approved ───────────────────────────────────────────────────────────

async def test_hitl_approved_via_redis(policy_path, make_ctx, mock_redis):
    """
    Destructive tool where Redis returns 'approved' immediately.
    PermissionResolver should allow the call.
    """
    mock_redis.get = AsyncMock(return_value=b"approved")
    resolver = PermissionResolver(policy_path=policy_path, redis=mock_redis)
    ctx = make_ctx("risky_tool", {"target": "staging-db"})

    result = await resolver.check(ctx)

    assert result is True
    assert ctx.permission_granted is True


# ── HITL — rejected ───────────────────────────────────────────────────────────

async def test_hitl_rejected_via_redis(policy_path, make_ctx, mock_redis):
    """Redis returns 'rejected' — the call must be blocked."""
    mock_redis.get = AsyncMock(return_value=b"rejected")
    resolver = PermissionResolver(policy_path=policy_path, redis=mock_redis)
    ctx = make_ctx("risky_tool", {"target": "prod"})

    result = await resolver.check(ctx)

    assert result is False
    assert ctx.result_status == ResultStatus.HITL_TIMEOUT   # treated same as timeout


# ── HITL — timeout ────────────────────────────────────────────────────────────

async def test_hitl_timeout_when_redis_always_pending(policy_path, make_ctx, mock_redis):
    """
    Redis keeps returning 'pending'. After approval_timeout_seconds=1
    the resolver should give up and fail closed.
    """
    mock_redis.get = AsyncMock(return_value=b"pending")
    resolver = PermissionResolver(policy_path=policy_path, redis=mock_redis)
    ctx = make_ctx("risky_tool", {"target": "prod"})

    result = await resolver.check(ctx)

    assert result is False
    assert ctx.result_status == ResultStatus.HITL_TIMEOUT
    assert ctx.rejected_at_layer == "PermissionResolver"


# ── Webhook fires but doesn't block ──────────────────────────────────────────

async def test_webhook_failure_does_not_block_approval(policy_path, make_ctx, mock_redis):
    """
    Even if the webhook call fails, the HITL flow continues and
    can still be approved via Redis.
    """
    mock_redis.get = AsyncMock(return_value=b"approved")

    with patch("src.harness.layers.permission_resolver.settings") as mock_settings:
        mock_settings.hitl_webhook_url = "http://bad-url-that-fails/"
        mock_settings.hitl_approval_timeout_seconds = 5
        resolver = PermissionResolver(policy_path=policy_path, redis=mock_redis)
        ctx = make_ctx("risky_tool", {})
        result = await resolver.check(ctx)

    assert result is True


# ── Policy file not found ─────────────────────────────────────────────────────

def test_missing_policy_file_raises():
    """Constructing PermissionResolver with a non-existent policy path must raise."""
    with pytest.raises(FileNotFoundError):
        PermissionResolver(policy_path="/does/not/exist/policy.yaml")
