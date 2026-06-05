"""
Tests for BudgetTracker (Layer 4).

Covers: under-limit approval, call/token limit enforcement, Redis error
fail-closed, atomic increment helpers, BudgetStatus correctness.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.harness.layers.budget_tracker import BudgetTracker, _key
from src.harness.models import BudgetStatus, ResultStatus


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_settings(calls_limit: int = 100, tokens_limit: int = 50_000):
    """Return a patch context manager for budget limit settings."""
    return patch(
        "src.harness.layers.budget_tracker._settings",
        session_budget_calls=calls_limit,
        session_budget_tokens=tokens_limit,
    )


# ── Happy path ────────────────────────────────────────────────────────────────

async def test_under_limit_allowed(make_ctx, mock_redis):
    """Session with zero spend must pass the gate."""
    mock_redis.get = AsyncMock(return_value=None)   # 0 calls, 0 tokens

    with _mock_settings():
        tracker = BudgetTracker(redis=mock_redis)
        ctx = make_ctx()
        result = await tracker.check(ctx)

    assert result is True
    assert ctx.result is None       # no error injected


async def test_partial_spend_allowed(make_ctx, mock_redis):
    """Session that has used half its budget must still be allowed."""
    mock_redis.get = AsyncMock(side_effect=[b"40", b"20000"])   # calls, tokens

    with _mock_settings(calls_limit=100, tokens_limit=50_000):
        tracker = BudgetTracker(redis=mock_redis)
        ctx = make_ctx()
        result = await tracker.check(ctx)

    assert result is True


# ── Call limit enforcement ────────────────────────────────────────────────────

async def test_call_limit_exactly_reached(make_ctx, mock_redis):
    """When call count equals the limit the gate must close."""
    mock_redis.get = AsyncMock(side_effect=[b"10", b"0"])   # calls=10, tokens=0

    with _mock_settings(calls_limit=10, tokens_limit=50_000):
        tracker = BudgetTracker(redis=mock_redis)
        ctx = make_ctx()
        result = await tracker.check(ctx)

    assert result is False
    assert ctx.result_status == ResultStatus.BUDGET_EXCEEDED
    assert ctx.rejected_at_layer == "BudgetTracker"
    assert "call limit" in ctx.error_message.lower()
    assert ctx.result.is_error is True


async def test_call_limit_exceeded(make_ctx, mock_redis):
    """Calls over the limit are also rejected."""
    mock_redis.get = AsyncMock(side_effect=[b"999", b"0"])

    with _mock_settings(calls_limit=100, tokens_limit=50_000):
        tracker = BudgetTracker(redis=mock_redis)
        ctx = make_ctx()
        result = await tracker.check(ctx)

    assert result is False
    assert ctx.result_status == ResultStatus.BUDGET_EXCEEDED


# ── Token limit enforcement ───────────────────────────────────────────────────

async def test_token_limit_exactly_reached(make_ctx, mock_redis):
    """When token count equals the limit the gate must close."""
    mock_redis.get = AsyncMock(side_effect=[b"0", b"50000"])  # calls=0, tokens=50000

    with _mock_settings(calls_limit=100, tokens_limit=50_000):
        tracker = BudgetTracker(redis=mock_redis)
        ctx = make_ctx()
        result = await tracker.check(ctx)

    assert result is False
    assert ctx.result_status == ResultStatus.BUDGET_EXCEEDED
    assert "token limit" in ctx.error_message.lower()


async def test_token_limit_exceeded(make_ctx, mock_redis):
    """Tokens over the limit are also rejected."""
    mock_redis.get = AsyncMock(side_effect=[b"1", b"99999"])

    with _mock_settings(calls_limit=100, tokens_limit=50_000):
        tracker = BudgetTracker(redis=mock_redis)
        ctx = make_ctx()
        result = await tracker.check(ctx)

    assert result is False
    assert ctx.result_status == ResultStatus.BUDGET_EXCEEDED


# ── Fail-closed on Redis error ────────────────────────────────────────────────

async def test_redis_unavailable_fails_closed(make_ctx, mock_redis):
    """
    If Redis raises during a budget check, the call must be rejected
    rather than silently allowed (fail-closed policy).
    """
    mock_redis.get = AsyncMock(side_effect=ConnectionError("Redis unreachable"))

    with _mock_settings():
        tracker = BudgetTracker(redis=mock_redis)
        ctx = make_ctx()
        result = await tracker.check(ctx)

    assert result is False
    assert ctx.result_status == ResultStatus.ERROR
    assert ctx.rejected_at_layer == "BudgetTracker"


# ── Increment helpers ─────────────────────────────────────────────────────────

async def test_increment_calls_uses_correct_key(mock_redis):
    """increment_calls must INCR the right Redis key."""
    mock_redis.incr = AsyncMock(return_value=7)

    with _mock_settings():
        tracker = BudgetTracker(redis=mock_redis)
        count = await tracker.increment_calls("sess-abc")

    assert count == 7
    mock_redis.incr.assert_called_once_with("budget:sess-abc:calls")
    mock_redis.expire.assert_called_once()


async def test_increment_tokens_uses_correct_key(mock_redis):
    """increment_tokens must INCRBY the right Redis key by the exact amount."""
    mock_redis.incrby = AsyncMock(return_value=3_500)

    with _mock_settings():
        tracker = BudgetTracker(redis=mock_redis)
        total = await tracker.increment_tokens("sess-abc", 500)

    assert total == 3_500
    mock_redis.incrby.assert_called_once_with("budget:sess-abc:tokens", 500)
    mock_redis.expire.assert_called_once()


# ── BudgetStatus ──────────────────────────────────────────────────────────────

async def test_get_status_returns_correct_snapshot(mock_redis):
    """get_status should return live call + token counts with limit metadata."""
    mock_redis.get = AsyncMock(side_effect=[b"15", b"12000"])

    with _mock_settings(calls_limit=100, tokens_limit=50_000):
        tracker = BudgetTracker(redis=mock_redis)
        status = await tracker.get_status("sess-xyz")

    assert isinstance(status, BudgetStatus)
    assert status.call_count == 15
    assert status.token_count == 12_000
    assert status.call_limit == 100
    assert status.token_limit == 50_000
    assert status.calls_remaining == 85
    assert status.tokens_remaining == 38_000
    assert status.is_exhausted is False


async def test_get_status_is_exhausted_when_calls_maxed(mock_redis):
    """is_exhausted must be True when call count meets the limit."""
    mock_redis.get = AsyncMock(side_effect=[b"100", b"0"])

    with _mock_settings(calls_limit=100, tokens_limit=50_000):
        tracker = BudgetTracker(redis=mock_redis)
        status = await tracker.get_status("sess-xyz")

    assert status.is_exhausted is True


# ── Redis key format ──────────────────────────────────────────────────────────

def test_key_format():
    """The private _key() helper must produce the expected Redis key string."""
    assert _key("my-session", "calls")  == "budget:my-session:calls"
    assert _key("my-session", "tokens") == "budget:my-session:tokens"
