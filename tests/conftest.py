"""
Shared fixtures for all test modules.
No real Redis, PostgreSQL, or MCP server required — all I/O is mocked.
"""
from __future__ import annotations

import textwrap
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import yaml

from src.harness.models import PipelineContext, ToolCall


# ── Redis mock ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_redis():
    """AsyncMock that behaves like redis.asyncio.Redis."""
    r = AsyncMock()
    r.get    = AsyncMock(return_value=None)
    r.set    = AsyncMock(return_value=True)
    r.setex  = AsyncMock(return_value=True)
    r.incr   = AsyncMock(return_value=1)
    r.incrby = AsyncMock(return_value=1)
    r.expire = AsyncMock(return_value=True)
    r.exists = AsyncMock(return_value=0)
    r.lpop   = AsyncMock(return_value=None)
    r.rpush  = AsyncMock(return_value=1)
    r.llen   = AsyncMock(return_value=0)
    return r


# ── Policy file ───────────────────────────────────────────────────────────────

@pytest.fixture
def policy_path(tmp_path):
    """
    Write a minimal policy.yaml to a temp directory and return its path.
    Uses approval_timeout_seconds=1 for destructive tools so HITL timeout
    tests complete quickly.
    """
    policy = {
        "tools": {
            "postgres_query": {"risk_level": "read_only",   "allowed": True},
            "http_api_call":  {"risk_level": "network",     "allowed": True},
            "file_read":      {"risk_level": "read_only",   "allowed": True},
            "risky_tool":     {"risk_level": "destructive", "allowed": True},
            "disabled_tool":  {"risk_level": "read_only",   "allowed": False},
        },
        "risk_levels": {
            "read_only": {
                "requires_approval": False,
                "audit": True,
                "cache_results": True,
                "cache_ttl_seconds": 60,
            },
            "network": {
                "requires_approval": False,
                "audit": True,
                "cache_results": False,
            },
            "destructive": {
                "requires_approval": True,
                "audit": True,
                "cache_results": False,
                "approval_timeout_seconds": 1,   # fast for tests
            },
        },
    }
    p = tmp_path / "policy.yaml"
    p.write_text(yaml.dump(policy))
    return str(p)


# ── Context helpers ───────────────────────────────────────────────────────────

@pytest.fixture
def make_ctx():
    """
    Factory that produces a fresh PipelineContext for any tool + args.
    Usage:  ctx = make_ctx("postgres_query", {"query": "SELECT 1"})
    """
    def _factory(
        tool_name: str = "postgres_query",
        args: dict | None = None,
        session_id: str = "test-session-001",
    ) -> PipelineContext:
        return PipelineContext(
            session_id=session_id,
            tool_call=ToolCall(
                tool_call_id=str(uuid4()),
                tool_name=tool_name,
                arguments=args if args is not None else {"query": "SELECT 1"},
            ),
        )
    return _factory
