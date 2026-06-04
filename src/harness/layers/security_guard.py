from __future__ import annotations

import re

from src.harness.models import PipelineContext, ResultStatus, ToolResult

# ── Compiled threat patterns ──────────────────────────────────────────────────

_PROMPT_INJECTION = [
    re.compile(r"ignore\s+(?:previous|all|your)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(?:your|the|all)\s+(?:system\s+)?(?:prompt|instructions?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a|an|the)\s+\w+", re.IGNORECASE),
    re.compile(r"new\s+instructions?:", re.IGNORECASE),
    re.compile(r"<\|(?:im_start|im_end|system|user|assistant)\|>", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]|\[SYS\]|\[/SYS\]"),
    re.compile(r"(?:system|user|assistant)\s*:\s*you\s+(?:must|should|will)\s+", re.IGNORECASE),
]

# Path traversal — covers encoded variants too
_PATH_TRAVERSAL = re.compile(
    r"(?:\.\.[/\\]|%2e%2e[%2f%5c]|%252e%252e|\.%2e|%2e\.)",
    re.IGNORECASE,
)
_ABSOLUTE_PATH = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")

# SSRF — private/loopback IPs and known metadata endpoints
_PRIVATE_IP = re.compile(
    r"(?:^|[/@])"
    r"(?:"
    r"localhost"
    r"|0\.0\.0\.0"
    r"|127\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|169\.254\.\d{1,3}\.\d{1,3}"
    r"|::1"
    r")"
    r"(?::\d+)?(?:/|$)",
    re.IGNORECASE,
)
_METADATA_ENDPOINT = re.compile(
    r"(?:169\.254\.169\.254|metadata\.google\.internal|instance-data)",
    re.IGNORECASE,
)
_BAD_SCHEME = re.compile(r"^(?:file|gopher|dict|ftp|ldap|tftp|jar)://", re.IGNORECASE)

# SQL timing / blind injection beyond what the read-only transaction blocks
_SQL_TIMING = re.compile(r"pg_sleep\s*\(|WAITFOR\s+DELAY|BENCHMARK\s*\(", re.IGNORECASE)


# ── Layer ─────────────────────────────────────────────────────────────────────

class SecurityGuard:
    """
    Layer 1. Inspects every tool call for injection threats before any
    downstream layer or tool execution runs.

    Checks performed per tool:
      - All tools    : prompt injection in any string argument
      - file_read    : path traversal, absolute paths
      - http_api_call: SSRF (private IPs, metadata endpoints, bad schemes)
      - postgres_query: SQL timing attacks (blind SQLi)
    """

    async def check(self, ctx: PipelineContext) -> bool:
        """Return True if safe. On threat: populate ctx and return False."""
        threat = self._scan(ctx.tool_call.tool_name, ctx.tool_call.arguments)
        if threat:
            self._block(ctx, threat)
            return False
        return True

    # ── Private ───────────────────────────────────────────────────────────────

    def _scan(self, tool: str, args: dict) -> str | None:
        for key, value in args.items():
            if not isinstance(value, str):
                continue

            for pattern in _PROMPT_INJECTION:
                if pattern.search(value):
                    return f"Prompt injection detected in argument '{key}'"

            if tool == "file_read" and key == "path":
                if _PATH_TRAVERSAL.search(value):
                    return f"Path traversal attempt in '{key}': {value!r}"
                if _ABSOLUTE_PATH.match(value):
                    return f"Absolute path not allowed in '{key}': {value!r}"

            if tool == "http_api_call" and key == "url":
                if _BAD_SCHEME.match(value):
                    return f"Non-HTTP scheme blocked in '{key}': {value!r}"
                if _PRIVATE_IP.search(value):
                    return f"SSRF: private/loopback address in '{key}': {value!r}"
                if _METADATA_ENDPOINT.search(value):
                    return f"SSRF: cloud metadata endpoint blocked in '{key}': {value!r}"

            if tool == "postgres_query" and key == "query":
                if _SQL_TIMING.search(value):
                    return f"SQL timing attack pattern detected in '{key}'"

        return None

    def _block(self, ctx: PipelineContext, threat: str) -> None:
        ctx.result_status = ResultStatus.SECURITY_BLOCKED
        ctx.rejected_at_layer = "SecurityGuard"
        ctx.error_message = threat
        ctx.result = ToolResult(
            tool_call_id=ctx.tool_call.tool_call_id,
            content=f"Request blocked by security policy: {threat}",
            is_error=True,
        )
