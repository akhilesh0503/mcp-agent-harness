from __future__ import annotations

import asyncio
import json
import logging
import uuid

from src.harness.metrics import AUDIT_DLQ_DEPTH, AUDIT_WRITES
from src.harness.models import PipelineContext

logger = logging.getLogger(__name__)

_DLQ_KEY        = "audit:dlq"
_DRAIN_INTERVAL = 30   # seconds between DLQ drain sweeps
_MAX_DLQ_RETRY  = 3    # abandon record after this many consecutive failures


class AuditLogger:
    """
    Layer 6. Always runs — even when earlier layers rejected or blocked the call.
    The pipeline calls record() in a finally block, so every tool attempt is logged.

    Durability guarantee:
      - Primary path: asyncpg INSERT into audit_log
      - Fallback: JSON payload pushed to Redis list 'audit:dlq'
      - Background drainer replays DLQ records every 30s
      - If DLQ push itself fails, the loss is logged at CRITICAL level —
        at least the failure is visible in logs even if the record is gone
    """

    def __init__(self, db_pool, redis) -> None:
        self._pool  = db_pool
        self._redis = redis
        self._drainer_task: asyncio.Task | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def record(self, ctx: PipelineContext) -> None:
        """Write one audit row for this tool call. Falls back to DLQ on DB failure."""
        row = _build_row(ctx)
        try:
            await self._write(row)
            AUDIT_WRITES.labels(result="success").inc()
            logger.debug(
                "AuditLogger: wrote trace=%s tool=%s status=%s",
                row["trace_id"], row["tool_name"], row["result_status"],
            )
        except Exception as exc:
            logger.error(
                "AuditLogger: DB write failed (trace=%s) — pushing to DLQ: %s",
                row["trace_id"], exc,
            )
            AUDIT_WRITES.labels(result="dlq").inc()
            await self._dlq_push(row)

    async def start_drainer(self) -> None:
        """Launch the background DLQ drain loop. Call once at startup."""
        self._drainer_task = asyncio.create_task(self._drain_loop(), name="audit-dlq-drainer")
        logger.info("AuditLogger: DLQ drainer started (interval=%ds)", _DRAIN_INTERVAL)

    async def stop_drainer(self) -> None:
        if self._drainer_task and not self._drainer_task.done():
            self._drainer_task.cancel()
            try:
                await self._drainer_task
            except asyncio.CancelledError:
                pass

    async def dlq_depth(self) -> int:
        """Return the current number of records waiting in the DLQ."""
        return await self._redis.llen(_DLQ_KEY)

    # ── DB write ──────────────────────────────────────────────────────────────

    async def _write(self, row: dict) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO audit_log (
                    trace_id, session_id, tool_name, input_hash,
                    risk_level, layer, permission_granted,
                    result_status, latency_ms, error_message
                ) VALUES (
                    $1::uuid, $2, $3, $4,
                    $5, $6, $7,
                    $8, $9, $10
                )
                """,
                row["trace_id"],
                row["session_id"],
                row["tool_name"],
                row["input_hash"],
                row["risk_level"],
                row["layer"],
                row["permission_granted"],
                row["result_status"],
                row["latency_ms"],
                row["error_message"],
            )

    # ── DLQ ───────────────────────────────────────────────────────────────────

    async def _drain_loop(self) -> None:
        while True:
            await asyncio.sleep(_DRAIN_INTERVAL)
            try:
                drained = await self._drain_once()
                if drained:
                    logger.info("AuditLogger DLQ: replayed %d record(s) to PostgreSQL", drained)
                depth = await self.dlq_depth()
                AUDIT_DLQ_DEPTH.set(depth)
            except Exception as exc:
                logger.error("AuditLogger DLQ drain sweep failed: %s", exc)

    async def _drain_once(self) -> int:
        """
        Pop records from the DLQ and replay them into PostgreSQL.
        Stops on first replay failure to avoid tight loops against a down DB.
        Returns the count of successfully replayed records.
        """
        drained = 0
        while True:
            raw = await self._redis.lpop(_DLQ_KEY)
            if raw is None:
                break
            try:
                row = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                # Increment retry counter
                row["_retries"] = row.get("_retries", 0) + 1
                await self._write(row)
                drained += 1
            except Exception as exc:
                retries = row.get("_retries", 1)
                if retries >= _MAX_DLQ_RETRY:
                    logger.critical(
                        "AuditLogger: DLQ record abandoned after %d retries "
                        "(trace=%s): %s",
                        retries, row.get("trace_id", "?"), exc,
                    )
                else:
                    # Push back to front of queue for next drain cycle
                    await self._redis.lpush(_DLQ_KEY, json.dumps(row))
                    logger.warning(
                        "AuditLogger: DLQ replay failed (retry %d/%d, trace=%s): %s",
                        retries, _MAX_DLQ_RETRY, row.get("trace_id", "?"), exc,
                    )
                break   # stop draining — DB is likely still down
        return drained

    async def _dlq_push(self, row: dict) -> None:
        try:
            await self._redis.rpush(_DLQ_KEY, json.dumps(row))
            logger.warning("AuditLogger: record queued in DLQ (trace=%s)", row["trace_id"])
        except Exception as exc:
            # Last resort — at least the structured data is in the log stream
            logger.critical(
                "AuditLogger: DLQ push FAILED — record may be lost. "
                "trace=%s tool=%s status=%s error=%s",
                row.get("trace_id"), row.get("tool_name"),
                row.get("result_status"), exc,
            )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_row(ctx: PipelineContext) -> dict:
    return {
        "trace_id":          str(ctx.trace_id),
        "session_id":        ctx.session_id,
        "tool_name":         ctx.tool_call.tool_name,
        "input_hash":        ctx.input_hash,
        "risk_level":        ctx.risk_level.value if ctx.risk_level else None,
        "layer":             ctx.rejected_at_layer,
        "permission_granted": ctx.permission_granted,
        "result_status":     ctx.result_status.value,
        "latency_ms":        ctx.latency_ms,
        "error_message":     ctx.error_message,
    }
