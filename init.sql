-- Audit log: one row per tool call attempt, all layers
CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGSERIAL PRIMARY KEY,
    trace_id    UUID          NOT NULL,
    session_id  VARCHAR(255)  NOT NULL,
    timestamp   TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    tool_name   VARCHAR(255)  NOT NULL,
    input_hash  VARCHAR(64)   NOT NULL,
    risk_level  VARCHAR(50),
    layer       VARCHAR(50),
    permission_granted BOOLEAN NOT NULL,
    result_status      VARCHAR(50) NOT NULL, -- success | error | rejected | timeout | circuit_open | budget_exceeded
    latency_ms  INTEGER,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_trace   ON audit_log(trace_id);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_time    ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_tool    ON audit_log(tool_name);

-- Dead-letter queue: audit records that failed to write to audit_log
CREATE TABLE IF NOT EXISTS audit_dlq (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    payload     JSONB        NOT NULL,
    retry_count INTEGER      NOT NULL DEFAULT 0,
    last_error  TEXT
);

-- HITL approval requests
CREATE TABLE IF NOT EXISTS hitl_approvals (
    id          BIGSERIAL PRIMARY KEY,
    trace_id    UUID          NOT NULL,
    session_id  VARCHAR(255)  NOT NULL,
    tool_name   VARCHAR(255)  NOT NULL,
    tool_input  JSONB         NOT NULL,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at  TIMESTAMPTZ,
    decision    VARCHAR(20),   -- approved | rejected | timeout
    decided_by  VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS idx_hitl_trace ON hitl_approvals(trace_id);
CREATE INDEX IF NOT EXISTS idx_hitl_pending ON hitl_approvals(decision) WHERE decision IS NULL;
