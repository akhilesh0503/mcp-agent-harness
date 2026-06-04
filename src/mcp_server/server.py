import asyncio
import os
from contextlib import asynccontextmanager

import asyncpg
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from src.mcp_server.tools.file_read import file_read_tool
from src.mcp_server.tools.http_api_call import http_api_call_tool
from src.mcp_server.tools.postgres_query import postgres_query_tool

load_dotenv()

# Module-level pool — initialised in lifespan, used by tool handlers
_db_pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(server: FastMCP):
    global _db_pool
    _db_pool = await asyncpg.create_pool(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "harness_db"),
        user=os.getenv("POSTGRES_USER", "harness"),
        password=os.getenv("POSTGRES_PASSWORD", "harness_secret"),
        min_size=2,
        max_size=10,
    )
    yield
    if _db_pool:
        await _db_pool.close()
    _db_pool = None


mcp = FastMCP("agent-harness-tools", lifespan=lifespan)


# ── Tool registrations ────────────────────────────────────────────────────────

@mcp.tool()
async def postgres_query(query: str) -> str:
    """
    Execute a read-only SQL SELECT query against PostgreSQL.
    Only SELECT statements are permitted — all DML/DDL is blocked.
    Returns a JSON array of row objects.
    """
    assert _db_pool is not None, "DB pool not initialised"
    return await postgres_query_tool(query, _db_pool)


@mcp.tool()
async def http_api_call(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    body: dict | None = None,
) -> str:
    """
    Make an HTTP GET or POST request to an external API.
    Internal/private IPs and non-http(s) schemes are blocked.
    Returns JSON with status_code, headers, and body.
    """
    return await http_api_call_tool(url, method, headers, body)


@mcp.tool()
async def file_read(path: str) -> str:
    """
    Read a file from the sandboxed base directory (FILE_READ_BASE_DIR).
    Path is relative to the base dir. Path traversal attempts are rejected.
    Returns JSON with path and content. Max file size: 1 MB.
    """
    return await file_read_tool(path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=int(os.getenv("MCP_SERVER_PORT", "8001")),
    )
