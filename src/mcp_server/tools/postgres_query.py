import json
import re
import asyncpg

# Only SELECT statements are permitted — block all DML/DDL
_SELECT_ONLY = re.compile(r"^\s*SELECT\b", re.IGNORECASE)


async def postgres_query_tool(query: str, pool: asyncpg.Pool) -> str:
    if not _SELECT_ONLY.match(query.strip()):
        return json.dumps({"error": "Only SELECT statements are permitted"})

    try:
        async with pool.acquire() as conn:
            # Wrap in a read-only transaction so even clever injection can't mutate
            async with conn.transaction(readonly=True):
                rows = await conn.fetch(query)
        return json.dumps([dict(row) for row in rows], default=str)
    except asyncpg.PostgresSyntaxError as e:
        return json.dumps({"error": f"Syntax error: {e}"})
    except asyncpg.PostgresError as e:
        return json.dumps({"error": str(e)})
