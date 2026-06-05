from __future__ import annotations

import logging

from src.harness.metrics import REGISTRY_CHECKS
from src.harness.models import PipelineContext, ResultStatus, ToolResult

logger = logging.getLogger(__name__)

# JSON Schema type → Python type(s)
_TYPE_MAP: dict[str, type | tuple] = {
    "string":  str,
    "integer": int,
    "number":  (int, float),
    "boolean": bool,
    "object":  dict,
    "array":   list,
    "null":    type(None),
}


class ToolRegistry:
    """
    Layer 3. Validates that the tool exists and that the LLM-supplied
    arguments conform to the tool's declared JSON Schema.

    Tool definitions are loaded from the MCP server's list_tools() response
    at pipeline startup via register_many(). This keeps the schema definition
    in one place (the MCP server) rather than duplicating it here.
    """

    def __init__(self) -> None:
        # tool_name → input_schema (JSON Schema object)
        self._schemas: dict[str, dict] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, name: str, input_schema: dict) -> None:
        self._schemas[name] = input_schema
        logger.debug("ToolRegistry: registered tool '%s'", name)

    def register_many(self, tools: list[dict]) -> None:
        """
        Accepts the list returned by MCP ClientSession.list_tools().
        Each entry has: name, description, inputSchema.
        """
        for tool in tools:
            self.register(tool["name"], tool.get("inputSchema", {}))
        logger.info("ToolRegistry: loaded %d tools from MCP server", len(tools))

    @property
    def registered_tools(self) -> list[str]:
        return list(self._schemas.keys())

    # ── Pipeline check ────────────────────────────────────────────────────────

    async def check(self, ctx: PipelineContext) -> bool:
        tool_name = ctx.tool_call.tool_name

        if tool_name not in self._schemas:
            REGISTRY_CHECKS.labels(tool=tool_name, result="unknown_tool").inc()
            return self._reject(
                ctx,
                f"Tool '{tool_name}' is not registered. "
                f"Available: {self.registered_tools}",
            )

        error = self._validate(self._schemas[tool_name], ctx.tool_call.arguments)
        if error:
            REGISTRY_CHECKS.labels(tool=tool_name, result="schema_error").inc()
            return self._reject(ctx, f"Invalid arguments for '{tool_name}': {error}")

        REGISTRY_CHECKS.labels(tool=tool_name, result="valid").inc()
        return True

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self, schema: dict, args: dict) -> str | None:
        """Return an error string, or None if the args are valid."""
        properties: dict = schema.get("properties", {})
        required: list[str] = schema.get("required", [])

        # Check required fields are present
        for field in required:
            if field not in args:
                return f"Missing required argument: '{field}'"

        # Check no unknown keys were supplied
        for key in args:
            if properties and key not in properties:
                return f"Unexpected argument: '{key}'"

        # Check types of supplied arguments
        for key, value in args.items():
            prop = properties.get(key, {})
            expected_type = prop.get("type")
            if expected_type and not self._type_ok(value, expected_type):
                return (
                    f"Argument '{key}' must be of type '{expected_type}', "
                    f"got '{type(value).__name__}'"
                )

        return None

    @staticmethod
    def _type_ok(value: object, expected: str) -> bool:
        cls = _TYPE_MAP.get(expected)
        if cls is None:
            return True  # unknown type annotation — allow
        # booleans are ints in Python; distinguish them explicitly
        if expected == "integer" and isinstance(value, bool):
            return False
        return isinstance(value, cls)

    # ── Helper ────────────────────────────────────────────────────────────────

    def _reject(self, ctx: PipelineContext, reason: str) -> bool:
        ctx.result_status = ResultStatus.REJECTED
        ctx.rejected_at_layer = "ToolRegistry"
        ctx.error_message = reason
        ctx.result = ToolResult(
            tool_call_id=ctx.tool_call.tool_call_id,
            content=reason,
            is_error=True,
        )
        return False
