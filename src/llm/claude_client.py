from __future__ import annotations

import logging
from uuid import uuid4

import anthropic

from src.config import settings
from src.harness.models import ConversationMessage, ToolCall
from src.llm.base import SYSTEM_PROMPT, LLMClient, LLMResponse

logger = logging.getLogger(__name__)

# Anthropic tool call block type
_TOOL_USE = "tool_use"


class ClaudeClient(LLMClient):
    """
    LLM client backed by the Anthropic Claude API.
    Uses the official anthropic Python SDK.

    Message format differs from OpenAI/Ollama:
      - Tool calls are content blocks of type 'tool_use' (not tool_calls array)
      - Tool results are injected as a user turn with 'tool_result' content blocks
      - Tool definitions use 'input_schema' (not 'parameters')

    All format differences are handled internally — the pipeline sees only
    the shared LLMResponse / ConversationMessage types.
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self._model  = model
        self._client = anthropic.AsyncAnthropic()   # reads ANTHROPIC_API_KEY from env

    # ── LLMClient interface ───────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[ConversationMessage],
        tools: list[dict],
    ) -> LLMResponse:
        api_messages = self._build_messages(messages)

        logger.debug("Claude request: model=%s messages=%d tools=%d",
                     self._model, len(api_messages), len(tools))

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=api_messages,
            tools=tools,
        )

        return self._parse(response)

    def tool_definitions(self, mcp_tools: list[dict]) -> list[dict]:
        """Convert MCP list_tools() response to Anthropic tool format."""
        return [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema", {
                    "type": "object",
                    "properties": {},
                }),
            }
            for tool in mcp_tools
        ]

    # ── Message conversion ────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(messages: list[ConversationMessage]) -> list[dict]:
        """
        Convert internal ConversationMessages to Anthropic's message format.
        Groups consecutive tool results into a single user turn as required
        by the Anthropic API.
        """
        api: list[dict] = []
        pending_tool_results: list[dict] = []

        def flush_tool_results() -> None:
            if pending_tool_results:
                api.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results.clear()

        for msg in messages:
            if msg.role == "system":
                continue  # passed as top-level system param, not in messages

            if msg.role == "tool":
                # Accumulate tool results — they go in a single user turn
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "",
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                })
                continue

            flush_tool_results()

            if msg.role == "assistant" and msg.tool_calls:
                # Reconstruct the assistant content blocks from raw tool_calls
                content: list[dict] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in (msg.tool_calls or []):
                    fn = tc.get("function", {})
                    import json
                    try:
                        input_data = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        input_data = {}
                    content.append({
                        "type": _TOOL_USE,
                        "id":   tc.get("id", str(uuid4())),
                        "name": fn.get("name", ""),
                        "input": input_data,
                    })
                api.append({"role": "assistant", "content": content})
            else:
                content_str = (
                    msg.content if isinstance(msg.content, str)
                    else str(msg.content or "")
                )
                api.append({"role": msg.role, "content": content_str})

        flush_tool_results()
        return api

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse(self, response) -> LLMResponse:
        token_count = (
            response.usage.input_tokens + response.usage.output_tokens
            if response.usage else 0
        )

        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []

        for block in response.content:
            if block.type == _TOOL_USE:
                tool_calls.append(ToolCall(
                    tool_call_id=block.id,
                    tool_name=block.name,
                    arguments=block.input,
                ))
            elif block.type == "text" and block.text:
                text_parts.append(block.text)

        if tool_calls:
            return LLMResponse(tool_calls=tool_calls, token_count=token_count)

        return LLMResponse(
            text=" ".join(text_parts) or "",
            token_count=token_count,
        )
