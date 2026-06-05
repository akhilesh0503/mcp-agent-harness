from __future__ import annotations

import json
import logging
from uuid import uuid4

import httpx

from src.config import settings
from src.harness.models import ConversationMessage, ToolCall
from src.llm.base import SYSTEM_PROMPT, LLMClient, LLMResponse

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """
    LLM client backed by a locally-running Ollama instance.
    Uses Ollama's OpenAI-compatible /v1/chat/completions endpoint.
    Zero API cost — model runs entirely on the local machine.

    Swap to ClaudeClient by changing one line in the pipeline factory.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model   = model   or settings.ollama_model
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._endpoint = f"{self._base_url}/v1/chat/completions"

    # ── LLMClient interface ───────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[ConversationMessage],
        tools: list[dict],
    ) -> LLMResponse:
        payload = {
            "model": self._model,
            "messages": self._build_messages(messages),
            "tools": tools,
            "tool_choice": "auto" if tools else "none",
            "stream": False,
        }

        logger.debug("Ollama request: model=%s messages=%d tools=%d",
                     self._model, len(payload["messages"]), len(tools))

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(self._endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return self._parse(data)

    def tool_definitions(self, mcp_tools: list[dict]) -> list[dict]:
        """Convert MCP list_tools() response to OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {
                        "type": "object",
                        "properties": {},
                    }),
                },
            }
            for tool in mcp_tools
        ]

    # ── Message conversion ────────────────────────────────────────────────────

    def _build_messages(self, messages: list[ConversationMessage]) -> list[dict]:
        """
        Prepend a system message if one isn't already present, then convert
        each ConversationMessage to the OpenAI wire format.
        """
        api_messages: list[dict] = []

        if not messages or messages[0].role != "system":
            api_messages.append({"role": "system", "content": SYSTEM_PROMPT})

        for msg in messages:
            api_messages.append(self._to_api(msg))

        return api_messages

    @staticmethod
    def _to_api(msg: ConversationMessage) -> dict:
        if msg.role == "tool":
            # Tool result — injected back after harness executes a tool call
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id or "",
                "content": msg.content if isinstance(msg.content, str) else json.dumps(msg.content),
            }

        if msg.role == "assistant" and msg.tool_calls:
            # Assistant message that requested tool calls
            return {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": msg.tool_calls,   # already in OpenAI format
            }

        # Plain user or assistant message
        content = msg.content
        if not isinstance(content, str):
            content = json.dumps(content) if content is not None else ""
        return {"role": msg.role, "content": content}

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse(self, data: dict) -> LLMResponse:
        token_count = data.get("usage", {}).get("total_tokens", 0)

        try:
            choice  = data["choices"][0]
            message = choice["message"]
            finish  = choice.get("finish_reason", "stop")
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Unexpected Ollama response shape: {exc}") from exc

        raw_tool_calls = message.get("tool_calls") or []

        if finish == "tool_calls" or raw_tool_calls:
            return LLMResponse(
                tool_calls=self._parse_tool_calls(raw_tool_calls),
                token_count=token_count,
            )

        return LLMResponse(
            text=message.get("content") or "",
            token_count=token_count,
        )

    @staticmethod
    def _parse_tool_calls(raw: list[dict]) -> list[ToolCall]:
        calls = []
        for tc in raw:
            fn = tc.get("function", {})
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                logger.warning("Could not parse tool call arguments: %r", args_str)
                args = {}
            calls.append(ToolCall(
                tool_call_id=tc.get("id") or str(uuid4()),
                tool_name=fn.get("name", ""),
                arguments=args,
            ))
        return calls
