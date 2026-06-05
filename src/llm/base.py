from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.harness.models import ConversationMessage, ToolCall

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools. "
    "Use tools whenever they can help you answer accurately. "
    "After receiving a tool result, use it to give a clear, concise response. "
    "Never reveal internal tool call details to the user."
)


@dataclass
class LLMResponse:
    """
    Unified response from any LLM provider.
    Exactly one of (text, tool_calls) will be populated per turn:
      - tool_calls non-empty → LLM wants to call tools; pipeline runs them
      - text set             → LLM is done; final answer goes back to the user
    token_count is always set for budget tracking.
    """
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    token_count: int = 0

    @property
    def wants_tools(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def is_final(self) -> bool:
        return self.text is not None and not self.tool_calls


class LLMClient(ABC):
    """
    Provider-agnostic interface for an LLM with tool-calling support.
    Concrete implementations: OllamaClient, ClaudeClient.

    The pipeline only ever calls:
      - chat()            — one conversational turn
      - tool_definitions() — convert MCP tool list to provider format
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[ConversationMessage],
        tools: list[dict],
    ) -> LLMResponse:
        """
        Send a conversation turn and return the model's response.

        Args:
            messages: full conversation history in internal format
            tools:    provider-specific tool definitions (from tool_definitions())

        Returns:
            LLMResponse with either tool_calls or text, plus token_count
        """
        ...

    @abstractmethod
    def tool_definitions(self, mcp_tools: list[dict]) -> list[dict]:
        """
        Convert the MCP server's list_tools() response to the provider's
        tool definition format, ready to pass directly into chat().

        Args:
            mcp_tools: list of {name, description, inputSchema} from MCP

        Returns:
            Provider-formatted list of tool definitions
        """
        ...
