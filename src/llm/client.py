"""
OpenRouter client for captain decision-making.

Uses OpenRouter API directly with httpx for tool/function calling.
"""

import os
import json
import httpx
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class LLMResponse:
    """Response from LLM API call."""
    content: str
    tool_calls: List[Any]
    model: str
    usage: Dict[str, int]
    raw_response: Any = None


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


class CaptainClient:
    """
    LLM client for captain decision-making using OpenRouter directly.

    Uses tool/function calling for structured command output.
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str = "anthropic/claude-3.5-sonnet",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        """
        Initialize the captain client.

        Args:
            model: Model ID (e.g., "anthropic/claude-3.5-sonnet")
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
        """
        # Strip openrouter/ prefix if present
        if model.startswith("openrouter/"):
            model = model[len("openrouter/"):]

        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key."
            )

        self._client = httpx.Client(timeout=60.0)

    def decide_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
    ) -> List[ToolCall]:
        """
        Make a decision using tool/function calling.

        Args:
            messages: Conversation messages (system, user, assistant)
            tools: Available tools in OpenAI function calling format

        Returns:
            List of ToolCall objects representing the LLM's decisions
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ai-commanders",
            "X-Title": "AI Commanders",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            response = self._client.post(
                self.BASE_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            # Extract tool calls from response
            message = data["choices"][0]["message"]
            tool_calls = []

            if "tool_calls" in message and message["tool_calls"]:
                for tc in message["tool_calls"]:
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = {}

                    tool_calls.append(ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=args,
                    ))

            return tool_calls

        except httpx.HTTPStatusError as e:
            print(f"[LLM ERROR] HTTP {e.response.status_code}: {e.response.text}")
            return []
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return []

    def complete(
        self,
        messages: List[Dict[str, str]],
    ) -> LLMResponse:
        """
        Make a simple completion without tools.

        Args:
            messages: Conversation messages

        Returns:
            LLMResponse with content
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ai-commanders",
            "X-Title": "AI Commanders",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = self._client.post(
            self.BASE_URL,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        message = data["choices"][0]["message"]

        return LLMResponse(
            content=message.get("content", ""),
            tool_calls=[],
            model=data.get("model", self.model),
            usage=data.get("usage", {}),
            raw_response=data,
        )
