"""Parser for OpenAI chat completion tool call responses."""

from __future__ import annotations

import json
from typing import Any

from russo._types import AgentResponse, ToolCall


class OpenAIResponseParser:
    """Parses OpenAI ChatCompletion responses into AgentResponse.

    Handles the OpenAI format where tool calls appear at:
    response.choices[*].message.tool_calls[*]

    Each tool call has: {id, type, function: {name, arguments}}.

    Works with both the raw dict format and the openai SDK objects.

    Usage:
        parser = OpenAIResponseParser()
        response = parser.parse(openai_raw_response)
    """

    def parse(self, raw_response: Any) -> AgentResponse:
        """Parse an OpenAI response into a normalized AgentResponse."""
        tool_calls: list[ToolCall] = []

        choices = _get_attr_or_key(raw_response, "choices", [])
        for choice in choices:
            message = _get_attr_or_key(choice, "message", None)
            if message is None:
                continue
            raw_tool_calls = _get_attr_or_key(message, "tool_calls", [])
            if not raw_tool_calls:
                continue
            for tc in raw_tool_calls:
                function = _get_attr_or_key(tc, "function", None)
                if function is None:
                    continue
                name = _get_attr_or_key(function, "name", "")
                arguments_raw = _get_attr_or_key(function, "arguments", "{}")
                if isinstance(arguments_raw, str):
                    try:
                        arguments = json.loads(arguments_raw)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {}
                elif isinstance(arguments_raw, dict):
                    arguments = arguments_raw
                else:
                    arguments = {}
                tool_calls.append(ToolCall(name=name, arguments=arguments))

        return AgentResponse(tool_calls=tool_calls, raw=raw_response)


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    """Get a value from an object by attribute or dict key."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
