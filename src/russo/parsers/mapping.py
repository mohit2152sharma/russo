"""Configurable parser for custom JSON response structures."""

from __future__ import annotations

import json
from typing import Any

from russo._types import AgentResponse, ToolCall


class JsonResponseParser:
    """Configurable parser for endpoints that return tool calls in a custom JSON structure.

    Instead of writing a full parser class, pass field name config to match
    whatever your HTTP or WebSocket endpoint returns.

    Args:
        tool_calls_key: Dot-separated path to the tool calls in the response.
            Supports nested paths (e.g. ``"result.toolCalls"``).
            Default: ``"tool_calls"``.
        name_key: Key for the function name within each tool call dict.
            Default: ``"name"``.
        arguments_key: Key for the arguments dict within each tool call.
            Default: ``"arguments"``.
        single: Set to ``True`` if the endpoint returns a single tool call
            object instead of a list. Default: ``False``.

    Usage::

        # Endpoint returns: {"toolCall": {"name": "fn", "arguments": {...}}}
        parser = JsonResponseParser(tool_calls_key="toolCall", single=True)

        # Endpoint returns: {"result": {"calls": [{"fn": "...", "params": {...}}]}}
        parser = JsonResponseParser(
            tool_calls_key="result.calls",
            name_key="fn",
            arguments_key="params",
        )

        # Use with HttpAgent or WebSocketAgent
        agent = HttpAgent(url="http://localhost:8000/agent", parser=parser)
        agent = WebSocketAgent(url="ws://localhost:8000/ws", parser=parser)
    """

    def __init__(
        self,
        *,
        tool_calls_key: str = "tool_calls",
        name_key: str = "name",
        arguments_key: str = "arguments",
        single: bool = False,
    ) -> None:
        self.tool_calls_key = tool_calls_key
        self.name_key = name_key
        self.arguments_key = arguments_key
        self.single = single

    def parse(self, raw_response: Any) -> AgentResponse:
        """Parse a JSON response into a normalized AgentResponse."""
        # If the response is a list (e.g. aggregated WebSocket messages),
        # search each item and return on first hit.
        if isinstance(raw_response, list):
            for item in raw_response:
                result = self._try_parse(item)
                if result is not None:
                    return AgentResponse(tool_calls=result, raw=raw_response)
            return AgentResponse(tool_calls=[], raw=raw_response)

        tool_calls = self._try_parse(raw_response)
        return AgentResponse(tool_calls=tool_calls or [], raw=raw_response)

    def _try_parse(self, obj: Any) -> list[ToolCall] | None:
        """Extract tool calls from a single response object. Returns None on miss."""
        raw_calls = _extract_path(obj, self.tool_calls_key)
        if raw_calls is None:
            return None

        if self.single:
            raw_calls = [raw_calls]

        if not isinstance(raw_calls, list):
            return None

        tool_calls: list[ToolCall] = []
        for tc in raw_calls:
            if not isinstance(tc, dict):
                continue
            name = tc.get(self.name_key, "")
            if not name:
                continue
            arguments = tc.get(self.arguments_key, {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
            tool_calls.append(ToolCall(name=name, arguments=arguments if isinstance(arguments, dict) else {}))

        return tool_calls if tool_calls else None


def _extract_path(obj: Any, path: str) -> Any:
    """Walk a dot-separated key path through nested dicts."""
    current = obj
    for key in path.split("."):
        if isinstance(current, dict):
            if key not in current:
                return None
            current = current[key]
        else:
            return None
    return current
