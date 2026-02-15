"""Parser for Google Gemini function call responses."""

from __future__ import annotations

from typing import Any

from russo._types import AgentResponse, ToolCall


class GeminiResponseParser:
    """Parses Gemini GenerateContentResponse into AgentResponse.

    Handles the Gemini response format where tool calls appear as
    function_call parts in response.candidates[*].content.parts[*].

    Works with both the raw dict format and the google-genai SDK objects.

    Usage:
        parser = GeminiResponseParser()
        response = parser.parse(gemini_raw_response)
    """

    def parse(self, raw_response: Any) -> AgentResponse:
        """Parse a Gemini response into a normalized AgentResponse."""
        tool_calls: list[ToolCall] = []

        # Handle google-genai SDK response objects
        candidates = _get_attr_or_key(raw_response, "candidates", [])
        for candidate in candidates:
            content = _get_attr_or_key(candidate, "content", None)
            if content is None:
                continue
            parts = _get_attr_or_key(content, "parts", [])
            for part in parts:
                fc = _get_attr_or_key(part, "function_call", None)
                if fc is not None:
                    name = _get_attr_or_key(fc, "name", "")
                    args = _get_attr_or_key(fc, "args", {})
                    if isinstance(args, str):
                        import json

                        args = json.loads(args)
                    tool_calls.append(ToolCall(name=name, arguments=dict(args) if args else {}))

        return AgentResponse(tool_calls=tool_calls, raw=raw_response)


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    """Get a value from an object by attribute or dict key."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
