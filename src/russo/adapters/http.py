"""HTTP agent adapter — sends audio to an API endpoint."""

from __future__ import annotations

import base64
import json
from typing import Any
from urllib.request import Request, urlopen

from russo._protocols import ResponseParser
from russo._types import AgentResponse, Audio


class HttpAgent:
    """Agent adapter that sends audio to an HTTP endpoint.

    Sends audio as base64-encoded JSON and parses the response
    using an optional ResponseParser.

    Usage:
        agent = HttpAgent(
            url="http://localhost:8000/voice-agent",
            parser=russo.parsers.GeminiResponseParser(),
        )
        response = await agent.run(audio)
    """

    def __init__(
        self,
        *,
        url: str,
        parser: ResponseParser | None = None,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        audio_field: str = "audio",
        format_field: str = "format",
        timeout: float = 60.0,
    ) -> None:
        self.url = url
        self.parser = parser
        self.method = method
        self.headers = headers or {}
        self.audio_field = audio_field
        self.format_field = format_field
        self.timeout = timeout

    async def run(self, audio: Audio) -> AgentResponse:
        """Send audio to the HTTP endpoint and parse the response."""
        payload = {
            self.audio_field: base64.b64encode(audio.data).decode("ascii"),
            self.format_field: audio.format,
        }

        raw_response = await self._send(payload)

        if self.parser:
            return self.parser.parse(raw_response)

        return self._default_parse(raw_response)

    async def _send(self, payload: dict[str, Any]) -> Any:
        """Send the HTTP request. Uses stdlib urllib for zero extra deps."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._send_sync, payload)

    def _send_sync(self, payload: dict[str, Any]) -> Any:
        """Synchronous HTTP send."""
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        req = Request(self.url, data=data, headers=headers, method=self.method)
        with urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    def _default_parse(self, raw: Any) -> AgentResponse:
        """Fallback parser: expects {"tool_calls": [{"name": ..., "arguments": ...}]}."""
        from russo._types import ToolCall

        if isinstance(raw, dict) and "tool_calls" in raw:
            calls = [ToolCall(name=tc["name"], arguments=tc.get("arguments", {})) for tc in raw["tool_calls"]]
            return AgentResponse(tool_calls=calls, raw=raw)
        return AgentResponse(tool_calls=[], raw=raw)
