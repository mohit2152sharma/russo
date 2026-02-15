"""WebSocket agent adapter — sends audio to a WebSocket endpoint.

Requires the `websockets` package: pip install russo[ws]
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable
from typing import Any

from russo._protocols import ResponseParser
from russo._types import AgentResponse, Audio, ToolCall

logger = logging.getLogger("russo.adapters.websocket")

try:
    import websockets  # noqa: F401

    _HAS_WEBSOCKETS = True
except ImportError:
    _HAS_WEBSOCKETS = False


class WebSocketAgent:
    """Agent adapter that communicates over WebSocket.

    Connects to the endpoint, sends audio, collects response messages
    until a completion condition is met, then parses tool calls.

    Supports two send modes:
    - **json** (default): sends ``{"audio": "<base64>", "format": "wav"}``
    - **bytes**: sends raw audio bytes directly on the wire

    And two ways to customize the protocol:
    - **on_send**: transform the Audio into whatever your server expects
    - **is_complete**: decide when to stop collecting response messages

    Usage:
        # Simple JSON protocol
        agent = WebSocketAgent(
            url="ws://localhost:8000/ws/agent",
            parser=russo.parsers.GeminiResponseParser(),
        )

        # Raw bytes protocol (e.g. streaming PCM)
        agent = WebSocketAgent(
            url="ws://localhost:8000/ws/agent",
            send_bytes=True,
        )

        # Fully custom send/receive
        agent = WebSocketAgent(
            url="ws://localhost:8000/ws/agent",
            on_send=lambda audio: json.dumps({"pcm": base64.b64encode(audio.data).decode()}),
            is_complete=lambda msgs: any('"done": true' in str(m) for m in msgs),
        )
    """

    def __init__(
        self,
        *,
        url: str,
        parser: ResponseParser | None = None,
        headers: dict[str, str] | None = None,
        # Send options
        send_bytes: bool = False,
        audio_field: str = "audio",
        format_field: str = "format",
        # Response collection
        response_timeout: float = 30.0,
        max_messages: int = 100,
        # Hooks
        on_send: Callable[[Audio], str | bytes] | None = None,
        is_complete: Callable[[list[Any]], bool] | None = None,
        aggregate: Callable[[list[Any]], Any] | None = None,
        # Connection
        open_timeout: float = 10.0,
        close_timeout: float = 5.0,
        extra_ws_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not _HAS_WEBSOCKETS:
            msg = "WebSocketAgent requires the 'websockets' package. Install with: pip install russo[ws]"
            raise ImportError(msg)

        self.url = url
        self.parser = parser
        self.headers = headers or {}
        self.send_bytes = send_bytes
        self.audio_field = audio_field
        self.format_field = format_field
        self.response_timeout = response_timeout
        self.max_messages = max_messages
        self.on_send = on_send
        self.is_complete = is_complete
        self.aggregate = aggregate
        self.open_timeout = open_timeout
        self.close_timeout = close_timeout
        self.extra_ws_kwargs = extra_ws_kwargs or {}

    async def run(self, audio: Audio) -> AgentResponse:
        """Send audio over WebSocket and collect the response."""
        import websockets

        async with websockets.connect(
            self.url,
            additional_headers=self.headers or None,
            open_timeout=self.open_timeout,
            close_timeout=self.close_timeout,
            **self.extra_ws_kwargs,
        ) as ws:
            # --- Send ---
            message = self._prepare_message(audio)
            if isinstance(message, bytes):
                await ws.send(message)
            else:
                await ws.send(message)
            logger.debug(
                "Sent %s message (%d bytes)",
                "binary" if isinstance(message, bytes) else "text",
                len(message),
            )

            # --- Collect responses ---
            messages = await self._collect_responses(ws)
            logger.debug("Collected %d response messages", len(messages))

        # --- Parse ---
        raw = self._aggregate(messages)

        if self.parser:
            return self.parser.parse(raw)
        return self._default_parse(raw)

    def _prepare_message(self, audio: Audio) -> str | bytes:
        """Build the outgoing message."""
        if self.on_send:
            return self.on_send(audio)
        if self.send_bytes:
            return audio.data
        return json.dumps(
            {
                self.audio_field: base64.b64encode(audio.data).decode("ascii"),
                self.format_field: audio.format,
            }
        )

    async def _collect_responses(self, ws: Any) -> list[Any]:
        """Read messages until completion condition or timeout."""
        messages: list[Any] = []
        try:
            async with asyncio.timeout(self.response_timeout):
                async for msg in ws:
                    parsed = self._parse_incoming(msg)
                    messages.append(parsed)

                    if self.is_complete and self.is_complete(messages):
                        break
                    if len(messages) >= self.max_messages:
                        logger.warning(
                            "Hit max_messages limit (%d), stopping collection",
                            self.max_messages,
                        )
                        break

                    # If no is_complete hook, take only the first message
                    if not self.is_complete:
                        break
        except TimeoutError:
            if not messages:
                logger.warning(
                    "WebSocket response timed out after %.1fs with no messages",
                    self.response_timeout,
                )
            else:
                logger.debug(
                    "Response collection timed out after %.1fs with %d messages",
                    self.response_timeout,
                    len(messages),
                )
        return messages

    def _parse_incoming(self, msg: str | bytes) -> Any:
        """Try to JSON-parse an incoming message, fall back to raw."""
        if isinstance(msg, bytes):
            try:
                return json.loads(msg.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return msg
        try:
            return json.loads(msg)
        except json.JSONDecodeError:
            return msg

    def _aggregate(self, messages: list[Any]) -> Any:
        """Combine collected messages into a single response object."""
        if self.aggregate:
            return self.aggregate(messages)
        if len(messages) == 1:
            return messages[0]
        return messages

    def _default_parse(self, raw: Any) -> AgentResponse:
        """Fallback parser: expects {"tool_calls": [{"name": ..., "arguments": ...}]}."""
        if isinstance(raw, dict) and "tool_calls" in raw:
            calls = [ToolCall(name=tc["name"], arguments=tc.get("arguments", {})) for tc in raw["tool_calls"]]
            return AgentResponse(tool_calls=calls, raw=raw)
        # If aggregated into a list, check each message
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and "tool_calls" in item:
                    calls = [ToolCall(name=tc["name"], arguments=tc.get("arguments", {})) for tc in item["tool_calls"]]
                    return AgentResponse(tool_calls=calls, raw=raw)
        return AgentResponse(tool_calls=[], raw=raw)
