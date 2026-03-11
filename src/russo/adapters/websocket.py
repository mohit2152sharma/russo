"""WebSocket agent adapter — sends audio to a WebSocket endpoint.

Requires the `websockets` package: pip install russo[ws]
"""

from __future__ import annotations

import asyncio
import base64
import collections
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from russo._protocols import ResponseParser
from russo._types import AgentResponse, Audio, ToolCall

logger = logging.getLogger("russo.adapters.websocket")

try:
    import websockets  # noqa: F401

    _HAS_WEBSOCKETS = True
except ImportError:
    _HAS_WEBSOCKETS = False


class WebSocketSession:
    """Active WebSocket session with full-duplex communication.

    Obtained via :meth:`WebSocketAgent.session`. Wraps a live ``websockets``
    connection and adds structured send/receive primitives plus event-triggered
    sending.

    **Sending**

    - :meth:`send_text` — send a raw text frame
    - :meth:`send_bytes` — send a raw binary frame
    - :meth:`send_json` — JSON-encode an object and send as a text frame

    **Receiving**

    - :meth:`receive` — pull the next message (drains internal buffer first)
    - :meth:`wait_for` — block until a matching message arrives; non-matching
      messages are pushed into an internal buffer so they are not lost
    - :meth:`send_after` — convenience: wait for a trigger event then
      immediately send, in a single call

    **Iteration**

    The session is async-iterable::

        async for msg in session:
            ...  # stops when the connection closes

    **Internal buffer**

    :meth:`wait_for` may read ahead to find a matching message. Any messages
    that arrive before the match are kept in an internal FIFO buffer.
    Subsequent calls to :meth:`receive` or async iteration drain that buffer
    before touching the wire again.
    """

    def __init__(
        self,
        ws: Any,
        parse_incoming: Callable[[str | bytes], Any],
    ) -> None:
        self._ws = ws
        self._parse = parse_incoming
        self._buffer: collections.deque[Any] = collections.deque()

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    async def send_text(self, text: str) -> None:
        """Send a text frame."""
        await self._ws.send(text)

    async def send_bytes(self, data: bytes) -> None:
        """Send a binary frame."""
        await self._ws.send(data)

    async def send_json(self, obj: Any) -> None:
        """JSON-encode *obj* and send as a text frame."""
        await self._ws.send(json.dumps(obj))

    # ------------------------------------------------------------------
    # Receive
    # ------------------------------------------------------------------

    async def _next_message(self) -> Any:
        """Return the next parsed message from the buffer or the wire."""
        if self._buffer:
            return self._buffer.popleft()
        raw = await self._ws.recv()
        return self._parse(raw)

    async def receive(self, timeout: float | None = None) -> Any:
        """Return the next incoming message.

        Drains the internal buffer before reading from the wire.

        Args:
            timeout: Seconds to wait. ``None`` (default) blocks indefinitely.

        Raises:
            TimeoutError: If *timeout* elapses with no message available.
        """
        if timeout is not None:
            async with asyncio.timeout(timeout):
                return await self._next_message()
        return await self._next_message()

    async def wait_for(
        self,
        predicate: Callable[[Any], bool],
        timeout: float = 30.0,
    ) -> Any:
        """Wait until a message satisfying *predicate* arrives.

        Messages that do not match are pushed into the internal buffer so they
        are not lost; subsequent :meth:`receive` calls or async iteration will
        return them in order.

        Args:
            predicate: Callable that receives a parsed message and returns
                ``True`` to stop waiting, ``False`` to keep reading.
            timeout: Maximum seconds to wait (default 30).

        Returns:
            The first message for which *predicate* returned ``True``.

        Raises:
            TimeoutError: If no matching message arrives within *timeout*.

        Example::

            # Block until the server signals readiness
            await session.wait_for(
                lambda m: isinstance(m, dict) and m.get("ready")
            )
        """
        async with asyncio.timeout(timeout):
            while True:
                msg = await self._next_message()
                if predicate(msg):
                    return msg
                self._buffer.append(msg)

    async def send_after(
        self,
        predicate: Callable[[Any], bool],
        *,
        text: str | None = None,
        data: bytes | None = None,
        json_data: Any = None,
        timeout: float = 30.0,
    ) -> Any:
        """Wait for a matching event, then send — in a single call.

        Exactly one of *text*, *data*, or *json_data* should be provided.
        Non-matching messages received while waiting are buffered (see
        :meth:`wait_for`).

        Args:
            predicate: Same semantics as :meth:`wait_for`.
            text: Text frame to send after the trigger.
            data: Binary frame to send after the trigger.
            json_data: Object to JSON-encode and send after the trigger.
            timeout: Forwarded to :meth:`wait_for`.

        Returns:
            The triggering message (the one matched by *predicate*).

        Example::

            # Send audio only after the server signals setup is complete
            await session.send_after(
                lambda m: isinstance(m, dict) and m.get("setupComplete"),
                json_data={"audio": base64_audio, "format": "wav"},
            )
        """
        trigger = await self.wait_for(predicate, timeout=timeout)
        if json_data is not None:
            await self.send_json(json_data)
        elif data is not None:
            await self.send_bytes(data)
        elif text is not None:
            await self.send_text(text)
        return trigger

    # ------------------------------------------------------------------
    # Async iteration
    # ------------------------------------------------------------------

    def __aiter__(self) -> WebSocketSession:
        return self

    async def __anext__(self) -> Any:
        import websockets.exceptions

        try:
            return await self._next_message()
        except (
            websockets.exceptions.ConnectionClosedOK,
            websockets.exceptions.ConnectionClosedError,
        ):
            raise StopAsyncIteration

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying WebSocket connection."""
        await self._ws.close()


class WebSocketAgent:
    """Agent adapter that communicates over WebSocket.

    Supports two usage modes:

    **High-level** — call :meth:`run` with an :class:`~russo.Audio` object.
    The adapter connects, runs the ``on_connect`` hook (if set), sends the
    audio, collects responses until a completion condition is met, then
    parses tool calls.

    **Session-based** — use :meth:`session` as an async context manager to
    obtain a :class:`WebSocketSession` and drive the protocol yourself::

        async with agent.session() as sess:
            # Wait for server-ready before sending anything
            await sess.wait_for(lambda m: m.get("ready"))

            await sess.send_json({"audio": base64_data, "format": "wav"})

            async for msg in sess:
                if is_done(msg):
                    break

    **Send modes**

    - **json** (default): ``{"audio": "<base64>", "format": "wav"}``
    - **bytes**: raw audio bytes on the wire (``send_bytes=True``)

    **Hooks** (apply to both :meth:`run` and :meth:`session`)

    - ``on_connect``: called with the :class:`WebSocketSession` right after
      the connection opens, before anything else — use it to wait for a
      server-ready message
    - ``on_send``: transform an :class:`~russo.Audio` into the wire format
      your server expects (bypasses *send_bytes* / *audio_field*)
    - ``is_complete``: return ``True`` when :meth:`run` should stop collecting
      response messages
    - ``aggregate``: fold multiple response messages into a single object
      before parsing

    Examples::

        # Simple JSON protocol
        agent = WebSocketAgent(url="ws://localhost:8000/ws/agent")

        # Raw bytes (e.g. streaming PCM)
        agent = WebSocketAgent(url="ws://localhost:8000/ws/stream", send_bytes=True)

        # Wait for setupComplete, then send a custom format
        async def wait_ready(session: WebSocketSession) -> None:
            await session.wait_for(lambda m: isinstance(m, dict) and m.get("setupComplete"))

        agent = WebSocketAgent(
            url="ws://localhost:8000/ws/agent",
            on_connect=wait_ready,
            on_send=lambda audio: json.dumps({"pcm": base64.b64encode(audio.data).decode()}),
            is_complete=lambda msgs: any(isinstance(m, dict) and m.get("done") for m in msgs),
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
        on_connect: Callable[[WebSocketSession], Awaitable[None]] | None = None,
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
        self.on_connect = on_connect
        self.on_send = on_send
        self.is_complete = is_complete
        self.aggregate = aggregate
        self.open_timeout = open_timeout
        self.close_timeout = close_timeout
        self.extra_ws_kwargs = extra_ws_kwargs or {}

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[WebSocketSession]:
        """Open a WebSocket connection and yield a :class:`WebSocketSession`.

        The connection is kept alive for the duration of the ``async with``
        block and closed cleanly on exit.  If an ``on_connect`` hook is
        configured it is invoked before the session is yielded, so the caller
        always receives a fully-initialised session.

        Example::

            async with agent.session() as sess:
                await sess.send_after(
                    lambda m: isinstance(m, dict) and m.get("ready"),
                    json_data={"audio": base64_audio},
                )
                result = await sess.receive(timeout=30.0)
        """
        import websockets

        async with websockets.connect(
            self.url,
            additional_headers=self.headers or None,
            open_timeout=self.open_timeout,
            close_timeout=self.close_timeout,
            **self.extra_ws_kwargs,
        ) as ws:
            sess = WebSocketSession(ws, self._parse_incoming)
            if self.on_connect:
                await self.on_connect(sess)
            yield sess

    async def run(self, audio: Audio) -> AgentResponse:
        """Send audio over WebSocket and collect the response."""
        async with self.session() as sess:
            message = self._prepare_message(audio)
            if isinstance(message, bytes):
                await sess.send_bytes(message)
            else:
                await sess.send_text(message)
            logger.debug(
                "Sent %s message (%d bytes)",
                "binary" if isinstance(message, bytes) else "text",
                len(message),
            )
            messages = await self._collect_from_session(sess)
            logger.debug("Collected %d response messages", len(messages))

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

    async def _collect_from_session(self, sess: WebSocketSession) -> list[Any]:
        """Read messages from *sess* until the completion condition or timeout."""
        messages: list[Any] = []
        try:
            async with asyncio.timeout(self.response_timeout):
                async for msg in sess:
                    messages.append(msg)

                    if self.is_complete and self.is_complete(messages):
                        break
                    if len(messages) >= self.max_messages:
                        logger.warning(
                            "Hit max_messages limit (%d), stopping collection",
                            self.max_messages,
                        )
                        break

                    # No is_complete hook → take only the first message
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
        """Try to JSON-parse an incoming message; fall back to the raw value."""
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
        """Fallback parser: expects ``{"tool_calls": [{"name": ..., "arguments": ...}]}``."""
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
