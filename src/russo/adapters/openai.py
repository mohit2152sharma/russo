"""OpenAI SDK agent adapters — wraps OpenAI clients as Agents.

Provides two adapters:

- **OpenAIAgent**: standard Chat Completions with audio input
  (``gpt-4o-audio-preview`` and similar models)
- **OpenAIRealtimeAgent**: Realtime API over WebSocket
  (``gpt-4o-realtime-preview``)

Requires the ``openai`` package: ``pip install russo[openai]``
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import wave
from typing import Any

from russo._types import AgentResponse, Audio, ToolCall
from russo.parsers.openai import OpenAIResponseParser

logger = logging.getLogger("russo.adapters.openai")


class OpenAIAgent:
    """Agent adapter that wraps an ``AsyncOpenAI`` client for Chat Completions.

    Sends audio via the Chat Completions API and auto-parses tool-call
    responses with :class:`OpenAIResponseParser`.

    Usage::

        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        agent = OpenAIAgent(
            client=client,
            model="gpt-4o-audio-preview",
            tools=[{
                "type": "function",
                "function": {
                    "name": "book_flight",
                    "parameters": {"type": "object", "properties": {...}},
                },
            }],
        )
        response = await agent.run(audio)
    """

    def __init__(
        self,
        *,
        client: Any,
        model: str = "gpt-4o-audio-preview",
        tools: list[Any] | None = None,
        system_prompt: str | None = None,
        extra_create_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            client: An ``openai.AsyncOpenAI`` instance.
            model: Model name supporting audio input.
            tools: OpenAI tool definitions (function-calling format).
            system_prompt: Optional system message prepended to the conversation.
            extra_create_kwargs: Additional kwargs forwarded to
                ``client.chat.completions.create()``.
        """
        self.client = client
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.extra_create_kwargs = extra_create_kwargs or {}
        self._parser = OpenAIResponseParser()

    async def run(self, audio: Audio) -> AgentResponse:
        """Send audio via Chat Completions and parse the tool-call response."""
        audio_b64 = base64.b64encode(audio.data).decode("ascii")

        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": audio.format},
                    }
                ],
            }
        )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **self.extra_create_kwargs,
        }
        if self.tools:
            kwargs["tools"] = self.tools

        logger.debug("Sending %d bytes of %s audio to %s", len(audio.data), audio.format, self.model)

        response = await self.client.chat.completions.create(**kwargs)
        return self._parser.parse(response)


class OpenAIRealtimeAgent:
    """Agent adapter for OpenAI's Realtime API.

    Connects via the SDK's ``client.beta.realtime.connect()`` interface,
    sends audio, and collects ``response.function_call_arguments.done`` events.

    Accepts either:

    - An ``AsyncOpenAI`` **client** (creates a new connection per ``run()``)
    - A pre-existing realtime **connection** (reuses it, no session config sent)

    Usage with client::

        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        agent = OpenAIRealtimeAgent(
            client=client,
            model="gpt-4o-realtime-preview",
            tools=[{
                "type": "function",
                "name": "book_flight",
                "description": "Book a flight",
                "parameters": {"type": "object", "properties": {...}},
            }],
        )
        response = await agent.run(audio)

    Usage with pre-existing connection::

        async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as conn:
            agent = OpenAIRealtimeAgent(connection=conn)
            response = await agent.run(audio)

    Note:
        The Realtime API expects **pcm16** audio at 24 kHz mono by default.
        If you pass WAV audio, the adapter automatically strips the WAV header
        to extract raw PCM frames.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        connection: Any | None = None,
        model: str = "gpt-4o-realtime-preview",
        tools: list[Any] | None = None,
        response_timeout: float = 30.0,
    ) -> None:
        """
        Args:
            client: An ``openai.AsyncOpenAI`` instance. Mutually preferred with *connection*.
            connection: A pre-existing realtime connection (from ``client.beta.realtime.connect()``).
            model: Realtime model name.
            tools: Tool definitions sent during session configuration.
            response_timeout: Max seconds to wait for a complete response.
        """
        if client is None and connection is None:
            msg = "Provide either 'client' (AsyncOpenAI) or 'connection' (realtime connection)"
            raise ValueError(msg)
        self.client = client
        self.connection = connection
        self.model = model
        self.tools = tools
        self.response_timeout = response_timeout

    async def run(self, audio: Audio) -> AgentResponse:
        """Send audio via the Realtime API and collect function calls."""
        if self.connection is not None:
            return await self._run_on(self.connection, audio, configure=False)

        async with self.client.beta.realtime.connect(model=self.model) as conn:
            return await self._run_on(conn, audio, configure=True)

    async def _run_on(self, conn: Any, audio: Audio, *, configure: bool) -> AgentResponse:
        """Execute a single audio→tool-call round on an open connection."""
        if configure and self.tools:
            await conn.session.update(session={"tools": self.tools})

        pcm_data = _extract_pcm(audio)
        audio_b64 = base64.b64encode(pcm_data).decode("ascii")

        await conn.input_audio_buffer.append(audio=audio_b64)
        await conn.input_audio_buffer.commit()
        await conn.response.create()

        logger.debug("Sent %d bytes of audio to realtime session, waiting for response", len(pcm_data))

        tool_calls: list[ToolCall] = []
        raw_events: list[Any] = []

        try:
            async with asyncio.timeout(self.response_timeout):
                async for event in conn:
                    raw_events.append(event)
                    if event.type == "response.function_call_arguments.done":
                        args = json.loads(event.arguments) if event.arguments else {}
                        tool_calls.append(ToolCall(name=event.name, arguments=args))
                    elif event.type == "response.done":
                        break
        except TimeoutError:
            logger.warning(
                "Realtime response timed out after %.1fs with %d events",
                self.response_timeout,
                len(raw_events),
            )

        return AgentResponse(tool_calls=tool_calls, raw=raw_events or None)


def _extract_pcm(audio: Audio) -> bytes:
    """Extract raw PCM frames from audio, stripping WAV headers if present."""
    if audio.format == "wav" and len(audio.data) > 44 and audio.data[:4] == b"RIFF":
        with wave.open(io.BytesIO(audio.data), "rb") as wf:
            return wf.readframes(wf.getnframes())
    return audio.data
