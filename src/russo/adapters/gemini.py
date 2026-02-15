"""Gemini SDK agent adapters — wraps a google-genai Client as an Agent.

Provides two adapters:

- **GeminiAgent**: standard ``generate_content`` (request/response)
- **GeminiLiveAgent**: Live API over WebSocket (streaming/real-time)

Send audio directly to a Gemini model via the SDK, no HTTP endpoint needed.
Requires the ``google-genai`` package (already a core dependency).
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from typing import Any

from russo._types import AgentResponse, Audio, ToolCall
from russo.parsers.gemini import GeminiResponseParser

logger = logging.getLogger("russo.adapters.gemini")


class GeminiAgent:
    """Agent adapter that wraps a ``google.genai.Client`` object directly.

    Sends audio to Gemini via ``client.aio.models.generate_content()``
    and auto-parses function-call responses with :class:`GeminiResponseParser`.

    Usage::

        from google import genai

        client = genai.Client(api_key="...")
        agent = GeminiAgent(
            client=client,
            model="gemini-2.0-flash",
            tools=[book_flight_declaration],
        )
        response = await agent.run(audio)

    For Vertex AI::

        client = genai.Client(vertexai=True, project="my-project", location="us-central1")
        agent = GeminiAgent(client=client, model="gemini-2.0-flash", tools=[...])
    """

    def __init__(
        self,
        *,
        client: Any,
        model: str = "gemini-2.0-flash",
        tools: list[Any] | None = None,
        system_instruction: str | None = None,
        config: Any | None = None,
    ) -> None:
        """
        Args:
            client: A ``google.genai.Client`` instance.
            model: Model name (e.g. ``"gemini-2.0-flash"``).
            tools: Gemini tool declarations (dicts or ``types.Tool`` objects).
            system_instruction: Optional system instruction prepended to the request.
            config: Full ``GenerateContentConfig`` override. When provided,
                    ``tools`` and ``system_instruction`` are ignored.
        """
        self.client = client
        self.model = model
        self.tools = tools
        self.system_instruction = system_instruction
        self.config = config
        self._parser = GeminiResponseParser()

    async def run(self, audio: Audio) -> AgentResponse:
        """Send audio to Gemini and parse the tool-call response."""
        from google.genai import types

        data, mime_type = _prepare_audio(audio)
        contents = [types.Part.from_bytes(data=data, mime_type=mime_type)]

        config = self._build_config(types)
        logger.debug("Sending %d bytes of %s audio to %s", len(data), mime_type, self.model)

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        return self._parser.parse(response)

    def _build_config(self, types: Any) -> Any:
        """Build GenerateContentConfig, merging user config with defaults."""
        if self.config is not None:
            return self.config

        kwargs: dict[str, Any] = {}
        if self.tools:
            kwargs["tools"] = self.tools
        if self.system_instruction:
            kwargs["system_instruction"] = self.system_instruction

        return types.GenerateContentConfig(**kwargs) if kwargs else None


def _prepare_audio(audio: Audio) -> tuple[bytes, str]:
    """Return (bytes, mime_type) ready for the Gemini API.

    If the Audio is marked as ``wav`` but the raw bytes lack a RIFF header
    (i.e. raw PCM from GoogleSynthesizer), we wrap them in a proper WAV
    container so the API accepts them.
    """
    if audio.format == "wav":
        if not (len(audio.data) >= 4 and audio.data[:4] == b"RIFF"):
            # Raw PCM — wrap in WAV container
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(audio.channels)
                wf.setsampwidth(audio.sample_width)
                wf.setframerate(audio.sample_rate)
                wf.writeframes(audio.data)
            return buf.getvalue(), "audio/wav"
        return audio.data, "audio/wav"

    mime_map = {
        "mp3": "audio/mp3",
        "pcm": "audio/l16",
        "ogg": "audio/ogg",
    }
    return audio.data, mime_map.get(audio.format, f"audio/{audio.format}")


def _extract_pcm(audio: Audio) -> bytes:
    """Extract raw PCM frames, stripping WAV headers if present."""
    if audio.format == "wav" and len(audio.data) > 44 and audio.data[:4] == b"RIFF":
        with wave.open(io.BytesIO(audio.data), "rb") as wf:
            return wf.readframes(wf.getnframes())
    return audio.data


class GeminiLiveAgent:
    """Agent adapter for Gemini's Live API (streaming/real-time).

    Connects via ``client.aio.live.connect()``, sends audio as a turn-based
    content message, and collects function-call responses.

    Accepts either:

    - A ``google.genai.Client`` (opens a new session per ``run()``)
    - A pre-existing Live **session** (reuses it)

    Usage with client::

        from google import genai

        client = genai.Client(vertexai=True, project="my-project", location="us-central1")
        agent = GeminiLiveAgent(
            client=client,
            model="gemini-2.0-flash-live-preview-04-09",
            tools=[book_flight_declaration],
        )
        response = await agent.run(audio)

    Usage with pre-existing session::

        config = types.LiveConnectConfig(tools=[...], response_modalities=["TEXT"])
        async with client.aio.live.connect(model="...", config=config) as session:
            agent = GeminiLiveAgent(session=session)
            response = await agent.run(audio)

    Note:
        Model names differ by backend:

        - **Vertex AI**: ``gemini-2.0-flash-live-preview-04-09``
        - **Google AI**: ``gemini-live-2.5-flash-preview``
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        session: Any | None = None,
        model: str = "gemini-2.0-flash-live-preview-04-09",
        tools: list[Any] | None = None,
        system_instruction: str | None = None,
        config: Any | None = None,
        response_timeout: float = 30.0,
    ) -> None:
        """
        Args:
            client: A ``google.genai.Client`` instance. Creates a new session per ``run()``.
            session: A pre-existing Live session (from ``client.aio.live.connect()``).
                When provided, ``tools``/``system_instruction``/``config`` are ignored
                (the session was already configured at connect time).
            model: Live model name.
            tools: Gemini tool declarations.
            system_instruction: System instruction for the session.
            config: Full ``LiveConnectConfig`` override.  When provided,
                ``tools``, ``system_instruction``, and ``response_modalities`` are ignored.
            response_timeout: Max seconds to wait for a complete response turn.
        """
        if client is None and session is None:
            msg = "Provide either 'client' (genai.Client) or 'session' (live session)"
            raise ValueError(msg)
        self.client = client
        self.session = session
        self.model = model
        self.tools = tools
        self.system_instruction = system_instruction
        self.config = config
        self.response_timeout = response_timeout

    async def run(self, audio: Audio) -> AgentResponse:
        """Send audio to a Live session and collect function calls."""
        if self.session is not None:
            return await self._run_on(self.session, audio)

        from google.genai import types

        assert self.client is not None  # guaranteed by __init__
        config = self._build_config(types)
        async with self.client.aio.live.connect(model=self.model, config=config) as session:
            return await self._run_on(session, audio)

    async def _run_on(self, session: Any, audio: Audio) -> AgentResponse:
        """Execute a single audio→tool-call turn on an open session."""
        from google.genai import types

        data, mime_type = _prepare_audio(audio)

        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part.from_bytes(data=data, mime_type=mime_type)],
            ),
            turn_complete=True,
        )
        logger.debug("Sent %d bytes of %s audio to Live session", len(data), mime_type)

        tool_calls: list[ToolCall] = []
        raw_messages: list[Any] = []

        try:
            async with asyncio.timeout(self.response_timeout):
                async for msg in session.receive():
                    raw_messages.append(msg)
                    if msg.tool_call and msg.tool_call.function_calls:
                        for fc in msg.tool_call.function_calls:
                            name = fc.name
                            args = dict(fc.args) if fc.args else {}
                            tool_calls.append(ToolCall(name=name, arguments=args))
                        break  # model is now waiting for tool_response — we're done
                    # receive() already breaks on turn_complete
        except TimeoutError:
            logger.warning(
                "Live session timed out after %.1fs with %d messages",
                self.response_timeout,
                len(raw_messages),
            )

        return AgentResponse(tool_calls=tool_calls, raw=raw_messages or None)

    def _build_config(self, types: Any) -> Any:
        """Build LiveConnectConfig, merging user config with defaults."""
        if self.config is not None:
            return self.config

        kwargs: dict[str, Any] = {
            # Default to TEXT for function-calling — AUDIO response is not useful for tool-call extraction
            "response_modalities": ["TEXT"],
        }
        if self.tools:
            kwargs["tools"] = self.tools
        if self.system_instruction:
            kwargs["system_instruction"] = self.system_instruction

        return types.LiveConnectConfig(**kwargs)
