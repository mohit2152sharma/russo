"""Gemini SDK agent adapters — wraps a google-genai Client as an Agent.

Provides two adapters:

- **GeminiAgent**: standard ``generate_content`` (request/response)
- **GeminiLiveAgent**: Live API over WebSocket (streaming/real-time)

Send audio directly to a Gemini model via the SDK, no HTTP endpoint needed.
Requires the ``google-genai`` package (already a core dependency).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from russo._types import AgentResponse, Audio, ToolCall
from russo.audio import AudioManager, Gemini
from russo.parsers.gemini import GeminiResponseParser

logger = logging.getLogger("russo.adapters.gemini")


def _make_client() -> Any:
    """Auto-create a ``genai.Client`` from environment variables.

    Resolution order:

    1. ``GOOGLE_API_KEY`` → Google AI API
    2. ``GOOGLE_CLOUD_PROJECT`` (or ``GOOGLE_PROJECT_ID``) **and** ``GOOGLE_CLOUD_LOCATION``
       → Vertex AI
    3. Plain ``genai.Client()`` — SDK's own resolution (ADC, etc.)
    """
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)

    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_PROJECT_ID")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    if project and location:
        # google-auth does not expand ~ in credential paths
        creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser(creds)
        return genai.Client(vertexai=True, project=project, location=location)

    return genai.Client()


# ---------------------------------------------------------------------------
# GeminiAgent
# ---------------------------------------------------------------------------
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
        client: Any | None = None,
        model: str = "gemini-2.0-flash",
        tools: list[Any] | None = None,
        system_instruction: str | None = None,
        config: Any | None = None,
    ) -> None:
        self.client = client if client is not None else _make_client()
        self.model = model
        self.tools = tools
        self.system_instruction = system_instruction
        self.config = config
        self._parser = GeminiResponseParser()

    async def run(self, audio: Audio) -> AgentResponse:
        """Send audio to Gemini and parse the tool-call response."""
        from google.genai import types

        data, mime_type = AudioManager.prepare_for_generate_content(audio)
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
        if self.config is not None:
            return self.config
        kwargs: dict[str, Any] = {}
        if self.tools:
            kwargs["tools"] = self.tools
        if self.system_instruction:
            kwargs["system_instruction"] = self.system_instruction
        return types.GenerateContentConfig(**kwargs) if kwargs else None


# ---------------------------------------------------------------------------
# GeminiLiveAgent
# ---------------------------------------------------------------------------
class GeminiLiveAgent:
    """Agent adapter for Gemini's Live API (streaming/real-time).

    Connects via ``client.aio.live.connect()``, sends audio via
    ``send_realtime_input``, and collects function-call responses.

    Accepts either a ``google.genai.Client`` (new session per run) or
    a pre-existing Live session.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        session: Any | None = None,
        model: str = "gemini-live-2.5-flash-native-audio",
        tools: list[Any] | None = None,
        system_instruction: str | None = None,
        config: Any | None = None,
        response_timeout: float = 30.0,
    ) -> None:
        if session is None and client is None:
            client = _make_client()
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

        assert self.client is not None
        config = self._build_config(types)
        async with self.client.aio.live.connect(model=self.model, config=config) as session:
            return await self._run_on(session, audio)

    async def _run_on(self, session: Any, audio: Audio) -> AgentResponse:
        """Execute a single audio→tool-call turn on an open session."""
        from google.genai import types

        pcm, mime_type = AudioManager.prepare_for_live(audio)

        await session.send_realtime_input(audio=types.Blob(data=pcm, mime_type=mime_type))
        await session.send_realtime_input(audio_stream_end=True)
        logger.debug("Sent %d bytes of %s audio to Live session", len(pcm), mime_type)

        tool_calls: list[ToolCall] = []
        raw_messages: list[Any] = []

        try:
            async with asyncio.timeout(self.response_timeout):
                async for msg in session.receive():
                    raw_messages.append(msg)
                    if msg.tool_call and msg.tool_call.function_calls:
                        for fc in msg.tool_call.function_calls:
                            tool_calls.append(
                                ToolCall(
                                    name=fc.name,
                                    arguments=dict(fc.args) if fc.args else {},
                                )
                            )
                        break
        except TimeoutError:
            logger.warning(
                "Live session timed out after %.1fs with %d messages",
                self.response_timeout,
                len(raw_messages),
            )

        return AgentResponse(tool_calls=tool_calls, raw=raw_messages or None)

    def _build_config(self, types: Any) -> Any:
        if self.config is not None:
            return self.config
        kwargs: dict[str, Any] = {
            "response_modalities": Gemini.LIVE_RESPONSE_MODALITIES,
        }
        if self.tools:
            kwargs["tools"] = self.tools
        if self.system_instruction:
            kwargs["system_instruction"] = self.system_instruction
        return types.LiveConnectConfig(**kwargs)
