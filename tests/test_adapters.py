"""Tests for Gemini adapters — GeminiAgent + GeminiLiveAgent, unit + integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from russo._protocols import Agent
from russo._types import AgentResponse, Audio
from russo.adapters.gemini import GeminiAgent, GeminiLiveAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_gemini_response(function_calls: list[dict[str, Any]]) -> MagicMock:
    """Build a mock mimicking a google-genai GenerateContentResponse."""
    parts = []
    for fc in function_calls:
        part = MagicMock()
        part.function_call = MagicMock()
        part.function_call.name = fc["name"]
        part.function_call.args = fc.get("args", {})
        parts.append(part)

    content = MagicMock()
    content.parts = parts
    candidate = MagicMock()
    candidate.content = content
    response = MagicMock()
    response.candidates = [candidate]
    return response


def _mock_client(response: MagicMock) -> MagicMock:
    """Return a MagicMock genai.Client whose aio.models.generate_content returns *response*."""
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------
class TestGeminiAgentProtocol:
    def test_satisfies_agent_protocol(self) -> None:
        agent = GeminiAgent(client=MagicMock())
        assert isinstance(agent, Agent)

    def test_has_run_method(self) -> None:
        agent = GeminiAgent(client=MagicMock())
        assert callable(agent.run)


# ---------------------------------------------------------------------------
# Init / config
# ---------------------------------------------------------------------------
class TestGeminiAgentInit:
    def test_defaults(self) -> None:
        client = MagicMock()
        agent = GeminiAgent(client=client)
        assert agent.client is client
        assert agent.model == "gemini-2.0-flash"
        assert agent.tools is None
        assert agent.system_instruction is None
        assert agent.config is None

    def test_custom_params(self) -> None:
        tools = [{"function_declarations": [{"name": "book_flight"}]}]
        agent = GeminiAgent(
            client=MagicMock(),
            model="gemini-2.5-pro",
            tools=tools,
            system_instruction="You are a travel agent.",
        )
        assert agent.model == "gemini-2.5-pro"
        assert agent.tools is tools
        assert agent.system_instruction == "You are a travel agent."

    def test_config_override(self) -> None:
        custom = MagicMock()
        agent = GeminiAgent(
            client=MagicMock(),
            tools=[MagicMock()],
            system_instruction="ignored",
            config=custom,
        )
        assert agent.config is custom


# ---------------------------------------------------------------------------
# Unit tests — run()
# ---------------------------------------------------------------------------
class TestGeminiAgentRun:
    @pytest.fixture
    def audio(self) -> Audio:
        return Audio(data=b"test-audio", format="wav", sample_rate=24000)

    async def test_single_tool_call(self, audio: Audio) -> None:
        resp = _make_gemini_response(
            [
                {"name": "book_flight", "args": {"from_city": "NYC", "to_city": "LA"}},
            ]
        )
        agent = GeminiAgent(client=_mock_client(resp), tools=[MagicMock()])
        result = await agent.run(audio)

        assert isinstance(result, AgentResponse)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "book_flight"
        assert result.tool_calls[0].arguments == {"from_city": "NYC", "to_city": "LA"}

    async def test_multiple_tool_calls(self, audio: Audio) -> None:
        resp = _make_gemini_response(
            [
                {"name": "book_flight", "args": {"from_city": "NYC", "to_city": "LA"}},
                {"name": "book_hotel", "args": {"city": "LA", "nights": 3}},
            ]
        )
        agent = GeminiAgent(client=_mock_client(resp))
        result = await agent.run(audio)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "book_flight"
        assert result.tool_calls[1].name == "book_hotel"
        assert result.tool_calls[1].arguments == {"city": "LA", "nights": 3}

    async def test_no_tool_calls(self, audio: Audio) -> None:
        resp = _make_gemini_response([])
        agent = GeminiAgent(client=_mock_client(resp))
        result = await agent.run(audio)

        assert result.tool_calls == []

    async def test_tool_call_empty_args(self, audio: Audio) -> None:
        resp = _make_gemini_response([{"name": "get_weather", "args": {}}])
        agent = GeminiAgent(client=_mock_client(resp))
        result = await agent.run(audio)

        assert result.tool_calls[0].arguments == {}

    async def test_raw_response_preserved(self, audio: Audio) -> None:
        resp = _make_gemini_response([{"name": "search", "args": {"q": "test"}}])
        agent = GeminiAgent(client=_mock_client(resp))
        result = await agent.run(audio)

        assert result.raw is resp

    async def test_custom_config_forwarded(self, audio: Audio) -> None:
        resp = _make_gemini_response([])
        client = _mock_client(resp)
        custom_config = MagicMock()

        agent = GeminiAgent(client=client, config=custom_config)
        await agent.run(audio)

        _, call_kwargs = client.aio.models.generate_content.call_args
        assert call_kwargs["config"] is custom_config

    async def test_model_forwarded(self, audio: Audio) -> None:
        resp = _make_gemini_response([])
        client = _mock_client(resp)

        agent = GeminiAgent(client=client, model="gemini-2.5-pro")
        await agent.run(audio)

        _, call_kwargs = client.aio.models.generate_content.call_args
        assert call_kwargs["model"] == "gemini-2.5-pro"

    async def test_tools_included_in_config(self, audio: Audio) -> None:
        resp = _make_gemini_response([])
        client = _mock_client(resp)
        tools = [{"function_declarations": [{"name": "f"}]}]

        agent = GeminiAgent(client=client, tools=tools)
        await agent.run(audio)

        _, call_kwargs = client.aio.models.generate_content.call_args
        config = call_kwargs["config"]
        # The config is a real GenerateContentConfig built from our tools kwarg
        assert config is not None

    async def test_system_instruction_included_in_config(self, audio: Audio) -> None:
        resp = _make_gemini_response([])
        client = _mock_client(resp)

        agent = GeminiAgent(client=client, system_instruction="be concise")
        await agent.run(audio)

        _, call_kwargs = client.aio.models.generate_content.call_args
        config = call_kwargs["config"]
        assert config is not None

    async def test_no_tools_no_instruction_sends_none_config(self, audio: Audio) -> None:
        resp = _make_gemini_response([])
        client = _mock_client(resp)

        agent = GeminiAgent(client=client)
        await agent.run(audio)

        _, call_kwargs = client.aio.models.generate_content.call_args
        assert call_kwargs["config"] is None

    async def test_audio_format_wav(self, audio: Audio) -> None:
        resp = _make_gemini_response([])
        client = _mock_client(resp)

        agent = GeminiAgent(client=client)
        await agent.run(audio)

        _, call_kwargs = client.aio.models.generate_content.call_args
        contents = call_kwargs["contents"]
        # Part.from_bytes was called — contents is a list with one Part
        assert len(contents) == 1

    async def test_audio_format_mp3(self) -> None:
        mp3_audio = Audio(data=b"mp3-data", format="mp3", sample_rate=24000)
        resp = _make_gemini_response([])
        client = _mock_client(resp)

        agent = GeminiAgent(client=client)
        await agent.run(mp3_audio)

        _, call_kwargs = client.aio.models.generate_content.call_args
        assert len(call_kwargs["contents"]) == 1

    async def test_api_error_propagates(self, audio: Audio) -> None:
        client = MagicMock()
        client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("quota exceeded"))

        agent = GeminiAgent(client=client)
        with pytest.raises(RuntimeError, match="quota exceeded"):
            await agent.run(audio)


# ---------------------------------------------------------------------------
# Integration tests (real Gemini API — skipped unless --integration)
# ---------------------------------------------------------------------------
BOOK_FLIGHT_TOOL = {
    "function_declarations": [
        {
            "name": "book_flight",
            "description": "Book a flight between two cities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_city": {"type": "string", "description": "Departure city"},
                    "to_city": {"type": "string", "description": "Arrival city"},
                },
                "required": ["from_city", "to_city"],
            },
        }
    ]
}

GET_WEATHER_TOOL = {
    "function_declarations": [
        {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        }
    ]
}


@pytest.mark.integration
class TestGeminiAgentIntegration:
    """Hit the real Gemini API.

    Run with: pytest tests/test_adapters.py --integration -v
    Requires: GOOGLE_API_KEY env var (or ADC for Vertex AI).
    """

    @pytest.fixture
    def agent(self, gemini_client: Any) -> GeminiAgent:
        return GeminiAgent(
            client=gemini_client,
            model="gemini-2.0-flash",
            tools=[BOOK_FLIGHT_TOOL],
            system_instruction=(
                "You are a travel assistant. When the user asks to book a flight, "
                "call the book_flight function with from_city and to_city."
            ),
        )

    async def test_audio_roundtrip_tool_call(self, agent: GeminiAgent, google_synth: Any) -> None:
        """Synthesize 'book a flight from NYC to LA' → send to Gemini → expect book_flight."""
        audio = await google_synth.synthesize("Book a flight from New York to Los Angeles")
        result = await agent.run(audio)

        assert len(result.tool_calls) >= 1, f"Expected tool calls, got: {result.raw}"
        call = result.tool_calls[0]
        assert call.name == "book_flight"
        assert "from_city" in call.arguments
        assert "to_city" in call.arguments

    async def test_no_tool_call_for_irrelevant_prompt(self, gemini_client: Any, google_synth: Any) -> None:
        """Audio unrelated to the tool should return zero tool calls (or text-only)."""
        agent = GeminiAgent(
            client=gemini_client,
            model="gemini-2.0-flash",
            tools=[BOOK_FLIGHT_TOOL],
            system_instruction=(
                "You are a travel assistant. ONLY call book_flight when the user "
                "explicitly asks to book a flight. For any other request, respond with text only."
            ),
        )
        audio = await google_synth.synthesize("What is the capital of France?")
        result = await agent.run(audio)

        assert result.tool_calls == [], f"Expected no tool calls, got: {result.tool_calls}"

    async def test_multiple_tools_selects_correct_one(self, gemini_client: Any, google_synth: Any) -> None:
        """With two tools declared, Gemini should pick the right one based on the prompt."""
        agent = GeminiAgent(
            client=gemini_client,
            model="gemini-2.0-flash",
            tools=[BOOK_FLIGHT_TOOL, GET_WEATHER_TOOL],
            system_instruction=("You are a helpful assistant. Call the appropriate tool based on the user's request."),
        )
        audio = await google_synth.synthesize("What's the weather in Tokyo?")
        result = await agent.run(audio)

        assert len(result.tool_calls) >= 1, f"Expected a tool call, got: {result.raw}"
        assert result.tool_calls[0].name == "get_weather"
        assert "city" in result.tool_calls[0].arguments

    async def test_raw_response_is_sdk_object(self, agent: GeminiAgent, google_synth: Any) -> None:
        """raw field should contain the actual SDK response object for debugging."""
        audio = await google_synth.synthesize("Book a flight from Chicago to Miami")
        result = await agent.run(audio)

        assert result.raw is not None
        # Should have candidates (it's a real genai response)
        assert hasattr(result.raw, "candidates")

    async def test_result_is_agent_response(self, agent: GeminiAgent, google_synth: Any) -> None:
        """Return type should always be AgentResponse regardless of content."""
        audio = await google_synth.synthesize("Hello, how are you today?")
        result = await agent.run(audio)

        assert isinstance(result, AgentResponse)

    async def test_tool_call_argument_values(self, gemini_client: Any, google_synth: Any) -> None:
        """Verify argument values are plausible, not just that keys exist."""
        agent = GeminiAgent(
            client=gemini_client,
            model="gemini-2.0-flash",
            tools=[GET_WEATHER_TOOL],
            system_instruction="When the user asks about weather, call get_weather with the city.",
        )
        audio = await google_synth.synthesize("Tell me the weather in London please")
        result = await agent.run(audio)

        assert len(result.tool_calls) >= 1
        city = result.tool_calls[0].arguments.get("city", "").lower()
        assert "london" in city, f"Expected 'london' in city arg, got: {city!r}"


# ---------------------------------------------------------------------------
# GeminiLiveAgent — helpers
# ---------------------------------------------------------------------------
def _make_live_tool_call_msg(function_calls: list[dict[str, Any]]) -> MagicMock:
    """Build a mock LiveServerMessage with tool_call."""
    fcs = []
    for fc in function_calls:
        mock_fc = MagicMock()
        mock_fc.name = fc["name"]
        mock_fc.args = fc.get("args", {})
        fcs.append(mock_fc)

    msg = MagicMock()
    msg.tool_call = MagicMock()
    msg.tool_call.function_calls = fcs
    msg.server_content = None
    return msg


def _make_live_turn_complete_msg() -> MagicMock:
    """Build a mock LiveServerMessage with turn_complete."""
    msg = MagicMock()
    msg.tool_call = None
    msg.server_content = MagicMock()
    msg.server_content.turn_complete = True
    return msg


def _make_live_text_msg() -> MagicMock:
    """Build a mock LiveServerMessage with text content (no tool call, not turn complete)."""
    msg = MagicMock()
    msg.tool_call = None
    msg.server_content = MagicMock()
    msg.server_content.turn_complete = False
    return msg


def _mock_live_session(messages: list[MagicMock]) -> MagicMock:
    """Build a mock live session whose receive() yields *messages*."""
    session = MagicMock()
    session.send_client_content = AsyncMock()

    async def _fake_receive():
        for m in messages:
            yield m

    session.receive = _fake_receive
    return session


# ---------------------------------------------------------------------------
# GeminiLiveAgent — protocol / init
# ---------------------------------------------------------------------------
class TestGeminiLiveAgentProtocol:
    def test_satisfies_agent_protocol(self) -> None:
        agent = GeminiLiveAgent(session=MagicMock())
        assert isinstance(agent, Agent)

    def test_requires_client_or_session(self) -> None:
        with pytest.raises(ValueError, match="Provide either"):
            GeminiLiveAgent()


class TestGeminiLiveAgentInit:
    def test_defaults(self) -> None:
        agent = GeminiLiveAgent(client=MagicMock())
        assert agent.model == "gemini-2.0-flash-live-preview-04-09"
        assert agent.tools is None
        assert agent.config is None
        assert agent.response_timeout == 30.0

    def test_with_session(self) -> None:
        sess = MagicMock()
        agent = GeminiLiveAgent(session=sess)
        assert agent.session is sess
        assert agent.client is None


# ---------------------------------------------------------------------------
# GeminiLiveAgent — unit tests
# ---------------------------------------------------------------------------
class TestGeminiLiveAgentRun:
    @pytest.fixture
    def audio(self) -> Audio:
        return Audio(data=b"test-audio", format="wav", sample_rate=24000)

    async def test_single_tool_call(self, audio: Audio) -> None:
        msg = _make_live_tool_call_msg(
            [
                {"name": "book_flight", "args": {"from_city": "NYC", "to_city": "LA"}},
            ]
        )
        session = _mock_live_session([msg])

        agent = GeminiLiveAgent(session=session)
        result = await agent.run(audio)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "book_flight"
        assert result.tool_calls[0].arguments == {"from_city": "NYC", "to_city": "LA"}

    async def test_multiple_tool_calls_in_one_message(self, audio: Audio) -> None:
        msg = _make_live_tool_call_msg(
            [
                {"name": "book_flight", "args": {"from_city": "NYC", "to_city": "LA"}},
                {"name": "book_hotel", "args": {"city": "LA"}},
            ]
        )
        session = _mock_live_session([msg])

        agent = GeminiLiveAgent(session=session)
        result = await agent.run(audio)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "book_flight"
        assert result.tool_calls[1].name == "book_hotel"

    async def test_no_tool_calls_turn_complete(self, audio: Audio) -> None:
        msg = _make_live_turn_complete_msg()
        session = _mock_live_session([msg])

        agent = GeminiLiveAgent(session=session)
        result = await agent.run(audio)

        assert result.tool_calls == []

    async def test_text_then_tool_call(self, audio: Audio) -> None:
        """Text messages before the tool call should be collected but not parsed as tools."""
        text_msg = _make_live_text_msg()
        tool_msg = _make_live_tool_call_msg([{"name": "search", "args": {"q": "test"}}])
        session = _mock_live_session([text_msg, tool_msg])

        agent = GeminiLiveAgent(session=session)
        result = await agent.run(audio)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        # raw should contain all messages received
        assert result.raw is not None and len(result.raw) == 2

    async def test_sends_audio_via_send_client_content(self, audio: Audio) -> None:
        msg = _make_live_turn_complete_msg()
        session = _mock_live_session([msg])

        agent = GeminiLiveAgent(session=session)
        await agent.run(audio)

        session.send_client_content.assert_awaited_once()
        call_kwargs = session.send_client_content.call_args.kwargs
        assert call_kwargs["turn_complete"] is True

    async def test_client_opens_session(self, audio: Audio) -> None:
        """When a client is passed (not a session), it should call aio.live.connect."""
        msg = _make_live_turn_complete_msg()
        session = _mock_live_session([msg])

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)

        client = MagicMock()
        client.aio.live.connect = MagicMock(return_value=ctx)

        agent = GeminiLiveAgent(client=client, tools=[MagicMock()])
        result = await agent.run(audio)

        client.aio.live.connect.assert_called_once()
        assert result.tool_calls == []

    async def test_custom_config_forwarded(self, audio: Audio) -> None:
        msg = _make_live_turn_complete_msg()
        session = _mock_live_session([msg])

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)

        client = MagicMock()
        client.aio.live.connect = MagicMock(return_value=ctx)

        custom_config = MagicMock()
        agent = GeminiLiveAgent(client=client, config=custom_config)
        await agent.run(audio)

        _, call_kwargs = client.aio.live.connect.call_args
        assert call_kwargs["config"] is custom_config


# ---------------------------------------------------------------------------
# GeminiLiveAgent — integration tests
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestGeminiLiveAgentIntegration:
    """Hit the real Gemini Live API via Vertex AI.

    Run with: pytest tests/test_adapters.py -k Live --integration -v
    Requires: ADC credentials + GOOGLE_CLOUD_PROJECT.
    """

    LIVE_MODEL = "gemini-2.0-flash-live-preview-04-09"

    @pytest.fixture
    def agent(self, gemini_client: Any) -> GeminiLiveAgent:
        return GeminiLiveAgent(
            client=gemini_client,
            model=self.LIVE_MODEL,
            tools=[BOOK_FLIGHT_TOOL],
            system_instruction=(
                "You are a travel assistant. When the user asks to book a flight, "
                "call the book_flight function with from_city and to_city."
            ),
        )

    async def test_audio_roundtrip_tool_call(self, agent: GeminiLiveAgent, google_synth: Any) -> None:
        """Synthesize → Live session → expect book_flight tool call."""
        audio = await google_synth.synthesize("Book a flight from New York to Los Angeles")
        result = await agent.run(audio)

        assert len(result.tool_calls) >= 1, f"Expected tool calls, got: {result.raw}"
        call = result.tool_calls[0]
        assert call.name == "book_flight"
        assert "from_city" in call.arguments
        assert "to_city" in call.arguments

    async def test_multiple_tools_selects_correct_one(self, gemini_client: Any, google_synth: Any) -> None:
        agent = GeminiLiveAgent(
            client=gemini_client,
            model=self.LIVE_MODEL,
            tools=[BOOK_FLIGHT_TOOL, GET_WEATHER_TOOL],
            system_instruction="Call the appropriate tool based on the user's request.",
        )
        audio = await google_synth.synthesize("What's the weather in Tokyo?")
        result = await agent.run(audio)

        assert len(result.tool_calls) >= 1, f"Expected a tool call, got: {result.raw}"
        assert result.tool_calls[0].name == "get_weather"
        assert "city" in result.tool_calls[0].arguments

    async def test_result_is_agent_response(self, agent: GeminiLiveAgent, google_synth: Any) -> None:
        audio = await google_synth.synthesize("Book a flight from London to Paris")
        result = await agent.run(audio)

        assert isinstance(result, AgentResponse)
