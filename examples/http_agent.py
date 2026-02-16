"""HttpAgent example — test any HTTP endpoint for tool-call accuracy.

HttpAgent sends audio as base64-encoded JSON to your endpoint and parses
the response for tool calls. No SDK dependency needed on the server side.

Default protocol:
  POST {"audio": "<base64>", "format": "wav"}
  Response: {"tool_calls": [{"name": "...", "arguments": {...}}]}
"""

import asyncio

import russo
from russo.adapters import HttpAgent
from russo.evaluators import ExactEvaluator


# ---------------------------------------------------------------------------
# Example 1: Basic HTTP agent (default JSON protocol)
# ---------------------------------------------------------------------------
async def example_basic_http():
    """Point russo at a local HTTP endpoint."""
    agent = HttpAgent(
        url="http://localhost:8000/voice-agent",
    )

    # In a real scenario, you'd use a real synthesizer:
    #   from russo.synthesizers import GoogleSynthesizer
    #   synthesizer = GoogleSynthesizer(api_key="...")

    class FakeSynthesizer:
        async def synthesize(self, text: str) -> russo.Audio:
            return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)

    result = await russo.run(
        prompt="Book a flight from NYC to LA",
        synthesizer=FakeSynthesizer(),
        agent=agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
    )

    print(result.summary())


# ---------------------------------------------------------------------------
# Example 2: HTTP agent with custom headers and parser
# ---------------------------------------------------------------------------
async def example_http_with_options():
    """HttpAgent with auth headers and a custom response parser."""
    from russo.parsers import GeminiResponseParser

    agent = HttpAgent(
        url="https://my-api.example.com/v1/agent",
        headers={
            "Authorization": "Bearer my-token",
            "X-Request-ID": "russo-test-001",
        },
        parser=GeminiResponseParser(),  # parse Gemini-format responses
        timeout=30.0,
    )

    class FakeSynthesizer:
        async def synthesize(self, text: str) -> russo.Audio:
            return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)

    result = await russo.run(
        prompt="What's the weather in Berlin?",
        synthesizer=FakeSynthesizer(),
        agent=agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("get_weather", city="Berlin")],
    )

    print(result.summary())


# ---------------------------------------------------------------------------
# Example 3: HTTP agent with custom field names
# ---------------------------------------------------------------------------
async def example_http_custom_fields():
    """If your server expects different JSON field names."""
    agent = HttpAgent(
        url="http://localhost:8000/api/voice",
        audio_field="audio_data",    # server expects "audio_data" instead of "audio"
        format_field="audio_format",  # server expects "audio_format" instead of "format"
    )

    class FakeSynthesizer:
        async def synthesize(self, text: str) -> russo.Audio:
            return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)

    result = await russo.run(
        prompt="Search for flights to Tokyo",
        synthesizer=FakeSynthesizer(),
        agent=agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("search_flights", destination="Tokyo")],
    )

    print(result.summary())


if __name__ == "__main__":
    # These require a running server — pick the one matching your setup:
    asyncio.run(example_basic_http())
    # asyncio.run(example_http_with_options())
    # asyncio.run(example_http_custom_fields())
