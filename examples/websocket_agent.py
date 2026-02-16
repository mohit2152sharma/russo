"""WebSocketAgent example — test a WebSocket endpoint for tool-call accuracy.

Supports JSON mode, raw bytes mode, and fully custom protocols via hooks.

Requires: pip install "russo[ws]"
"""

import asyncio
import base64
import json

import russo
from russo.adapters import WebSocketAgent
from russo.evaluators import ExactEvaluator


# ---------------------------------------------------------------------------
# Example 1: JSON protocol (default)
# ---------------------------------------------------------------------------
async def example_json_mode():
    """Send audio as base64 JSON, receive JSON response."""
    agent = WebSocketAgent(
        url="ws://localhost:8000/ws/agent",
        response_timeout=15.0,
    )

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

    print("JSON mode:", result.summary())


# ---------------------------------------------------------------------------
# Example 2: Raw bytes mode (streaming PCM)
# ---------------------------------------------------------------------------
async def example_bytes_mode():
    """Send raw audio bytes directly on the wire."""
    agent = WebSocketAgent(
        url="ws://localhost:8000/ws/stream",
        send_bytes=True,
        response_timeout=10.0,
    )

    class FakeSynthesizer:
        async def synthesize(self, text: str) -> russo.Audio:
            return russo.Audio(data=b"\x00" * 4800, format="pcm", sample_rate=24000)

    result = await russo.run(
        prompt="What's the weather?",
        synthesizer=FakeSynthesizer(),
        agent=agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("get_weather", city="Berlin")],
    )

    print("Bytes mode:", result.summary())


# ---------------------------------------------------------------------------
# Example 3: Custom protocol with hooks
# ---------------------------------------------------------------------------
async def example_custom_protocol():
    """Full control over send/receive with on_send, is_complete, and aggregate hooks."""

    def custom_send(audio: russo.Audio) -> str:
        """Build a custom JSON message for your server."""
        return json.dumps({
            "type": "audio_input",
            "pcm": base64.b64encode(audio.data).decode(),
            "sample_rate": audio.sample_rate,
        })

    def is_done(messages: list) -> bool:
        """Stop collecting when the server sends a 'done' message."""
        return any(
            isinstance(m, dict) and m.get("type") == "done"
            for m in messages
        )

    def aggregate_responses(messages: list):
        """Combine all tool_call messages into one response."""
        tool_calls = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("type") == "tool_call":
                tool_calls.append(msg)
        return {"tool_calls": tool_calls}

    agent = WebSocketAgent(
        url="ws://localhost:8000/ws/custom",
        on_send=custom_send,
        is_complete=is_done,
        aggregate=aggregate_responses,
        response_timeout=20.0,
        max_messages=50,
    )

    class FakeSynthesizer:
        async def synthesize(self, text: str) -> russo.Audio:
            return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)

    result = await russo.run(
        prompt="Search for hotels in Paris",
        synthesizer=FakeSynthesizer(),
        agent=agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("search_hotels", city="Paris")],
    )

    print("Custom protocol:", result.summary())


# ---------------------------------------------------------------------------
# Example 4: With a response parser
# ---------------------------------------------------------------------------
async def example_with_parser():
    """Use a built-in response parser for Gemini-format responses."""
    from russo.parsers import GeminiResponseParser

    agent = WebSocketAgent(
        url="ws://localhost:8000/ws/gemini",
        parser=GeminiResponseParser(),
        headers={"Authorization": "Bearer my-token"},
    )

    class FakeSynthesizer:
        async def synthesize(self, text: str) -> russo.Audio:
            return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)

    result = await russo.run(
        prompt="Get directions to the airport",
        synthesizer=FakeSynthesizer(),
        agent=agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("get_directions", destination="airport")],
    )

    print("With parser:", result.summary())


if __name__ == "__main__":
    # These require a running WebSocket server:
    asyncio.run(example_json_mode())
    # asyncio.run(example_bytes_mode())
    # asyncio.run(example_custom_protocol())
    # asyncio.run(example_with_parser())
