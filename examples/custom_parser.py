"""JsonResponseParser example — parse custom HTTP/WebSocket response structures.

JsonResponseParser lets you describe your endpoint's response shape via
constructor args instead of writing a full parser class.

Works with both HttpAgent and WebSocketAgent via the parser= argument.
"""

import asyncio

import russo
from russo.adapters import HttpAgent, WebSocketAgent
from russo.evaluators import ExactEvaluator
from russo.parsers import JsonResponseParser


# ---------------------------------------------------------------------------
# Example 1: Custom top-level key (single list)
# ---------------------------------------------------------------------------
# Endpoint returns:
#   {"toolCall": [{"name": "book_flight", "arguments": {"from_city": "NYC", "to_city": "LA"}}]}
async def example_custom_key():
    """Endpoint uses 'toolCall' instead of the default 'tool_calls'."""
    agent = HttpAgent(
        url="http://localhost:8000/agent",
        parser=JsonResponseParser(tool_calls_key="toolCall"),
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

    print("Custom key:", result.summary())


# ---------------------------------------------------------------------------
# Example 2: Single tool call object (not a list)
# ---------------------------------------------------------------------------
# Endpoint returns:
#   {"toolCall": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}
async def example_single_call():
    """Endpoint returns one tool call object, not wrapped in a list."""
    agent = HttpAgent(
        url="http://localhost:8000/agent",
        parser=JsonResponseParser(tool_calls_key="toolCall", single=True),
    )

    class FakeSynthesizer:
        async def synthesize(self, text: str) -> russo.Audio:
            return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)

    result = await russo.run(
        prompt="What's the weather in Tokyo?",
        synthesizer=FakeSynthesizer(),
        agent=agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("get_weather", city="Tokyo")],
    )

    print("Single call:", result.summary())


# ---------------------------------------------------------------------------
# Example 3: Custom field names for name and arguments
# ---------------------------------------------------------------------------
# Endpoint returns:
#   {"calls": [{"function": "search", "params": {"query": "hello"}}]}
async def example_custom_field_names():
    """Endpoint uses non-standard field names for function name and arguments."""
    agent = HttpAgent(
        url="http://localhost:8000/agent",
        parser=JsonResponseParser(
            tool_calls_key="calls",
            name_key="function",
            arguments_key="params",
        ),
    )

    class FakeSynthesizer:
        async def synthesize(self, text: str) -> russo.Audio:
            return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)

    result = await russo.run(
        prompt="Search for Python tutorials",
        synthesizer=FakeSynthesizer(),
        agent=agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("search", query="Python tutorials")],
    )

    print("Custom field names:", result.summary())


# ---------------------------------------------------------------------------
# Example 4: Nested key path (dot notation)
# ---------------------------------------------------------------------------
# Endpoint returns:
#   {"response": {"data": {"tool_calls": [{"name": "book_hotel", "arguments": {...}}]}}}
async def example_nested_path():
    """Endpoint wraps tool calls under a nested response envelope."""
    agent = HttpAgent(
        url="http://localhost:8000/agent",
        parser=JsonResponseParser(tool_calls_key="response.data.tool_calls"),
    )

    class FakeSynthesizer:
        async def synthesize(self, text: str) -> russo.Audio:
            return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)

    result = await russo.run(
        prompt="Book a hotel in Paris for 3 nights",
        synthesizer=FakeSynthesizer(),
        agent=agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("book_hotel", city="Paris", nights=3)],
    )

    print("Nested path:", result.summary())


# ---------------------------------------------------------------------------
# Example 5: WebSocket endpoint with custom response structure
# ---------------------------------------------------------------------------
# Server sends multiple messages; parser scans each one for the first match.
# Message containing tool call:
#   {"status": "done", "toolCall": [{"name": "book_flight", "arguments": {...}}]}
async def example_websocket_custom_parser():
    """WebSocket endpoint with a custom response structure."""
    agent = WebSocketAgent(
        url="ws://localhost:8000/ws/agent",
        parser=JsonResponseParser(tool_calls_key="toolCall"),
        is_complete=lambda msgs: any(isinstance(m, dict) and m.get("status") == "done" for m in msgs),
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

    print("WebSocket custom parser:", result.summary())


if __name__ == "__main__":
    # These require a running server — pick the one matching your setup:
    asyncio.run(example_custom_key())
    # asyncio.run(example_single_call())
    # asyncio.run(example_custom_field_names())
    # asyncio.run(example_nested_path())
    # asyncio.run(example_websocket_custom_parser())
