"""End-to-end WebSocket agent tests using russo's pytest plugin.

Tests a FastAPI server that wraps Gemini with travel-related tools.
Requires Google AI or Vertex AI credentials in the environment.

Run:
    pytest examples/websocket_testing/ -v
"""

import pytest

import russo


# ---------------------------------------------------------------------------
# Single-run tests
# ---------------------------------------------------------------------------
@pytest.mark.russo(
    prompt="Book a flight from Berlin to Rome",
    expect=[russo.tool_call("book_flight", from_city="Berlin", to_city="Rome")],
)
async def test_book_flight(russo_result):
    """Verify the agent calls book_flight with correct cities."""
    russo.assert_tool_calls(russo_result)


@pytest.mark.russo(
    prompt="Search for hotels in Paris",
    expect=[russo.tool_call("search_hotels", city="Paris")],
)
async def test_search_hotels(russo_result):
    """Verify the agent calls search_hotels for the right city."""
    russo.assert_tool_calls(russo_result)


@pytest.mark.russo(
    prompt="What is the weather in Tokyo",
    expect=[russo.tool_call("get_weather", city="Tokyo")],
)
async def test_get_weather(russo_result):
    """Verify the agent calls get_weather for the right city."""
    russo.assert_tool_calls(russo_result)


# ---------------------------------------------------------------------------
# Reliability test (multiple runs)
# ---------------------------------------------------------------------------
@pytest.mark.russo(
    prompt="Book a flight from Berlin to Rome",
    expect=[russo.tool_call("book_flight", from_city="Berlin", to_city="Rome")],
    runs=3,
)
async def test_flight_reliability(russo_result):
    """Run the same prompt 3 times and check pass rate."""
    assert isinstance(russo_result, russo.BatchResult)
    assert russo_result.pass_rate >= 0.66


# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------
@pytest.mark.russo(
    prompts=[
        "I need to fly from Berlin to Rome",
        "Please book me a flight, departing Berlin, arriving Rome",
        "Get me on a plane from Berlin to Rome",
    ],
    expect=[russo.tool_call("book_flight", from_city="Berlin", to_city="Rome")],
)
async def test_prompt_variants(russo_result):
    """Multiple phrasings should all trigger the same tool call."""
    assert isinstance(russo_result, russo.BatchResult)
    assert russo_result.pass_rate >= 0.66
