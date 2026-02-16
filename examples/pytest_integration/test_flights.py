"""Example test file using the russo pytest plugin.

Uses @pytest.mark.russo to declaratively define test scenarios.
The plugin handles synthesizing audio, running the agent, and evaluating.

Run with:
    pytest examples/pytest_integration/ -v

With caching options:
    pytest --russo-cache                # enable audio cache (default)
    pytest --russo-no-cache             # disable caching
    pytest --russo-clear-cache          # clear cache before running
    pytest --russo-cache-dir .my_cache  # custom cache directory

Generate an HTML report:
    pytest --russo-report report.html
"""

import pytest

import russo


# ---------------------------------------------------------------------------
# Basic test using the marker
# ---------------------------------------------------------------------------
@pytest.mark.russo(
    prompt="Book a flight from NYC to LA",
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
async def test_book_flight(russo_result):
    """Verify the agent calls book_flight with the right arguments."""
    russo.assert_tool_calls(russo_result)


# ---------------------------------------------------------------------------
# Test with match_rate assertion
# ---------------------------------------------------------------------------
@pytest.mark.russo(
    prompt="Book a flight from NYC to LA",
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
async def test_match_rate(russo_result):
    """Check that every expected tool call was matched."""
    assert russo_result.passed
    assert russo_result.match_rate == 1.0


# ---------------------------------------------------------------------------
# Test with custom failure message
# ---------------------------------------------------------------------------
@pytest.mark.russo(
    prompt="Book a flight from NYC to LA",
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
async def test_with_custom_message(russo_result):
    """assert_tool_calls supports a custom error message for failures."""
    russo.assert_tool_calls(russo_result, message="Flight booking agent failed")


# ---------------------------------------------------------------------------
# Test that inspects the result in detail
# ---------------------------------------------------------------------------
@pytest.mark.russo(
    prompt="Book a flight from NYC to LA",
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
async def test_detailed_inspection(russo_result):
    """Access individual match details for richer assertions."""
    assert russo_result.passed

    # Inspect individual matches
    for match in russo_result.matches:
        assert match.matched, f"Expected {match.expected.name} was not matched"
        assert match.actual is not None
        assert match.actual.name == match.expected.name

    # Check the actual tool calls
    assert len(russo_result.actual) >= 1
    assert russo_result.actual[0].name == "book_flight"
