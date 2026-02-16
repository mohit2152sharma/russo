"""Example tests using concurrent runs via the russo pytest plugin.

Demonstrates three scenarios:
  1. Single prompt, multiple runs  (runs=N)
  2. Multiple prompts, one run each  (prompts=[...])
  3. Combined: multiple prompts × multiple runs

Run with:
    pytest examples/pytest_integration/ -v

Override run count from the CLI:
    pytest examples/pytest_integration/ --russo-runs 5

Limit concurrency:
    pytest examples/pytest_integration/ --russo-max-concurrency 2
"""

import pytest

import russo


# ---------------------------------------------------------------------------
# Scenario 1: Single prompt, multiple runs
# ---------------------------------------------------------------------------
@pytest.mark.russo(
    prompt="Book a flight from NYC to LA",
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
    runs=3,
)
async def test_single_prompt_multiple_runs(russo_result):
    """Run the same prompt 3 times and check that all pass."""
    assert isinstance(russo_result, russo.BatchResult)
    assert russo_result.total == 3
    assert russo_result.pass_rate >= 0.5, f"At least half should pass:\n{russo_result.summary()}"


# ---------------------------------------------------------------------------
# Scenario 2: Multiple prompts, single run each
# ---------------------------------------------------------------------------
@pytest.mark.russo(
    prompts=[
        "Book a flight from NYC to LA",
        "I need to fly from NYC to LA",
    ],
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
async def test_multiple_prompts(russo_result):
    """Two different phrasings of the same intent — both should trigger book_flight."""
    assert isinstance(russo_result, russo.BatchResult)
    assert russo_result.total == 2
    assert russo_result.passed


# ---------------------------------------------------------------------------
# Scenario 3: Multiple prompts × multiple runs
# ---------------------------------------------------------------------------
@pytest.mark.russo(
    prompts=[
        "Book a flight from NYC to LA",
        "I need to fly from NYC to LA",
    ],
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
    runs=2,
    max_concurrency=3,
)
async def test_prompts_times_runs(russo_result):
    """Full matrix: 2 prompts × 2 runs = 4 pipeline executions."""
    assert isinstance(russo_result, russo.BatchResult)
    assert russo_result.total == 4
    assert russo_result.pass_rate >= 0.5


# ---------------------------------------------------------------------------
# Scenario 4: runs=1, single prompt (backward-compatible)
# ---------------------------------------------------------------------------
@pytest.mark.russo(
    prompt="Book a flight from NYC to LA",
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
async def test_single_run_still_returns_eval_result(russo_result):
    """Default (runs=1, single prompt) still returns EvalResult for backward compat."""
    assert isinstance(russo_result, russo.EvalResult)
    assert russo_result.passed
