# pytest Integration

The most natural way to use russo -- declarative test scenarios using markers and fixtures.

!!! tip "Source files"
    [`examples/pytest_integration/`](https://github.com/mohit2152sharma/russo/tree/main/examples/pytest_integration)

## How it works

The russo pytest plugin (auto-discovered via the `pytest11` entry point) provides:

- **`@pytest.mark.russo`** -- marker to declare test scenarios
- **`russo_result`** -- fixture that runs the full pipeline and returns an `EvalResult`
- **Overridable fixtures** -- `russo_synthesizer`, `russo_agent`, `russo_evaluator`
- **CLI options** -- caching, reporting, and more

## Step 1: Configure fixtures in `conftest.py`

Define your synthesizer and agent as pytest fixtures:

```python
# conftest.py
import os
import pytest
import russo
from russo.evaluators import ExactEvaluator


@pytest.fixture(scope="session")
def russo_synthesizer():
    """TTS synthesizer for all russo tests."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        from russo.synthesizers import GoogleSynthesizer
        return GoogleSynthesizer(api_key=api_key)

    # Fallback for CI / offline use
    class FakeSynthesizer:
        async def synthesize(self, text: str) -> russo.Audio:
            return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)

    return FakeSynthesizer()


@pytest.fixture(scope="session")
def russo_agent():
    """The agent under test."""
    # Replace with your real agent:
    #   from russo.adapters import GeminiLiveAgent
    #   return GeminiLiveAgent(client=..., model="...", tools=[...])

    @russo.agent
    async def fake_agent(audio: russo.Audio) -> russo.AgentResponse:
        return russo.AgentResponse(
            tool_calls=[
                russo.ToolCall(
                    name="book_flight",
                    arguments={"from_city": "NYC", "to_city": "LA"},
                ),
            ]
        )

    return fake_agent


@pytest.fixture
def russo_evaluator():
    """Override to use a custom evaluator (defaults to ExactEvaluator)."""
    return ExactEvaluator()
```

## Step 2: Write tests using the marker

The `@pytest.mark.russo` marker declares the prompt and expected tool calls. The `russo_result` fixture runs the full pipeline automatically.

### Basic assertion

```python
# test_flights.py
import pytest
import russo


@pytest.mark.russo(
    prompt="Book a flight from NYC to LA",
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
async def test_book_flight(russo_result):
    """Verify the agent calls book_flight with the right arguments."""
    russo.assert_tool_calls(russo_result)
```

### Match rate check

```python
@pytest.mark.russo(
    prompt="Book a flight from NYC to LA",
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
async def test_match_rate(russo_result):
    """Check that every expected tool call was matched."""
    assert russo_result.passed
    assert russo_result.match_rate == 1.0
```

### Custom failure message

```python
@pytest.mark.russo(
    prompt="Book a flight from NYC to LA",
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
async def test_with_custom_message(russo_result):
    russo.assert_tool_calls(russo_result, message="Flight booking agent failed")
```

### Detailed inspection

```python
@pytest.mark.russo(
    prompt="Book a flight from NYC to LA",
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
async def test_detailed_inspection(russo_result):
    """Access individual match details for richer assertions."""
    assert russo_result.passed

    for match in russo_result.matches:
        assert match.matched, f"Expected {match.expected.name} was not matched"
        assert match.actual is not None
        assert match.actual.name == match.expected.name

    assert len(russo_result.actual) >= 1
    assert russo_result.actual[0].name == "book_flight"
```

## Step 3: Run the tests

```bash
pytest examples/pytest_integration/ -v
```

Expected output:

```
test_flights.py::test_book_flight PASSED
test_flights.py::test_match_rate PASSED
test_flights.py::test_with_custom_message PASSED
test_flights.py::test_detailed_inspection PASSED
```

## CLI options

The russo plugin adds several command-line options:

```bash
# Caching
pytest --russo-cache                # enable audio cache (default)
pytest --russo-no-cache             # disable caching
pytest --russo-clear-cache          # clear cache before running
pytest --russo-cache-dir .my_cache  # custom cache directory

# Reporting
pytest --russo-report report.html   # generate HTML report
```
