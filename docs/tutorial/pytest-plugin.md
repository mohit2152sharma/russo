# pytest Plugin

russo ships with a pytest plugin that's auto-discovered via the `pytest11` entry point. No configuration needed — just install russo and it's available.

## Markers

Use `@pytest.mark.russo` to declare test scenarios:

```python
import pytest
import russo

@pytest.mark.russo(
    prompt="Book a flight from NYC to LA",
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
async def test_book_flight(russo_result):
    russo.assert_tool_calls(russo_result)
```

### Marker Arguments

| Argument | Type | Description |
|---|---|---|
| `prompt` | `str` | Text prompt to synthesize |
| `expect` | `list[ToolCall]` | Expected tool calls |

## Fixtures

### Required Fixtures (You Provide)

You must define these in your `conftest.py`:

```python
# conftest.py
import pytest
from russo.synthesizers import GoogleSynthesizer
from russo.adapters import GeminiLiveAgent

@pytest.fixture(scope="session")
def russo_synthesizer():
    """The TTS synthesizer to use."""
    return GoogleSynthesizer(api_key="...")

@pytest.fixture(scope="session")
def russo_agent():
    """The agent under test."""
    return GeminiLiveAgent(api_key="...", tools=[...])
```

### Built-in Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `russo_result` | function | Runs the pipeline, returns `EvalResult` |
| `russo_evaluator` | function | Default `ExactEvaluator()` — override to customize |
| `russo_audio_cache` | session | `AudioCache` instance — override for custom dir |

### Override the Evaluator

```python
@pytest.fixture
def russo_evaluator():
    from russo.evaluators import ExactEvaluator
    return ExactEvaluator(ignore_extra_args=True, match_order=True)
```

## CLI Options

```bash
pytest --russo-report report.html    # Generate HTML report
pytest --russo-no-cache              # Disable audio caching
pytest --russo-clear-cache           # Clear cache before running
pytest --russo-cache-dir ./cache     # Custom cache directory
```

## Terminal Summary

After tests complete, russo prints a summary:

```
═══════════════════════ russo results ═══════════════════════
PASSED  test_book_flight (100% match rate)
FAILED  test_weather (0% match rate)
─────────────────────────────────────────────────────────────
Total: 2 | Passed: 1 | Failed: 1
```

## HTML Report

Use `--russo-report` to generate a standalone HTML report:

```bash
pytest --russo-report russo_report.html
```

## API Reference

See the [pytest plugin reference](../reference/pytest-plugin.md) for full API docs.
