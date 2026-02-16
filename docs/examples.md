# Examples

Self-contained, runnable examples for every major russo feature. Browse the source in the [`examples/`](https://github.com/mohit2152sharma/russo/tree/main/examples) directory.

## Quick Reference

| Example | What it shows |
|---------|---------------|
| [`basic_pipeline.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/basic_pipeline.py) | Minimal `russo.run()` end-to-end pipeline |
| [`custom_agent.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/custom_agent.py) | Wrap any async function with `@russo.agent` |
| [`custom_evaluator.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/custom_evaluator.py) | Build a custom evaluator (structural subtyping, no inheritance) |
| [`custom_synthesizer.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/custom_synthesizer.py) | Build a file-based or silence synthesizer for offline testing |
| [`gemini_adapter.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/gemini_adapter.py) | `GeminiAgent` + `GeminiLiveAgent` + Vertex AI |
| [`openai_adapter.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/openai_adapter.py) | `OpenAIAgent` + `OpenAIRealtimeAgent` |
| [`http_agent.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/http_agent.py) | Test HTTP endpoints with `HttpAgent` |
| [`websocket_agent.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/websocket_agent.py) | Test WebSocket endpoints with custom hooks |
| [`caching.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/caching.py) | `CachedSynthesizer` + `AudioCache` |
| [`pytest_integration/`](https://github.com/mohit2152sharma/russo/tree/main/examples/pytest_integration) | pytest markers, fixtures, and CLI options |
| [`config_driven/`](https://github.com/mohit2152sharma/russo/tree/main/examples/config_driven) | YAML-driven pipeline via CLI or programmatic loader |

---

## Basics

### Minimal Pipeline

The simplest way to use russo -- synthesize audio, send it to an agent, and evaluate tool calls:

```python
import asyncio
import russo
from russo.evaluators import ExactEvaluator

@russo.agent
async def my_agent(audio: russo.Audio) -> russo.AgentResponse:
    return russo.AgentResponse(
        tool_calls=[
            russo.ToolCall(name="book_flight", arguments={"from_city": "Berlin", "to_city": "Rome"})
        ]
    )

class FakeSynthesizer:
    async def synthesize(self, text: str) -> russo.Audio:
        return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)

async def main():
    result = await russo.run(
        prompt="Book a flight from Berlin to Rome",
        synthesizer=FakeSynthesizer(),
        agent=my_agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("book_flight", from_city="Berlin", to_city="Rome")],
    )
    print(result.summary())
    russo.assert_tool_calls(result)

asyncio.run(main())
```

### pytest Integration

The most natural way to use russo with your test suite:

=== "conftest.py"

    ```python
    import pytest
    import russo
    from russo.synthesizers import GoogleSynthesizer

    @pytest.fixture(scope="session")
    def russo_synthesizer():
        return GoogleSynthesizer(api_key="...")

    @pytest.fixture(scope="session")
    def russo_agent():
        @russo.agent
        async def agent(audio: russo.Audio) -> russo.AgentResponse:
            # call your real agent here
            ...
        return agent
    ```

=== "test_flights.py"

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

---

## Adapters

### Gemini

```python
from google import genai
from russo.adapters import GeminiLiveAgent

client = genai.Client(api_key="...")
agent = GeminiLiveAgent(
    client=client,
    model="gemini-live-2.5-flash-preview",
    tools=[BOOK_FLIGHT_TOOL],
    system_instruction="You are a travel assistant...",
)
```

### OpenAI

```python
from openai import AsyncOpenAI
from russo.adapters import OpenAIAgent

client = AsyncOpenAI()
agent = OpenAIAgent(
    client=client,
    model="gpt-4o-audio-preview",
    tools=[BOOK_FLIGHT_TOOL],
)
```

### HTTP / WebSocket

```python
from russo.adapters import HttpAgent, WebSocketAgent

http_agent = HttpAgent(url="http://localhost:8000/voice-agent")
ws_agent = WebSocketAgent(url="ws://localhost:8000/ws/agent")
```

---

## Extensibility

### Custom Evaluator

Just implement the right method signature -- no inheritance needed:

```python
class FuzzyEvaluator:
    def evaluate(self, expected, actual):
        # your matching logic here
        ...
```

### Custom Synthesizer

```python
class FileSynthesizer:
    async def synthesize(self, text: str) -> russo.Audio:
        data = Path(f"audio/{hash(text)}.wav").read_bytes()
        return russo.Audio(data=data, format="wav", sample_rate=24000)
```

---

## Caching

Wrap any synthesizer to cache audio on disk:

```python
from russo import CachedSynthesizer
from russo.synthesizers import GoogleSynthesizer

synth = CachedSynthesizer(
    GoogleSynthesizer(api_key="..."),
    cache_key_extra={"voice": "Kore"},
)
```

See the full examples in the [`examples/`](https://github.com/mohit2152sharma/russo/tree/main/examples) directory.
