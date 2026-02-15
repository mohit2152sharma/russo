# First Test

This guide walks you through writing your first russo test. You'll need a Google AI API key for the synthesizer and an agent to test against.

## Setup

```bash
pip install russo
export GOOGLE_API_KEY="your-api-key"
```

## The Pipeline

Every russo test follows the same flow:

1. **Synthesize** — convert a text prompt to audio
2. **Run** — send the audio to your agent
3. **Evaluate** — compare the agent's tool calls against expectations

## Writing the Test

### Step 1: Define Your Agent

You can wrap any async function as an agent using the `@russo.agent` decorator:

```python
import russo

@russo.agent
async def my_agent(audio: russo.Audio) -> russo.AgentResponse:
    # Call your real agent here with audio.data
    # For this example, we'll simulate a response
    return russo.AgentResponse(
        tool_calls=[
            russo.ToolCall(name="book_flight", arguments={"from_city": "NYC", "to_city": "LA"})
        ]
    )
```

### Step 2: Run the Pipeline

```python
import asyncio
from russo.synthesizers import GoogleSynthesizer
from russo.evaluators import ExactEvaluator

async def main():
    result = await russo.run(
        prompt="Book a flight from New York to Los Angeles",
        synthesizer=GoogleSynthesizer(api_key="..."),
        agent=my_agent,
        evaluator=ExactEvaluator(),
        expect=[
            russo.tool_call("book_flight", from_city="NYC", to_city="LA"),
        ],
    )

    print(result.summary())
    # PASSED (100% match rate)
    #   [+] book_flight({'from_city': 'NYC', 'to_city': 'LA'}) -> book_flight(...)

asyncio.run(main())
```

### Step 3: Use pytest

The most natural way to use russo is with pytest:

```python
# conftest.py
import pytest
from russo.synthesizers import GoogleSynthesizer
from russo.adapters import GeminiLiveAgent

@pytest.fixture(scope="session")
def russo_synthesizer():
    return GoogleSynthesizer(api_key="...")

@pytest.fixture(scope="session")
def russo_agent():
    return GeminiLiveAgent(api_key="...", model="gemini-2.0-flash-live-001", tools=[...])
```

```python
# test_flights.py
import pytest
import russo

@pytest.mark.russo(
    prompt="Book a flight from NYC to LA",
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
async def test_book_flight(russo_result):
    russo.assert_tool_calls(russo_result)

@pytest.mark.russo(
    prompt="What's the weather in Berlin?",
    expect=[russo.tool_call("get_weather", city="Berlin")],
)
async def test_weather(russo_result):
    assert russo_result.passed
    assert russo_result.match_rate == 1.0
```

Run it:

```bash
pytest -v
```

## What's Next?

- Learn about [adapters](../tutorial/adapters.md) for different agent types
- Explore [caching](../tutorial/caching.md) to speed up repeated test runs
- See the full [API reference](../reference/index.md)
