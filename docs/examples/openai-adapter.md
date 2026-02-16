# OpenAI Adapters

Test OpenAI models for tool-call accuracy. russo provides two adapters:

- **`OpenAIAgent`** -- Chat Completions with audio input (`gpt-4o-audio-preview`)
- **`OpenAIRealtimeAgent`** -- Realtime API over WebSocket (`gpt-4o-realtime-preview`)

!!! tip "Source file"
    [`examples/openai_adapter.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/openai_adapter.py)

## Prerequisites

```bash
pip install "russo[openai]"
export OPENAI_API_KEY="your-api-key"
export GOOGLE_API_KEY="your-google-key"  # for TTS synthesizer
```

## Tool definitions

OpenAI uses slightly different formats for Chat Completions vs Realtime:

=== "Chat Completions"

    ```python
    BOOK_FLIGHT_TOOL = {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a flight between two cities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_city": {"type": "string", "description": "Departure city"},
                    "to_city": {"type": "string", "description": "Arrival city"},
                },
                "required": ["from_city", "to_city"],
            },
        },
    }
    ```

=== "Realtime API"

    ```python
    BOOK_FLIGHT_REALTIME_TOOL = {
        "type": "function",
        "name": "book_flight",
        "description": "Book a flight between two cities.",
        "parameters": {
            "type": "object",
            "properties": {
                "from_city": {"type": "string", "description": "Departure city"},
                "to_city": {"type": "string", "description": "Arrival city"},
            },
            "required": ["from_city", "to_city"],
        },
    }
    ```

## Example 1: OpenAIAgent (Chat Completions)

Send audio via the Chat Completions API:

```python
import os
import russo
from openai import AsyncOpenAI
from russo.adapters import OpenAIAgent
from russo.evaluators import ExactEvaluator
from russo.synthesizers import GoogleSynthesizer

client = AsyncOpenAI()  # reads OPENAI_API_KEY from env

agent = OpenAIAgent(
    client=client,
    model="gpt-4o-audio-preview",
    tools=[BOOK_FLIGHT_TOOL],
    system_prompt=(
        "You are a travel assistant. When the user asks to book "
        "a flight, call the book_flight function."
    ),
)

synthesizer = GoogleSynthesizer(api_key=os.environ["GOOGLE_API_KEY"])

result = await russo.run(
    prompt="Book a flight from Berlin to Rome for tomorrow",
    synthesizer=synthesizer,
    agent=agent,
    evaluator=ExactEvaluator(),
    expect=[russo.tool_call("book_flight", from_city="Berlin", to_city="Rome")],
)
```

## Example 2: OpenAIRealtimeAgent

Stream audio via the Realtime API:

```python
from russo.adapters import OpenAIRealtimeAgent

agent = OpenAIRealtimeAgent(
    client=client,
    model="gpt-4o-realtime-preview",
    tools=[BOOK_FLIGHT_REALTIME_TOOL],
    response_timeout=30.0,
)

result = await russo.run(
    prompt="Book a flight from NYC to LA",
    synthesizer=synthesizer,
    agent=agent,
    evaluator=ExactEvaluator(),
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
```

!!! note
    The Realtime API expects **pcm16** audio at 24 kHz mono by default. If you pass WAV audio, the adapter automatically strips the WAV header to extract raw PCM frames.

## Example 3: Pre-existing Realtime connection

Reuse a single WebSocket connection for multiple tests:

```python
async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as conn:
    agent = OpenAIRealtimeAgent(connection=conn)

    result = await russo.run(
        prompt="Book a flight from London to Paris",
        synthesizer=synthesizer,
        agent=agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("book_flight", from_city="London", to_city="Paris")],
    )
```

This avoids the overhead of opening a new WebSocket per test.

```bash
python examples/openai_adapter.py
```
