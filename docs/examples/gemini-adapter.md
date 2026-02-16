# Gemini Adapters

Test Gemini models for tool-call accuracy using the Google GenAI SDK. russo provides two adapters:

- **`GeminiAgent`** -- standard `generate_content` (request/response)
- **`GeminiLiveAgent`** -- Live API over WebSocket (streaming/real-time)

!!! tip "Source file"
    [`examples/gemini_adapter.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/gemini_adapter.py)

## Prerequisites

```bash
pip install russo
export GOOGLE_API_KEY="your-api-key"
```

## Tool declarations

Gemini uses the `function_declarations` format for tool definitions:

```python
BOOK_FLIGHT_TOOL = {
    "function_declarations": [
        {
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
    ]
}
```

## Example 1: GeminiAgent (request/response)

Standard request/response via `generate_content`. Best for non-streaming use cases.

```python
import os
import russo
from google import genai
from russo.adapters import GeminiAgent
from russo.evaluators import ExactEvaluator
from russo.synthesizers import GoogleSynthesizer

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

agent = GeminiAgent(
    client=client,
    model="gemini-2.0-flash",
    tools=[BOOK_FLIGHT_TOOL],
    system_instruction=(
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

## Example 2: GeminiLiveAgent (streaming)

Real-time streaming via the Live API. Opens a WebSocket connection, sends audio, and collects function calls as they arrive.

```python
from russo.adapters import GeminiLiveAgent

agent = GeminiLiveAgent(
    client=client,
    model="gemini-live-2.5-flash-native-audio",
    tools=[BOOK_FLIGHT_TOOL],
    system_instruction=(
        "You are a travel assistant. When the user asks to book "
        "a flight, call the book_flight function."
    ),
)

result = await russo.run(
    prompt="Book a flight from NYC to LA",
    synthesizer=synthesizer,
    agent=agent,
    evaluator=ExactEvaluator(),
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
```

!!! note "Model"
    Use ``gemini-live-2.5-flash-native-audio`` for both Google AI and Vertex AI.

## Example 3: Vertex AI authentication

Same adapters, but authenticated via Vertex AI using Application Default Credentials:

```python
# Vertex AI client — uses ADC
client = genai.Client(
    vertexai=True,
    project=os.environ.get("GOOGLE_CLOUD_PROJECT", "my-project"),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
)

agent = GeminiLiveAgent(
    client=client,
    model="gemini-live-2.5-flash-native-audio",
    tools=[BOOK_FLIGHT_TOOL],
)

synthesizer = GoogleSynthesizer(
    vertexai=True,
    project=os.environ.get("GOOGLE_CLOUD_PROJECT", "my-project"),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
)
```

Run whichever example matches your setup:

```bash
python examples/gemini_adapter.py
```
