# HTTP Agent

Test any HTTP endpoint for tool-call accuracy. `HttpAgent` sends audio as base64-encoded JSON and parses the response -- no SDK dependency needed on the server side.

!!! tip "Source file"
    [`examples/http_agent.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/http_agent.py)

## Default protocol

`HttpAgent` uses a simple JSON protocol out of the box:

**Request:**

```json
POST /voice-agent
{
    "audio": "<base64-encoded audio>",
    "format": "wav"
}
```

**Expected response:**

```json
{
    "tool_calls": [
        {"name": "book_flight", "arguments": {"from_city": "NYC", "to_city": "LA"}}
    ]
}
```

## Example 1: Basic HTTP agent

Point russo at any HTTP endpoint:

```python
import russo
from russo.adapters import HttpAgent
from russo.evaluators import ExactEvaluator

agent = HttpAgent(
    url="http://localhost:8000/voice-agent",
)

result = await russo.run(
    prompt="Book a flight from NYC to LA",
    synthesizer=my_synthesizer,
    agent=agent,
    evaluator=ExactEvaluator(),
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
```

## Example 2: Custom headers and response parser

Add auth headers and use a built-in parser for provider-specific response formats:

```python
from russo.parsers import GeminiResponseParser

agent = HttpAgent(
    url="https://my-api.example.com/v1/agent",
    headers={
        "Authorization": "Bearer my-token",
        "X-Request-ID": "russo-test-001",
    },
    parser=GeminiResponseParser(),  # parse Gemini-format responses
    timeout=30.0,
)
```

When a `parser` is provided, `HttpAgent` passes the raw HTTP response body to the parser instead of using the default JSON protocol.

## Example 3: Custom field names

If your server expects different JSON field names:

```python
agent = HttpAgent(
    url="http://localhost:8000/api/voice",
    audio_field="audio_data",     # sends "audio_data" instead of "audio"
    format_field="audio_format",  # sends "audio_format" instead of "format"
)
```

This sends:

```json
{
    "audio_data": "<base64>",
    "audio_format": "wav"
}
```

```bash
python examples/http_agent.py
```

!!! note
    These examples require a running HTTP server. Replace the URL with your actual endpoint.
