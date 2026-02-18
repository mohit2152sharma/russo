# WebSocket Agent

Test WebSocket endpoints for tool-call accuracy. `WebSocketAgent` supports JSON mode, raw bytes mode, and fully custom protocols via hooks.

!!! tip "Source file"
    [`examples/websocket_agent.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/websocket_agent.py)

## Prerequisites

```bash
pip install "russo[ws]"
```

## Example 1: JSON protocol (default)

Send audio as base64 JSON, receive a JSON response:

```python
import russo
from russo.adapters import WebSocketAgent
from russo.evaluators import ExactEvaluator

agent = WebSocketAgent(
    url="ws://localhost:8000/ws/agent",
    response_timeout=15.0,
)

result = await russo.run(
    prompt="Book a flight from NYC to LA",
    synthesizer=my_synthesizer,
    agent=agent,
    evaluator=ExactEvaluator(),
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
```

The default sends `{"audio": "<base64>", "format": "wav"}` and expects the same response format as `HttpAgent`.

## Example 2: Raw bytes mode

Send raw audio bytes directly on the wire -- useful for streaming PCM to servers that expect binary frames:

```python
agent = WebSocketAgent(
    url="ws://localhost:8000/ws/stream",
    send_bytes=True,
    response_timeout=10.0,
)
```

## Example 3: Custom protocol with hooks

Full control over the send/receive cycle using three hooks:

### `on_send` -- customize the outgoing message

```python
import base64
import json

def custom_send(audio: russo.Audio) -> str:
    """Build a custom JSON message for your server."""
    return json.dumps({
        "type": "audio_input",
        "pcm": base64.b64encode(audio.data).decode(),
        "sample_rate": audio.sample_rate,
    })
```

### `is_complete` -- decide when to stop collecting responses

```python
def is_done(messages: list) -> bool:
    """Stop collecting when the server sends a 'done' message."""
    return any(
        isinstance(m, dict) and m.get("type") == "done"
        for m in messages
    )
```

### `aggregate` -- combine collected messages into one response

```python
def aggregate_responses(messages: list):
    """Combine all tool_call messages into one response."""
    tool_calls = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("type") == "tool_call":
            tool_calls.append(msg)
    return {"tool_calls": tool_calls}
```

### Putting it together

```python
agent = WebSocketAgent(
    url="ws://localhost:8000/ws/custom",
    on_send=custom_send,
    is_complete=is_done,
    aggregate=aggregate_responses,
    response_timeout=20.0,
    max_messages=50,
)
```

## Example 4: Custom response structure

If your endpoint returns tool calls under a different key or structure, use `JsonResponseParser`:

```python
from russo.parsers import JsonResponseParser

# Endpoint returns: {"toolCall": [{"name": "book_flight", "arguments": {...}}]}
agent = WebSocketAgent(
    url="ws://localhost:8000/ws/agent",
    parser=JsonResponseParser(tool_calls_key="toolCall"),
)
```

When the WebSocket adapter aggregates multiple messages into a list, the parser automatically scans each message and returns tool calls from the first matching one.

See [Custom Response Parser](custom-parser.md) for full examples.

## Example 5: Provider-specific parser

Use a built-in parser for provider-specific response formats:

```python
from russo.parsers import GeminiResponseParser

agent = WebSocketAgent(
    url="ws://localhost:8000/ws/gemini",
    parser=GeminiResponseParser(),
    headers={"Authorization": "Bearer my-token"},
)
```

```bash
python examples/websocket_agent.py
```

!!! note
    These examples require a running WebSocket server. Replace the URL with your actual endpoint.
