# Custom Response Parser

Use `JsonResponseParser` to handle any HTTP or WebSocket endpoint response structure without writing a full parser class.

!!! tip "Source file"
    [`examples/custom_parser.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/custom_parser.py)

## When to use it

The default `HttpAgent` and `WebSocketAgent` expect responses in this shape:

```json
{"tool_calls": [{"name": "...", "arguments": {...}}]}
```

If your endpoint returns tool calls under a different key, with different field names, or nested inside an envelope, use `JsonResponseParser` to describe the structure declaratively.

## Example 1: Custom top-level key

```python
from russo.parsers import JsonResponseParser
from russo.adapters import HttpAgent

# Endpoint returns: {"toolCall": [{"name": "book_flight", "arguments": {...}}]}
agent = HttpAgent(
    url="http://localhost:8000/agent",
    parser=JsonResponseParser(tool_calls_key="toolCall"),
)
```

## Example 2: Single tool call object

```python
# Endpoint returns: {"toolCall": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}
agent = HttpAgent(
    url="http://localhost:8000/agent",
    parser=JsonResponseParser(tool_calls_key="toolCall", single=True),
)
```

Set `single=True` when the endpoint returns one tool call dict directly instead of a list.

## Example 3: Custom field names

```python
# Endpoint returns: {"calls": [{"function": "search", "params": {"query": "hello"}}]}
agent = HttpAgent(
    url="http://localhost:8000/agent",
    parser=JsonResponseParser(
        tool_calls_key="calls",
        name_key="function",
        arguments_key="params",
    ),
)
```

## Example 4: Nested key path

Use dot notation to reach tool calls inside a response envelope:

```python
# Endpoint returns: {"response": {"data": {"tool_calls": [...]}}}
agent = HttpAgent(
    url="http://localhost:8000/agent",
    parser=JsonResponseParser(tool_calls_key="response.data.tool_calls"),
)
```

## Example 5: WebSocket with custom structure

`JsonResponseParser` also works with `WebSocketAgent`. When the adapter aggregates multiple messages into a list, the parser scans each message and returns tool calls from the first matching one.

```python
from russo.adapters import WebSocketAgent

# Server streams several messages; one contains the tool call
agent = WebSocketAgent(
    url="ws://localhost:8000/ws/agent",
    parser=JsonResponseParser(tool_calls_key="toolCall"),
    is_complete=lambda msgs: any(
        isinstance(m, dict) and m.get("status") == "done" for m in msgs
    ),
    response_timeout=15.0,
)
```

## All options

| Argument | Default | Description |
|----------|---------|-------------|
| `tool_calls_key` | `"tool_calls"` | Dot-path to the tool calls in the response |
| `name_key` | `"name"` | Key for the function name within each call |
| `arguments_key` | `"arguments"` | Key for the arguments dict within each call |
| `single` | `False` | `True` if the key points to a single object instead of a list |

```bash
python examples/custom_parser.py
```

!!! note
    These examples require a running server. Replace the URL with your actual endpoint.
