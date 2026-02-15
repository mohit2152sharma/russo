# Adapters

Adapters connect russo to your agent. Each adapter implements the `Agent` protocol:

```python
class Agent(Protocol):
    async def run(self, audio: Audio) -> AgentResponse: ...
```

russo ships with adapters for the most common patterns.

## Built-in Adapters

### Gemini SDK

```python
from russo.adapters import GeminiAgent, GeminiLiveAgent
```

**`GeminiAgent`** — standard `generate_content` (request/response):

```python
agent = GeminiAgent(
    api_key="...",
    model="gemini-2.0-flash",
    tools=[{"function_declarations": [...]}],
)
```

**`GeminiLiveAgent`** — Live API over WebSocket (streaming/real-time):

```python
agent = GeminiLiveAgent(
    api_key="...",
    model="gemini-2.0-flash-live-001",
    tools=[{"function_declarations": [...]}],
)
```

### OpenAI SDK

!!! note
    Requires the `openai` extra: `pip install "russo[openai]"`

```python
from russo.adapters import OpenAIAgent, OpenAIRealtimeAgent
```

**`OpenAIAgent`** — Chat Completions with audio input:

```python
agent = OpenAIAgent(
    api_key="...",
    model="gpt-4o-audio-preview",
    tools=[...],
)
```

**`OpenAIRealtimeAgent`** — Realtime API over WebSocket:

```python
agent = OpenAIRealtimeAgent(
    api_key="...",
    model="gpt-4o-realtime-preview",
    tools=[...],
)
```

### HTTP

Send audio to any HTTP endpoint:

```python
from russo.adapters import HttpAgent
from russo.parsers import GeminiResponseParser

agent = HttpAgent(
    url="https://my-agent.example.com/audio",
    parser=GeminiResponseParser(),
)
```

### WebSocket

!!! note
    Requires the `ws` extra: `pip install "russo[ws]"`

```python
from russo.adapters import WebSocketAgent

agent = WebSocketAgent(
    url="wss://my-agent.example.com/ws",
    parser=my_parser,
)
```

### Callable (Custom)

Wrap any async function:

```python
import russo

@russo.agent
async def my_agent(audio: russo.Audio) -> russo.AgentResponse:
    # Your custom logic here
    result = await call_my_api(audio.data)
    return russo.AgentResponse(
        tool_calls=[russo.ToolCall(name="...", arguments={...})]
    )
```

Or use the class directly:

```python
from russo.adapters import CallableAgent

agent = CallableAgent(my_async_function)
```

## Custom Adapters

You don't need to inherit from anything. Just implement the `run` method:

```python
class MyCustomAgent:
    async def run(self, audio: russo.Audio) -> russo.AgentResponse:
        # Send audio to your service
        raw = await my_service.process(audio.data)
        # Parse and return
        return russo.AgentResponse(
            tool_calls=[russo.ToolCall(name=raw["tool"], arguments=raw["args"])]
        )
```

russo uses structural typing — if your class has `async def run(self, audio: Audio) -> AgentResponse`, it's an Agent.

## API Reference

See the [Adapters reference](../reference/adapters/index.md) for full API docs on each adapter.
