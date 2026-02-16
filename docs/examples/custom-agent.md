# Custom Agent

The `@russo.agent` decorator turns any async function into a valid agent. This is useful when your agent is a REST API call, a gRPC service, or anything that doesn't fit the built-in adapters.

!!! tip "Source file"
    [`examples/custom_agent.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/custom_agent.py)

## The `@russo.agent` decorator

The decorator wraps any async function with the signature `(Audio) -> AgentResponse`:

```python
import russo

@russo.agent
async def my_travel_agent(audio: russo.Audio) -> russo.AgentResponse:
    # In a real scenario you'd call your backend:
    #   response = await httpx.post("https://my-api/voice", content=audio.data)
    #   parsed = response.json()
    #   tool_calls = [
    #       russo.ToolCall(name=tc["name"], arguments=tc["args"])
    #       for tc in parsed["tool_calls"]
    #   ]

    return russo.AgentResponse(
        tool_calls=[
            russo.ToolCall(
                name="search_hotels",
                arguments={"city": "Paris", "checkin": "2026-03-01"},
            ),
        ]
    )
```

The decorated function satisfies the `Agent` protocol, so you can pass it directly to `russo.run()`.

## How it works

Under the hood, `@russo.agent` wraps your function in a `_CallableAgent` class that has a `run(audio)` method -- exactly what the `Agent` protocol requires:

```python
# These two are equivalent:
agent = my_travel_agent  # decorated function

# vs. implementing the protocol manually:
class MyAgent:
    async def run(self, audio: russo.Audio) -> russo.AgentResponse:
        return await my_travel_agent(audio)
```

## Running the test

```python
import asyncio
from russo.evaluators import ExactEvaluator

class FakeSynthesizer:
    async def synthesize(self, text: str) -> russo.Audio:
        return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)

async def main():
    result = await russo.run(
        prompt="Find hotels in Paris for March first",
        synthesizer=FakeSynthesizer(),
        agent=my_travel_agent,
        evaluator=ExactEvaluator(),
        expect=[
            russo.tool_call("search_hotels", city="Paris", checkin="2026-03-01"),
        ],
    )

    print(result.summary())
    russo.assert_tool_calls(result)

asyncio.run(main())
```

Run it:

```bash
python examples/custom_agent.py
```

Expected output:

```
PASSED (100% match rate)
  [+] search_hotels({'city': 'Paris', 'checkin': '2026-03-01'}) -> search_hotels(...)

Custom agent test passed!
```
