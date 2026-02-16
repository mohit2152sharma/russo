# Basic Pipeline

The simplest way to use russo: synthesize audio from text, send it to an agent, and evaluate the tool calls it makes.

!!! tip "Source file"
    [`examples/basic_pipeline.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/basic_pipeline.py)

## Step 1: Define an agent

Every russo test needs an **agent** -- the thing you're testing. The `@russo.agent` decorator turns any async function with the right signature into a valid agent:

```python
import russo

@russo.agent
async def fake_agent(audio: russo.Audio) -> russo.AgentResponse:
    return russo.AgentResponse(
        tool_calls=[
            russo.ToolCall(
                name="book_flight",
                arguments={"from_city": "Berlin", "to_city": "Rome"},
            ),
        ]
    )
```

In a real test you'd swap this for a `GeminiLiveAgent`, `OpenAIAgent`, or any other adapter. We're using a fake here so the example runs without API keys.

## Step 2: Define a synthesizer

A **synthesizer** converts a text prompt into audio. Again, we use a fake one here -- in production you'd use `GoogleSynthesizer`:

```python
class FakeSynthesizer:
    async def synthesize(self, text: str) -> russo.Audio:
        return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)
```

Any object with an `async def synthesize(self, text: str) -> Audio` method satisfies the `Synthesizer` protocol.

## Step 3: Run the pipeline

`russo.run()` chains everything together: synthesize -> agent -> evaluate.

```python
from russo.evaluators import ExactEvaluator

result = await russo.run(
    prompt="Book a flight from Berlin to Rome for tomorrow",
    synthesizer=FakeSynthesizer(),
    agent=fake_agent,
    evaluator=ExactEvaluator(),
    expect=[
        russo.tool_call("book_flight", from_city="Berlin", to_city="Rome"),
    ],
)
```

The `expect` list defines what tool calls the agent *should* make. `russo.tool_call()` is a shorthand for creating `ToolCall` objects.

## Step 4: Check the results

```python
# Human-readable summary
print(result.summary())
# PASSED (100% match rate)
#   [+] book_flight({'from_city': 'Berlin', 'to_city': 'Rome'}) -> book_flight(...)

# Programmatic checks
assert result.passed
assert result.match_rate == 1.0

# Or use the assertion helper (raises ToolCallAssertionError on failure)
russo.assert_tool_calls(result)
```

`result` is an `EvalResult` with:

- `passed` -- did all expected tool calls match?
- `match_rate` -- fraction of expected calls matched (0.0 to 1.0)
- `matches` -- per-call match details
- `summary()` -- human-readable output

## Full example

```python
import asyncio
import russo
from russo.evaluators import ExactEvaluator


@russo.agent
async def fake_agent(audio: russo.Audio) -> russo.AgentResponse:
    return russo.AgentResponse(
        tool_calls=[
            russo.ToolCall(
                name="book_flight",
                arguments={"from_city": "Berlin", "to_city": "Rome"},
            ),
        ]
    )


class FakeSynthesizer:
    async def synthesize(self, text: str) -> russo.Audio:
        return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)


async def main():
    result = await russo.run(
        prompt="Book a flight from Berlin to Rome for tomorrow",
        synthesizer=FakeSynthesizer(),
        agent=fake_agent,
        evaluator=ExactEvaluator(),
        expect=[
            russo.tool_call("book_flight", from_city="Berlin", to_city="Rome"),
        ],
    )

    print(result.summary())
    assert result.passed
    assert result.match_rate == 1.0
    russo.assert_tool_calls(result)
    print("\nAll checks passed!")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python examples/basic_pipeline.py
```

Expected output:

```
PASSED (100% match rate)
  [+] book_flight({'from_city': 'Berlin', 'to_city': 'Rome'}) -> book_flight({'from_city': 'Berlin', 'to_city': 'Rome'})

All checks passed!
```
