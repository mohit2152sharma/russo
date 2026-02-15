# Pipeline

The pipeline is the core of russo. It chains three components together: **Synthesizer → Agent → Evaluator**.

## `russo.run()`

The `run` function is the main entry point:

```python
result = await russo.run(
    prompt="Book a flight from Berlin to Rome",
    synthesizer=my_synthesizer,
    agent=my_agent,
    evaluator=ExactEvaluator(),
    expect=[russo.tool_call("book_flight", from_city="Berlin", to_city="Rome")],
)
```

All arguments are keyword-only. Here's what each does:

| Argument | Type | Purpose |
|---|---|---|
| `prompt` | `str` | Text to synthesize into audio |
| `synthesizer` | `Synthesizer` | Converts text → audio |
| `agent` | `Agent` | The agent under test (audio → tool calls) |
| `evaluator` | `Evaluator` | Compares expected vs actual tool calls |
| `expect` | `list[ToolCall]` | The tool calls you expect the agent to make |

## What Happens Inside

```python
async def run(*, prompt, synthesizer, agent, evaluator, expect):
    audio = await synthesizer.synthesize(prompt)    # 1. Text → Audio
    response = await agent.run(audio)               # 2. Audio → AgentResponse
    return evaluator.evaluate(                      # 3. Compare
        expected=expect,
        actual=response.tool_calls,
    )
```

That's it. Three steps. Each step is pluggable.

## The Result

`russo.run()` returns an `EvalResult`:

```python
result.passed       # bool — did all expected tool calls match?
result.match_rate   # float — fraction of expected calls that matched (0.0–1.0)
result.expected     # list[ToolCall] — what you expected
result.actual       # list[ToolCall] — what the agent returned
result.matches      # list[ToolCallMatch] — per-call match details
result.summary()    # str — human-readable summary
```

## Assertions

Use `russo.assert_tool_calls()` for rich error messages:

```python
russo.assert_tool_calls(result)
# Raises ToolCallAssertionError with detailed diff if it fails
```

Or use standard assertions:

```python
assert result.passed
assert result.match_rate >= 0.8  # at least 80% match
```

## API Reference

See [`russo.run()`](../reference/core/pipeline.md) for the full API docs.
