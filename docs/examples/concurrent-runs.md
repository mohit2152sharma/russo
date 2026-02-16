# Concurrent Runs

Run the pipeline multiple times asynchronously — for reliability testing, prompt variant testing, or a full matrix of both. All runs execute concurrently via `asyncio.gather()`.

!!! tip "Source file"
    [`examples/concurrent_runs.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/concurrent_runs.py)

## Three scenarios

| Scenario | What it does | Use case |
|----------|-------------|----------|
| Single prompt, N runs | Same prompt repeated N times | Reliability / flakiness testing |
| Multiple prompts, 1 run | Each prompt runs once | Prompt variant testing |
| M prompts × N runs | Full matrix (M × N tasks) | Comprehensive coverage |

## Scenario 1: Single prompt, multiple runs

Test how reliably your agent handles the same prompt by running it N times concurrently:

```python
import russo
from russo.evaluators import ExactEvaluator

result = await russo.run_concurrent(
    prompts="Book a flight from NYC to LA",
    synthesizer=my_synthesizer,
    agent=my_agent,
    evaluator=ExactEvaluator(ignore_extra_args=True),
    expect=[russo.tool_call("book_flight")],
    runs=5,
)

print(result.summary())
assert result.pass_rate >= 0.8  # at least 80% should pass
```

`result` is a `BatchResult` with:

- `total` — number of runs (5)
- `passed` — `True` only if *every* run passed
- `pass_rate` — fraction of runs that passed (0.0 to 1.0)
- `match_rate` — average match rate across all runs
- `runs` — list of `SingleRunResult` for per-run inspection

## Scenario 2: Multiple prompts, single run each

Test different phrasings of the same intent:

```python
result = await russo.run_concurrent(
    prompts=[
        "Book a flight from NYC to LA",
        "I need to fly from New York to Los Angeles",
        "Please book me a flight, departing NYC, arriving LA",
    ],
    synthesizer=my_synthesizer,
    agent=my_agent,
    evaluator=ExactEvaluator(ignore_extra_args=True),
    expect=[russo.tool_call("book_flight")],
)

assert result.total == 3
assert result.passed
```

## Scenario 3: Multiple prompts × multiple runs

Full matrix — every prompt is run N times:

```python
result = await russo.run_concurrent(
    prompts=[
        "Book a flight from Berlin to Rome",
        "Fly me from Tokyo to Sydney",
    ],
    synthesizer=my_synthesizer,
    agent=my_agent,
    evaluator=ExactEvaluator(ignore_extra_args=True),
    expect=[russo.tool_call("book_flight")],
    runs=3,
    max_concurrency=4,  # at most 4 simultaneous pipeline runs
)

assert result.total == 6  # 2 prompts × 3 runs
```

## Controlling concurrency

By default, all runs execute simultaneously. Use `max_concurrency` to limit the number of parallel tasks — useful when hitting rate-limited APIs:

```python
result = await russo.run_concurrent(
    prompts="test prompt",
    ...,
    runs=20,
    max_concurrency=5,  # at most 5 calls at a time
)
```

## Inspecting individual results

Each run is accessible as a `SingleRunResult`:

```python
for run in result.runs:
    icon = "+" if run.eval_result.passed else "-"
    print(f"[{icon}] prompt={run.prompt!r}  run={run.run_index}  "
          f"match_rate={run.eval_result.match_rate:.0%}")
```

The `summary()` method groups results by prompt automatically:

```
PASSED (100% pass rate, 6 runs)
  Passed: 6/6
  Prompt: 'Book a flight from Berlin to Rome'
    3/3 passed
    [+] run 0: 100% match
    [+] run 1: 100% match
    [+] run 2: 100% match
  Prompt: 'Fly me from Tokyo to Sydney'
    3/3 passed
    [+] run 0: 100% match
    [+] run 1: 100% match
    [+] run 2: 100% match
```

## pytest integration

The same functionality is available via the `@pytest.mark.russo` marker:

### Single prompt, multiple runs

```python
@pytest.mark.russo(
    prompt="Book a flight from NYC to LA",
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
    runs=3,
)
async def test_reliability(russo_result):
    assert isinstance(russo_result, russo.BatchResult)
    assert russo_result.pass_rate >= 0.8
```

### Multiple prompts

```python
@pytest.mark.russo(
    prompts=[
        "Book a flight from NYC to LA",
        "I need to fly from NYC to LA",
    ],
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
async def test_prompt_variants(russo_result):
    assert russo_result.total == 2
    assert russo_result.passed
```

### Full matrix

```python
@pytest.mark.russo(
    prompts=["Book from NYC to LA", "Fly NYC to LA"],
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
    runs=3,
    max_concurrency=4,
)
async def test_full_matrix(russo_result):
    assert russo_result.total == 6  # 2 prompts × 3 runs
```

### CLI run count override

Set a default run count for *all* russo tests from the command line:

```bash
pytest --russo-runs 5
pytest --russo-runs 5 --russo-max-concurrency 3
```

Marker-level `runs=` takes precedence over the CLI option.

!!! tip "Source files"
    [`examples/pytest_integration/test_concurrent.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/pytest_integration/test_concurrent.py)

## CLI (config-driven)

Add `runs` and `max_concurrency` to your config file:

```yaml
pipeline:
  # ... component references ...

tests:
  - id: "book_flight_test"
    # ... test spec ...

runs: 5               # run each test 5 times
max_concurrency: 3    # at most 3 simultaneous runs
```

Or override from the command line:

```bash
russo --config config.yaml --runs 10 --max-concurrency 5
```

The `--runs` flag overrides the config file value.

## Full example

```python
import asyncio
import russo
from russo.evaluators import ExactEvaluator


class FakeSynthesizer:
    async def synthesize(self, text: str) -> russo.Audio:
        return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)


@russo.agent
async def fake_agent(audio: russo.Audio) -> russo.AgentResponse:
    return russo.AgentResponse(
        tool_calls=[russo.ToolCall(name="book_flight", arguments={"from_city": "NYC", "to_city": "LA"})]
    )


async def main():
    result = await russo.run_concurrent(
        prompts=["Book a flight from NYC to LA", "Fly me from NYC to LA"],
        synthesizer=FakeSynthesizer(),
        agent=fake_agent,
        evaluator=ExactEvaluator(ignore_extra_args=True),
        expect=[russo.tool_call("book_flight")],
        runs=3,
        max_concurrency=4,
    )

    print(result.summary())
    assert result.passed
    print(f"\nAll {result.total} runs passed!")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python examples/concurrent_runs.py
```
