# Custom Evaluator

russo uses structural subtyping (protocols), so you don't need to inherit from anything. Just implement the right method signature and you have a valid evaluator.

!!! tip "Source file"
    [`examples/custom_evaluator.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/custom_evaluator.py)

## The Evaluator protocol

Any class with this method is a valid evaluator:

```python
def evaluate(self, expected: list[ToolCall], actual: list[ToolCall]) -> EvalResult
```

No base class, no registration -- if the method signature matches, russo accepts it.

## Building a fuzzy evaluator

The built-in `ExactEvaluator` requires exact name and argument matches. Let's build a **fuzzy** evaluator that:

- Matches tool names **case-insensitively**
- Only checks that expected arguments are a **subset** of actual arguments (extra args are OK)

### Step 1: Define the evaluator class

```python
import russo
from russo._types import EvalResult, ToolCall, ToolCallMatch


class FuzzyEvaluator:
    """Case-insensitive name matching + subset argument checking."""

    def evaluate(self, expected: list[ToolCall], actual: list[ToolCall]) -> EvalResult:
        matches: list[ToolCallMatch] = []
        remaining = list(actual)

        for exp in expected:
            match = self._find_match(exp, remaining)
            matches.append(match)
            if match.matched and match.actual in remaining:
                remaining.remove(match.actual)

        passed = all(m.matched for m in matches)
        return EvalResult(passed=passed, expected=expected, actual=actual, matches=matches)
```

### Step 2: Implement the matching logic

```python
    def _find_match(self, expected: ToolCall, candidates: list[ToolCall]) -> ToolCallMatch:
        for candidate in candidates:
            if self._is_match(expected, candidate):
                return ToolCallMatch(expected=expected, actual=candidate, matched=True)

        return ToolCallMatch(
            expected=expected,
            matched=False,
            details=f"No fuzzy match found for {expected.name}",
        )

    def _is_match(self, expected: ToolCall, actual: ToolCall) -> bool:
        # Case-insensitive name matching
        if expected.name.lower() != actual.name.lower():
            return False
        # Expected args must be a subset of actual args
        return all(actual.arguments.get(k) == v for k, v in expected.arguments.items())
```

### Step 3: Test it

The fuzzy evaluator handles cases where the agent returns different casing or extra arguments:

```python
@russo.agent
async def agent_with_extra_args(audio: russo.Audio) -> russo.AgentResponse:
    return russo.AgentResponse(
        tool_calls=[
            russo.ToolCall(
                name="Book_Flight",  # different casing!
                arguments={
                    "from_city": "NYC",
                    "to_city": "LA",
                    "airline": "Delta",  # extra argument
                    "class": "economy",  # extra argument
                },
            ),
        ]
    )
```

Now run the pipeline with the fuzzy evaluator:

```python
result = await russo.run(
    prompt="Book a flight from NYC to LA",
    synthesizer=FakeSynthesizer(),
    agent=agent_with_extra_args,
    evaluator=FuzzyEvaluator(),  # our custom evaluator
    expect=[
        russo.tool_call("book_flight", from_city="NYC", to_city="LA"),
    ],
)

russo.assert_tool_calls(result)
# Passes! "Book_Flight" matches "book_flight", extra args are ignored.
```

Run it:

```bash
python examples/custom_evaluator.py
```

Expected output:

```
PASSED (100% match rate)
  [+] book_flight({'from_city': 'NYC', 'to_city': 'LA'}) -> Book_Flight({'from_city': 'NYC', 'to_city': 'LA', 'airline': 'Delta', 'class': 'economy'})

Fuzzy evaluator matched despite case difference and extra args!
```

## When to use a custom evaluator

- **Fuzzy matching** -- case-insensitive names, partial argument matches
- **Semantic matching** -- use an LLM to judge if arguments are "close enough"
- **Scoring** -- return a confidence score instead of pass/fail
- **Subset matching** -- only require some of the expected tool calls to match
