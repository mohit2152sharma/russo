# Evaluators

Evaluators compare expected tool calls against the actual tool calls returned by the agent.

## Protocol

```python
class Evaluator(Protocol):
    def evaluate(self, expected: list[ToolCall], actual: list[ToolCall]) -> EvalResult: ...
```

## Built-in: ExactEvaluator

The default evaluator performs exact name + arguments matching:

```python
from russo.evaluators import ExactEvaluator

evaluator = ExactEvaluator()
```

### Configuration

```python
evaluator = ExactEvaluator(
    match_order=False,         # True = tool calls must match in order
    ignore_extra_args=False,   # True = actual may have extra arguments
    ignore_extra_calls=True,   # True = extra actual calls don't cause failure
)
```

### Examples

```python
from russo._types import ToolCall

expected = [ToolCall(name="book_flight", arguments={"from": "NYC", "to": "LA"})]
actual = [ToolCall(name="book_flight", arguments={"from": "NYC", "to": "LA"})]

result = evaluator.evaluate(expected=expected, actual=actual)
assert result.passed  # True
assert result.match_rate == 1.0
```

With `ignore_extra_args=True`:

```python
evaluator = ExactEvaluator(ignore_extra_args=True)

expected = [ToolCall(name="book_flight", arguments={"from": "NYC"})]
actual = [ToolCall(name="book_flight", arguments={"from": "NYC", "to": "LA", "class": "economy"})]

result = evaluator.evaluate(expected=expected, actual=actual)
assert result.passed  # True — extra args are ignored
```

## Custom Evaluators

Implement the protocol for custom matching logic:

```python
class FuzzyEvaluator:
    """Evaluator that uses fuzzy string matching on argument values."""

    def evaluate(self, expected: list[ToolCall], actual: list[ToolCall]) -> EvalResult:
        # Your custom logic here
        ...
```

## API Reference

See [`ExactEvaluator`](../reference/evaluators/exact.md) for the full API docs.
