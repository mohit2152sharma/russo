# Protocols

russo uses Python's `typing.Protocol` for all extension points. This means **structural subtyping** — you don't need to inherit from a base class. If your class has the right methods with the right signatures, it satisfies the protocol.

## Why Protocols?

- **No coupling** — your code never depends on russo base classes
- **Duck typing with type safety** — mypy/pyright verify protocol conformance at check time
- **Runtime checkable** — all russo protocols are `@runtime_checkable`, so `isinstance()` works too

## The Four Protocols

### Synthesizer

Converts text to audio.

```python
@runtime_checkable
class Synthesizer(Protocol):
    async def synthesize(self, text: str) -> Audio: ...
```

**Implementations**: `GoogleSynthesizer`, `CachedSynthesizer`, or any class with a matching `synthesize` method.

### Agent

The agent under test. Takes audio, returns tool calls.

```python
@runtime_checkable
class Agent(Protocol):
    async def run(self, audio: Audio) -> AgentResponse: ...
```

**Implementations**: `GeminiAgent`, `GeminiLiveAgent`, `OpenAIAgent`, `OpenAIRealtimeAgent`, `HttpAgent`, `WebSocketAgent`, `CallableAgent`, or any class with a matching `run` method.

### Evaluator

Compares expected tool calls against actual ones.

```python
@runtime_checkable
class Evaluator(Protocol):
    def evaluate(self, expected: list[ToolCall], actual: list[ToolCall]) -> EvalResult: ...
```

**Implementations**: `ExactEvaluator`, or any class with a matching `evaluate` method.

### ResponseParser

Normalizes provider-specific raw responses into `AgentResponse`.

```python
@runtime_checkable
class ResponseParser(Protocol):
    def parse(self, raw_response: Any) -> AgentResponse: ...
```

**Implementations**: `GeminiResponseParser`, `OpenAIResponseParser`, or any class with a matching `parse` method.

## Implementing a Protocol

Just write a class with the right method:

```python
import russo

class MyEvaluator:
    def evaluate(
        self,
        expected: list[russo.ToolCall],
        actual: list[russo.ToolCall],
    ) -> russo.EvalResult:
        # Your custom matching logic
        passed = len(expected) == len(actual)
        return russo.EvalResult(
            passed=passed,
            expected=expected,
            actual=actual,
        )

# Works with russo.run() — no inheritance needed
result = await russo.run(
    ...,
    evaluator=MyEvaluator(),
    ...,
)
```

## Runtime Checking

```python
from russo._protocols import Synthesizer, Agent, Evaluator

assert isinstance(my_synth, Synthesizer)    # True if it has .synthesize()
assert isinstance(my_agent, Agent)          # True if it has .run()
assert isinstance(my_eval, Evaluator)       # True if it has .evaluate()
```

## API Reference

See the [Protocols reference](../reference/core/protocols.md) for full API docs.
