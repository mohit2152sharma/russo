<p align="center">
  <a href="https://mohit2152sharma.github.io/russo"><img src="https://mohit2152sharma.github.io/russo/assets/logo.svg" alt="russo" width="300"></a>
</p>
<p align="center">
    <em>Testing framework for LLM tool-call accuracy — audio & text</em>
</p>
<p align="center">
<a href="https://pypi.org/project/russo" target="_blank">
    <img src="https://img.shields.io/pypi/v/russo?color=%2334D058&label=pypi" alt="PyPI version">
</a>
<a href="https://pypi.org/project/russo" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/russo.svg?color=%2334D058" alt="Supported Python versions">
</a>
<a href="https://github.com/mohit2152sharma/russo/actions/workflows/ci.yml" target="_blank">
    <img src="https://github.com/mohit2152sharma/russo/actions/workflows/ci.yml/badge.svg" alt="CI">
</a>
<a href="https://github.com/mohit2152sharma/russo/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/mohit2152sharma/russo.svg?color=%2334D058" alt="License">
</a>
</p>

---

**Documentation**: <a href="https://mohit2152sharma.github.io/russo" target="_blank">https://mohit2152sharma.github.io/russo</a>

**Source Code**: <a href="https://github.com/mohit2152sharma/russo" target="_blank">https://github.com/mohit2152sharma/russo</a>

---

**russo** is a testing framework for verifying that LLM agents make the correct tool calls when given audio (or text) input. Think of it as pytest for voice AI tool-calling accuracy.

## Key Features

- **Provider-agnostic** — works with Gemini, OpenAI, or any custom agent via protocols
- **Audio-first** — synthesize text prompts to audio, send to your agent, evaluate tool calls
- **pytest integration** — use markers, fixtures, and familiar test patterns
- **Built-in caching** — skip TTS on repeated runs, saving time and money
- **Extensible** — swap synthesizers, agents, evaluators, and parsers via structural typing

## Quick Start

```bash
pip install russo
```

With optional providers:

```bash
pip install "russo[openai]"    # OpenAI support
pip install "russo[ws]"        # WebSocket agents
pip install "russo[all]"       # Everything
```

## Minimal Example

```python
import russo
from russo.synthesizers import GoogleSynthesizer
from russo.adapters import GeminiLiveAgent
from russo.evaluators import ExactEvaluator

result = await russo.run(
    prompt="Book a flight from Berlin to Rome for tomorrow",
    synthesizer=GoogleSynthesizer(api_key="..."),
    agent=GeminiLiveAgent(api_key="...", tools=[...]),
    evaluator=ExactEvaluator(),
    expect=[
        russo.tool_call("book_flight", from_city="Berlin", to_city="Rome"),
    ],
)

russo.assert_tool_calls(result)
```

## pytest Integration

```python
import pytest
import russo

@pytest.mark.russo(
    prompt="Book a flight from NYC to LA",
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
async def test_book_flight(russo_result):
    russo.assert_tool_calls(russo_result)
```

## License

This project is licensed under the terms of the [MIT license](LICENSE).
