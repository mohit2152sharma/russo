# russo Examples

Runnable examples demonstrating every major way to use the **russo** testing framework.

## Prerequisites

```bash
pip install russo
# For provider-specific examples:
pip install "russo[openai]"     # OpenAI adapters
pip install "russo[ws]"         # WebSocket adapter
pip install "russo[all]"        # Everything
```

Most examples that hit real APIs expect a `GOOGLE_API_KEY` or `OPENAI_API_KEY` environment variable.

## Examples

| File | Description |
|------|-------------|
| [`basic_pipeline.py`](basic_pipeline.py) | Minimal `russo.run()` end-to-end pipeline with a fake agent |
| [`custom_agent.py`](custom_agent.py) | Wrap any async function as an agent with `@russo.agent` |
| [`custom_evaluator.py`](custom_evaluator.py) | Build a custom evaluator using the `Evaluator` protocol (no inheritance) |
| [`custom_synthesizer.py`](custom_synthesizer.py) | Build a file-based synthesizer for offline testing |
| [`gemini_adapter.py`](gemini_adapter.py) | Use `GeminiAgent` and `GeminiLiveAgent` with the Google GenAI SDK |
| [`openai_adapter.py`](openai_adapter.py) | Use `OpenAIAgent` and `OpenAIRealtimeAgent` |
| [`http_agent.py`](http_agent.py) | Test an HTTP endpoint with `HttpAgent` |
| [`websocket_agent.py`](websocket_agent.py) | Test a WebSocket endpoint with `WebSocketAgent` |
| [`custom_parser.py`](custom_parser.py) | Parse custom response structures with `JsonResponseParser` |
| [`caching.py`](caching.py) | Cache synthesized audio with `CachedSynthesizer` and `AudioCache` |
| [`pytest_integration/`](pytest_integration/) | Full pytest setup with markers, fixtures, and CLI options |
| [`config_driven/`](config_driven/) | YAML-driven pipeline using `russo` CLI or programmatic config loader |
| [`websocket_testing/`](websocket_testing/) | End-to-end WebSocket agent testing with Gemini |

## Running

Most examples are async scripts you can run directly:

```bash
python examples/basic_pipeline.py
python examples/caching.py
```

For the pytest integration example:

```bash
cd examples/pytest_integration
pytest -v
```

For the config-driven example:

```bash
# Via CLI
russo --config examples/config_driven/config.yaml --report report.json

# Or programmatically
python examples/config_driven/run.py
```
