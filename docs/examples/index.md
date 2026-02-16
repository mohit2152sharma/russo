# Examples

Runnable, step-by-step examples for every major russo feature. Each example is also available as a standalone Python file in the [`examples/`](https://github.com/mohit2152sharma/russo/tree/main/examples) directory.

## Quick Reference

| Example | What it shows |
|---------|---------------|
| [Basic Pipeline](basic-pipeline.md) | Minimal `russo.run()` end-to-end pipeline |
| [Custom Agent](custom-agent.md) | Wrap any async function with `@russo.agent` |
| [Custom Evaluator](custom-evaluator.md) | Build a custom evaluator via structural subtyping |
| [Custom Synthesizer](custom-synthesizer.md) | Build a file-based or silence synthesizer for offline testing |
| [Gemini Adapters](gemini-adapter.md) | `GeminiAgent`, `GeminiLiveAgent`, and Vertex AI |
| [OpenAI Adapters](openai-adapter.md) | `OpenAIAgent` and `OpenAIRealtimeAgent` |
| [HTTP Agent](http-agent.md) | Test HTTP endpoints with `HttpAgent` |
| [WebSocket Agent](websocket-agent.md) | Test WebSocket endpoints with custom hooks |
| [Caching](caching.md) | `CachedSynthesizer` and `AudioCache` |
| [Concurrent Runs](concurrent-runs.md) | `russo.run_concurrent()` — multi-prompt and multi-run testing |
| [pytest Integration](pytest-integration.md) | Markers, fixtures, and CLI options |
| [Config-Driven Pipeline](config-driven.md) | YAML-driven pipeline via CLI or programmatic loader |
