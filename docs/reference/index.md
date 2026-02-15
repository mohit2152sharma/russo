# API Reference

Complete API documentation for russo, auto-generated from docstrings.

## Core

The main public API surface:

- [`russo`](core/init.md) — top-level exports
- [`Types`](core/types.md) — `Audio`, `ToolCall`, `AgentResponse`, `EvalResult`, `ToolCallMatch`
- [`Pipeline`](core/pipeline.md) — `russo.run()`
- [`Protocols`](core/protocols.md) — `Synthesizer`, `Agent`, `Evaluator`, `ResponseParser`
- [`Cache`](core/cache.md) — `AudioCache`, `CachedSynthesizer`
- [`Assertions`](core/assertions.md) — `assert_tool_calls`, `ToolCallAssertionError`

## Adapters

Agent adapters for different invocation styles:

- [`Gemini`](adapters/gemini.md) — `GeminiAgent`, `GeminiLiveAgent`
- [`OpenAI`](adapters/openai.md) — `OpenAIAgent`, `OpenAIRealtimeAgent`
- [`HTTP`](adapters/http.md) — `HttpAgent`
- [`WebSocket`](adapters/websocket.md) — `WebSocketAgent`
- [`Callable`](adapters/callable.md) — `CallableAgent`, `@agent` decorator

## Synthesizers

- [`Google`](synthesizers/google.md) — `GoogleSynthesizer`

## Evaluators

- [`Exact`](evaluators/exact.md) — `ExactEvaluator`

## Parsers

- [`Gemini`](parsers/gemini.md) — `GeminiResponseParser`
- [`OpenAI`](parsers/openai.md) — `OpenAIResponseParser`

## Integrations

- [`pytest Plugin`](pytest-plugin.md) — markers, fixtures, CLI options
