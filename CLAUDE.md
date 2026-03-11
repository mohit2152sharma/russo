# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**russo** is a pytest-compatible testing framework for verifying that LLM agents make correct tool calls when given audio (or text) input. It's designed for testing voice AI tool-calling accuracy with audio caching, concurrent execution, and provider-agnostic extensibility.

## Commands

```bash
# Install all extras (development)
uv sync --all-extras

# Run tests
uv run pytest -v                          # All unit tests
uv run pytest tests/test_pipeline.py -v   # Specific file
uv run pytest -v -k "test_name"           # Match by name
uv run pytest -v -m "not integration"     # Skip integration tests
uv run pytest -v --integration            # Integration tests (requires API keys)

# Lint and format
uv run ruff check .                       # Check issues
uv run ruff check . --fix                 # Auto-fix
uv run ruff format --check .              # Check formatting
uv run ruff format .                      # Auto-format

# Docs
uv run mkdocs serve                       # Local preview at http://127.0.0.1:8000
uv run mkdocs build --strict              # CI-style build
```

## Architecture

The core pipeline flows through three composable, protocol-based components:

```
Text Prompt → Synthesizer.synthesize() → Audio
           → Agent.run(audio)          → AgentResponse (tool_calls)
           → Evaluator.evaluate()      → EvalResult
           → assert_tool_calls()       → pass/fail
```

**All extension points use structural typing (duck typing)** — no inheritance required. Any class implementing the right methods works.

### Key modules in `src/russo/`

| File | Purpose |
|------|---------|
| `_pipeline.py` | `run()` and `run_concurrent()` — the main entry points |
| `_types.py` | Core data types: `Audio`, `ToolCall`, `AgentResponse`, `EvalResult`, `BatchResult` |
| `_protocols.py` | Protocols: `Synthesizer`, `Agent`, `Evaluator`, `ResponseParser` |
| `_cache.py` | `AudioCache`, `CachedSynthesizer` — TTS result caching |
| `_assertions.py` | `assert_tool_calls()` |
| `_helpers.py` | `tool_call()` helper and `@agent` decorator |
| `pytest_plugin.py` | pytest integration (`@pytest.mark.russo`, fixtures, CLI flags) |
| `adapters/` | `GeminiAgent`, `GeminiLiveAgent`, `OpenAIAgent`, `HttpAgent`, `WebSocketAgent` |
| `synthesizers/google.py` | `GoogleSynthesizer` (TTS via Google AI) |
| `evaluators/exact.py` | `ExactEvaluator` |
| `parsers/` | `GeminiResponseParser`, `OpenAIResponseParser` |
| `audio/manager.py` | `AudioManager` — resampling, WAV handling |
| `report/terminal.py` | Terminal reporter |

### pytest Plugin CLI Flags

The pytest plugin (`pytest_plugin.py`) adds these flags:

```bash
--russo-cache / --russo-no-cache    # Toggle audio cache (default: enabled)
--russo-clear-cache                 # Clear cache before run
--russo-cache-dir PATH              # Custom cache directory
--russo-runs N                      # Runs per test (for reliability testing)
--russo-max-concurrency N           # Max simultaneous runs
--russo-report PATH                 # Generate HTML report
```

Tests use `@pytest.mark.russo(prompt="...", expect=[...])` and receive results via the `russo_result` fixture.

## Environment Variables

```
GOOGLE_API_KEY                    # Google AI API key
GOOGLE_APPLICATION_CREDENTIALS    # Vertex AI service account JSON path
GOOGLE_CLOUD_PROJECT              # GCP project ID
GOOGLE_CLOUD_LOCATION             # GCP region (e.g., us-central1)
```

Integration tests require at least `GOOGLE_API_KEY` or Vertex AI credentials.

## Build & Packaging

- **Build tool:** Hatchling
- **Package manager:** `uv`
- **Python:** 3.12+
- Optional extras: `[openai]`, `[ws]`, `[all]`

CI runs lint + unit tests on Python 3.12/3.13/3.14, then integration tests against both Google AI API key and Vertex AI (separate jobs in `.github/workflows/ci.yml`).
