# Config-Driven Pipeline

Define your entire test pipeline and test suite in a YAML file. Run it via the `russo` CLI or load it programmatically from Python.

!!! tip "Source files"
    [`examples/config_driven/`](https://github.com/mohit2152sharma/russo/tree/main/examples/config_driven)

## How it works

A config file has two sections:

1. **`pipeline`** -- references to Python classes for each pipeline component
2. **`tests`** -- list of test case specifications

Each component is referenced by an import path (`class_path`) and constructor parameters (`params`). The russo CLI resolves imports, instantiates components, and runs the tests.

## The config file

```yaml
# config.yaml

pipeline:
  sample_generator:
    name: tts_generator
    class_path: "russo.synthesizers.google.GoogleSynthesizer"
    params:
      voice: "Kore"
      model: "gemini-2.5-flash-preview-tts"

  stream_source:
    name: pcm_streamer
    class_path: "russo.synthesizers.google.GoogleSynthesizer"
    params: {}

  model_adapter:
    name: gemini_live
    class_path: "russo.adapters.gemini.GeminiLiveAgent"
    params:
      model: "gemini-live-2.5-flash-preview"

  tool_recorder:
    name: recorder
    class_path: "russo.evaluators.exact.ExactEvaluator"
    params: {}

  matcher:
    name: exact_matcher
    class_path: "russo.evaluators.exact.ExactEvaluator"
    params: {}

  # audio_evaluator is optional — omit to skip audio response evaluation

tests:
  - id: "book_flight_berlin_rome"
    description: "User asks to book a flight from Berlin to Rome"
    audio_spec:
      id: "tts_berlin_rome"
      generator: "tts_generator"
      parameters:
        text: "Book a flight from Berlin to Rome for tomorrow"
    instructions: >
      You are a travel assistant. When the user asks to book a flight,
      call the book_flight function with from_city and to_city.
    tools:
      - name: "book_flight"
        description: "Book a flight between two cities"
        json_schema:
          type: object
          properties:
            from_city:
              type: string
              description: "Departure city"
            to_city:
              type: string
              description: "Arrival city"
          required:
            - from_city
            - to_city
    tool_expectation:
      name: "book_flight"
      arguments:
        from_city: "Berlin"
        to_city: "Rome"

  - id: "get_weather_tokyo"
    description: "User asks about weather in Tokyo"
    audio_spec:
      id: "tts_weather_tokyo"
      generator: "tts_generator"
      parameters:
        text: "What's the weather like in Tokyo today?"
    instructions: >
      You are a helpful assistant with access to weather tools.
      When the user asks about weather, call get_weather with the city.
    tools:
      - name: "get_weather"
        description: "Get current weather for a city"
        json_schema:
          type: object
          properties:
            city:
              type: string
              description: "City name"
          required:
            - city
    tool_expectation:
      name: "get_weather"
      arguments:
        city: "Tokyo"

# Run each test 3 times concurrently (optional, default: 1)
runs: 3
# Limit to 2 simultaneous pipeline runs (optional, default: unlimited)
max_concurrency: 2
```

## Running via CLI

The simplest way -- just point the `russo` command at your config:

```bash
russo --config config.yaml

# Save the report as JSON
russo --config config.yaml --report report.json

# Override runs from the command line (takes precedence over config)
russo --config config.yaml --runs 5 --max-concurrency 3
```

## Running programmatically

For more control over component construction and result handling:

```python
import asyncio
import json
from pathlib import Path

from russo.config import build_component, build_registry, load_config
from russo.pipeline import DefaultTestRunner, PipelineDependencies


async def main():
    config_path = Path("config.yaml")

    # 1. Load configuration
    config = load_config(config_path)
    print(f"Loaded {len(config.suite.tests)} test case(s)")

    # 2. Build the component registry
    registry = build_registry(config)

    # 3. Instantiate pipeline components
    deps = PipelineDependencies(
        sample_generator=build_component(registry, config.pipeline.sample_generator),
        stream_source=build_component(registry, config.pipeline.stream_source),
        model_adapter=build_component(registry, config.pipeline.model_adapter),
        tool_recorder=build_component(registry, config.pipeline.tool_recorder),
        matcher=build_component(registry, config.pipeline.matcher),
        audio_evaluator=(
            build_component(registry, config.pipeline.audio_evaluator)
            if config.pipeline.audio_evaluator
            else None
        ),
    )

    # 4. Run all tests
    runner = DefaultTestRunner(deps)
    report = await runner.run_many(config.suite.tests)

    # 5. Print results
    print(f"Run ID: {report.run_id}")
    print(f"Summary: {report.summary}")
    for result in report.results:
        status = "PASS" if result.tool_call_result.passed else "FAIL"
        print(f"  [{status}] {result.test_id}")

    # 6. Save report
    Path("report.json").write_text(json.dumps(report.to_dict(), indent=2))


asyncio.run(main())
```

### What each step does

1. **`load_config()`** -- reads YAML or JSON, parses into `Config` with pipeline and test suite sections
2. **`build_registry()`** -- resolves `class_path` imports and registers them as named factories
3. **`build_component()`** -- instantiates each component from the registry, passing `params` as `**kwargs`
4. **`DefaultTestRunner`** -- orchestrates the pipeline: generate audio -> stream to model -> record tool calls -> match expectations
5. **`run_many()`** -- runs all test cases and returns a `TestRunReport` with per-case results

```bash
python examples/config_driven/run.py
```
