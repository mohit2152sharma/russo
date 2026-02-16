"""Programmatic usage of the config-driven pipeline.

Instead of using the `russo` CLI, you can load a config file and
run the pipeline directly from Python. This gives you full control
over component construction and result handling.

Usage:
    python examples/config_driven/run.py
"""

import asyncio
import json
from pathlib import Path

from russo.config import build_component, build_registry, load_config
from russo.pipeline import DefaultTestRunner, PipelineDependencies


async def main():
    config_path = Path(__file__).parent / "config.yaml"

    # 1. Load configuration from YAML (or JSON)
    config = load_config(config_path)
    print(f"Loaded {len(config.suite.tests)} test case(s) from {config_path.name}")

    # 2. Build the component registry (resolves class_path imports)
    registry = build_registry(config)

    # 3. Instantiate pipeline components from the registry
    deps = PipelineDependencies(
        sample_generator=build_component(registry, config.pipeline.sample_generator),
        stream_source=build_component(registry, config.pipeline.stream_source),
        model_adapter=build_component(registry, config.pipeline.model_adapter),
        tool_recorder=build_component(registry, config.pipeline.tool_recorder),
        matcher=build_component(registry, config.pipeline.matcher),
        audio_evaluator=(
            build_component(registry, config.pipeline.audio_evaluator) if config.pipeline.audio_evaluator else None
        ),
    )

    # 4. Create the runner and execute all tests
    runner = DefaultTestRunner(deps)
    report = await runner.run_many(config.suite.tests)

    # 5. Print results
    print(f"\nRun ID: {report.run_id}")
    print(f"Summary: {report.summary}")
    for result in report.results:
        status = "PASS" if result.tool_call_result.passed else "FAIL"
        print(f"  [{status}] {result.test_id}")
        if result.errors:
            for err in result.errors:
                print(f"         Error: {err}")

    # 6. Optionally save the report as JSON
    report_path = Path(__file__).parent / "report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
