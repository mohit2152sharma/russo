from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from russo.config import build_component, build_registry, load_config
from russo.models import TestRunReport
from russo.pipeline import DefaultTestRunner, PipelineDependencies


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="russo", description="Audio LLM test runner")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config")
    parser.add_argument("--report", help="Path to write report JSON")
    return parser.parse_args()


async def _run_from_config(config_path: str, report_path: str | None) -> TestRunReport:
    config = load_config(config_path)
    registry = build_registry(config)

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

    runner = DefaultTestRunner(deps)
    report = await runner.run_many(config.suite.tests)

    if report_path:
        Path(report_path).write_text(json.dumps(report.to_dict(), indent=2))
    return report


def main() -> int:
    args = _parse_args()
    asyncio.run(_run_from_config(args.config, args.report))
    return 0


__all__ = ["main"]
