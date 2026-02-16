from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from russo.models import (
    AudioResponseExpectation,
    AudioSampleSpec,
    TestCaseSpec,
    ToolCallExpectation,
    ToolDefinition,
)
from russo.registry import ComponentRegistry


@dataclass(frozen=True)
class ComponentRef:
    name: str
    class_path: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    sample_generator: ComponentRef
    stream_source: ComponentRef
    model_adapter: ComponentRef
    tool_recorder: ComponentRef
    matcher: ComponentRef
    audio_evaluator: ComponentRef | None = None


@dataclass(frozen=True)
class TestSuiteConfig:
    tests: list[TestCaseSpec]


@dataclass(frozen=True)
class Config:
    pipeline: PipelineConfig
    suite: TestSuiteConfig
    runs: int = 1
    max_concurrency: int | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML configs.") from exc
    return yaml.safe_load(path.read_text())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _parse_component(payload: dict[str, Any]) -> ComponentRef:
    return ComponentRef(
        name=payload["name"],
        class_path=payload.get("class_path"),
        params=payload.get("params", {}),
    )


def _parse_tool_definition(payload: dict[str, Any]) -> ToolDefinition:
    return ToolDefinition(
        name=payload["name"],
        description=payload.get("description", ""),
        json_schema=payload.get("json_schema", {}),
    )


def _parse_test_case(payload: dict[str, Any]) -> TestCaseSpec:
    audio_spec = AudioSampleSpec(**payload["audio_spec"])
    tools = [_parse_tool_definition(item) for item in payload.get("tools", [])]
    tool_expectation = ToolCallExpectation(**payload["tool_expectation"])
    audio_expectation = None
    if "audio_expectation" in payload:
        audio_expectation = AudioResponseExpectation(**payload["audio_expectation"])
    return TestCaseSpec(
        id=payload["id"],
        description=payload.get("description", ""),
        audio_spec=audio_spec,
        instructions=payload["instructions"],
        tools=tools,
        tool_expectation=tool_expectation,
        audio_expectation=audio_expectation,
        metadata=payload.get("metadata", {}),
    )


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if path.suffix in {".yaml", ".yml"}:
        payload = _load_yaml(path)
    else:
        payload = _load_json(path)

    pipeline = PipelineConfig(
        sample_generator=_parse_component(payload["pipeline"]["sample_generator"]),
        stream_source=_parse_component(payload["pipeline"]["stream_source"]),
        model_adapter=_parse_component(payload["pipeline"]["model_adapter"]),
        tool_recorder=_parse_component(payload["pipeline"]["tool_recorder"]),
        matcher=_parse_component(payload["pipeline"]["matcher"]),
        audio_evaluator=(
            _parse_component(payload["pipeline"]["audio_evaluator"])
            if payload["pipeline"].get("audio_evaluator")
            else None
        ),
    )
    suite = TestSuiteConfig(tests=[_parse_test_case(item) for item in payload["tests"]])
    runs = payload.get("runs", 1)
    max_concurrency = payload.get("max_concurrency")
    return Config(
        pipeline=pipeline, suite=suite, runs=runs, max_concurrency=max_concurrency
    )


def build_registry(config: Config) -> ComponentRegistry:
    registry = ComponentRegistry()
    for ref in [
        config.pipeline.sample_generator,
        config.pipeline.stream_source,
        config.pipeline.model_adapter,
        config.pipeline.tool_recorder,
        config.pipeline.matcher,
        config.pipeline.audio_evaluator,
    ]:
        if ref is None:
            continue
        if ref.class_path:
            registry.register_path(ref.name, ref.class_path)
    return registry


def build_component(registry: ComponentRegistry, ref: ComponentRef) -> Any:
    if ref.class_path and ref.name not in registry._registry:
        registry.register_path(ref.name, ref.class_path)
    return registry.build(ref.name, **ref.params)
