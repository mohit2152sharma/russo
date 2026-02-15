from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RussoModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        ser_json_bytes="base64",
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RussoModel":
        return cls.model_validate(payload)


class AudioSampleSpec(RussoModel):
    id: str
    generator: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION


class AudioSample(RussoModel):
    id: str
    spec_id: str
    audio_format: str
    sample_rate_hz: int
    channels: int
    duration_ms: int | None = None
    pcm_bytes: bytes | None = None
    uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _check_data(self) -> "AudioSample":
        if self.pcm_bytes is None and self.uri is None:
            raise ValueError("AudioSample requires pcm_bytes or uri.")
        return self


class ToolDefinition(RussoModel):
    name: str
    description: str
    json_schema: dict[str, Any]
    schema_version: str = SCHEMA_VERSION


class ToolCall(RussoModel):
    name: str
    arguments: dict[str, Any]
    call_id: str | None = None
    provider: str | None = None
    timestamp_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


class ToolCallExpectation(RussoModel):
    name: str
    arguments: dict[str, Any] | None = None
    allow_extra_arguments: bool = True
    allow_additional_calls: bool = True
    schema_version: str = SCHEMA_VERSION


class AudioResponse(RussoModel):
    audio_format: str
    sample_rate_hz: int
    channels: int
    duration_ms: int | None = None
    pcm_bytes: bytes | None = None
    uri: str | None = None
    transcript: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _check_data(self) -> "AudioResponse":
        if self.pcm_bytes is None and self.uri is None:
            raise ValueError("AudioResponse requires pcm_bytes or uri.")
        return self


class AudioResponseExpectation(RussoModel):
    min_duration_ms: int | None = None
    max_duration_ms: int | None = None
    transcript_contains: list[str] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION


class EvaluationResult(RussoModel):
    passed: bool
    score: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION


class ModelRunContext(RussoModel):
    instructions: str
    tools: list[ToolDefinition]
    metadata: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float | None = None
    schema_version: str = SCHEMA_VERSION


class ModelRunResult(RussoModel):
    tool_calls: list[ToolCall] = Field(default_factory=list)
    audio_responses: list[AudioResponse] = Field(default_factory=list)
    raw_events: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION


class TestCaseSpec(RussoModel):
    id: str
    description: str
    audio_spec: AudioSampleSpec
    instructions: str
    tools: list[ToolDefinition]
    tool_expectation: ToolCallExpectation
    audio_expectation: AudioResponseExpectation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


class TestCaseResult(RussoModel):
    test_id: str
    tool_call_result: EvaluationResult
    audio_result: EvaluationResult | None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    audio_responses: list[AudioResponse] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION


class TestRunReport(RussoModel):
    run_id: str
    started_at: str
    ended_at: str
    results: list[TestCaseResult]
    summary: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


def build_summary(results: Iterable[TestCaseResult]) -> dict[str, Any]:
    total = 0
    passed = 0
    for result in results:
        total += 1
        if result.tool_call_result.passed and (
            result.audio_result is None or result.audio_result.passed
        ):
            passed += 1
    return {"total": total, "passed": passed, "failed": total - passed}


def new_report(run_id: str, results: list[TestCaseResult]) -> TestRunReport:
    started = _utc_now_iso()
    ended = _utc_now_iso()
    return TestRunReport(
        run_id=run_id,
        started_at=started,
        ended_at=ended,
        results=results,
        summary=build_summary(results),
    )
