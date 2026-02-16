from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Protocol

from russo.models import (
    AudioResponse,
    AudioResponseExpectation,
    AudioSample,
    AudioSampleSpec,
    EvaluationResult,
    ModelRunContext,
    ModelRunResult,
    TestCaseResult,
    TestCaseSpec,
    TestRunReport,
    ToolCall,
    ToolCallExpectation,
)


class AudioSampleGenerator(Protocol):
    def generate(self, spec: AudioSampleSpec) -> list[AudioSample]:
        """Create one or more audio samples from a spec."""


class AudioStreamSource(Protocol):
    async def stream(self, sample: AudioSample) -> AsyncIterator[bytes]:
        """Yield raw audio chunks for the model adapter."""


class ModelSessionAdapter(Protocol):
    async def run(
        self, audio_stream: AsyncIterator[bytes], context: ModelRunContext
    ) -> ModelRunResult:
        """Run a model session and return tool calls and audio responses."""


class ToolCallRecorder(Protocol):
    async def record(self, tool_call: ToolCall) -> None:
        """Record tool calls emitted by the model."""

    def get_tool_calls(self) -> list[ToolCall]:
        """Return recorded tool calls in time order."""


class ExpectationMatcher(Protocol):
    def match(
        self, expectation: ToolCallExpectation, tool_calls: Sequence[ToolCall]
    ) -> EvaluationResult:
        """Compare expected tool calls against observed tool calls."""


class AudioResponseEvaluator(Protocol):
    def evaluate(
        self, expectation: AudioResponseExpectation, responses: Sequence[AudioResponse]
    ) -> EvaluationResult:
        """Evaluate audio responses against an expectation."""


class TestRunner(Protocol):
    async def run(self, test_case: TestCaseSpec) -> TestCaseResult:
        """Run a single test case."""

    async def run_many(
        self,
        test_cases: Iterable[TestCaseSpec],
        *,
        runs: int = 1,
        max_concurrency: int | None = None,
    ) -> TestRunReport:
        """Run multiple test cases and return a report.

        Args:
            test_cases: The test cases to execute.
            runs: Number of times to run each test case concurrently.
            max_concurrency: Cap on simultaneous runs (None = unlimited).
        """


class Reporter(Protocol):
    def report(self, report: TestRunReport) -> None:
        """Emit a report to stdout, file, or external system."""

    def serialize(self, report: TestRunReport) -> str:
        """Serialize a report to a string."""


class BaseReporter(ABC):
    @abstractmethod
    def report(self, report: TestRunReport) -> None:
        raise NotImplementedError

    @abstractmethod
    def serialize(self, report: TestRunReport) -> str:
        raise NotImplementedError
