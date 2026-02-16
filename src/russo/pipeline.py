from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Iterable

from russo.interfaces import (
    AudioResponseEvaluator,
    AudioSampleGenerator,
    AudioStreamSource,
    ExpectationMatcher,
    ModelSessionAdapter,
    TestRunner,
    ToolCallRecorder,
)
from russo.models import (
    EvaluationResult,
    ModelRunContext,
    TestCaseResult,
    TestCaseSpec,
    TestRunReport,
    new_report,
)


@dataclass(frozen=True)
class PipelineDependencies:
    sample_generator: AudioSampleGenerator
    stream_source: AudioStreamSource
    model_adapter: ModelSessionAdapter
    tool_recorder: ToolCallRecorder
    matcher: ExpectationMatcher
    audio_evaluator: AudioResponseEvaluator | None = None


class DefaultTestRunner(TestRunner):
    def __init__(self, deps: PipelineDependencies) -> None:
        self._deps = deps

    async def run(self, test_case: TestCaseSpec) -> TestCaseResult:
        samples = self._deps.sample_generator.generate(test_case.audio_spec)
        if not samples:
            raise ValueError("AudioSampleGenerator returned no samples.")

        tool_calls = []
        audio_responses = []
        errors: list[str] = []

        for sample in samples:
            context = ModelRunContext(
                instructions=test_case.instructions, tools=test_case.tools
            )
            audio_stream = self._deps.stream_source.stream(sample)
            result = await self._deps.model_adapter.run(audio_stream, context)
            tool_calls.extend(result.tool_calls)
            audio_responses.extend(result.audio_responses)
            errors.extend(result.errors)

            for call in result.tool_calls:
                await self._deps.tool_recorder.record(call)

        recorded_calls = self._deps.tool_recorder.get_tool_calls()
        tool_eval = self._deps.matcher.match(test_case.tool_expectation, recorded_calls)

        audio_eval = None
        if test_case.audio_expectation and self._deps.audio_evaluator:
            audio_eval = self._deps.audio_evaluator.evaluate(
                test_case.audio_expectation, audio_responses
            )
        elif test_case.audio_expectation:
            audio_eval = EvaluationResult(
                passed=False,
                errors=[
                    "AudioResponseExpectation provided but no AudioResponseEvaluator configured."
                ],
            )

        return TestCaseResult(
            test_id=test_case.id,
            tool_call_result=tool_eval,
            audio_result=audio_eval,
            tool_calls=recorded_calls,
            audio_responses=audio_responses,
            errors=errors,
        )

    async def run_many(
        self,
        test_cases: Iterable[TestCaseSpec],
        *,
        runs: int = 1,
        max_concurrency: int | None = None,
    ) -> TestRunReport:
        """Run test cases concurrently, each repeated *runs* times.

        When ``runs=1`` (default) this behaves like the original sequential loop
        but executes all test cases concurrently.  With ``runs > 1`` each test
        case is launched *runs* times for reliability / flakiness testing.
        """
        cases = list(test_cases)
        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

        async def _guarded_run(tc: TestCaseSpec) -> TestCaseResult:
            if semaphore:
                async with semaphore:
                    return await self.run(tc)
            return await self.run(tc)

        tasks = [_guarded_run(tc) for tc in cases for _ in range(runs)]
        results = list(await asyncio.gather(*tasks))
        run_id = uuid.uuid4().hex
        return new_report(run_id, results)
