"""Tests for concurrent pipeline execution (run_concurrent, BatchResult)."""

from __future__ import annotations

import asyncio

import pytest

from russo._pipeline import run_concurrent
from russo._types import (
    AgentResponse,
    Audio,
    BatchResult,
    EvalResult,
    SingleRunResult,
    ToolCall,
)
from russo.evaluators.exact import ExactEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class FakeSynthesizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> Audio:
        self.calls.append(text)
        return Audio(data=b"fake", format="wav", sample_rate=24000)


class FakeAgent:
    """Agent that always returns a book_flight tool call."""

    def __init__(self, tool_name: str = "book_flight", delay: float = 0) -> None:
        self.tool_name = tool_name
        self.delay = delay
        self.call_count = 0

    async def run(self, audio: Audio) -> AgentResponse:
        if self.delay:
            await asyncio.sleep(self.delay)
        self.call_count += 1
        return AgentResponse(tool_calls=[ToolCall(name=self.tool_name, arguments={"from": "A", "to": "B"})])


class FailingAgent:
    """Agent that alternates between correct and wrong tool calls."""

    def __init__(self) -> None:
        self.call_count = 0

    async def run(self, audio: Audio) -> AgentResponse:
        self.call_count += 1
        if self.call_count % 2 == 0:
            return AgentResponse(tool_calls=[ToolCall(name="wrong_tool")])
        return AgentResponse(tool_calls=[ToolCall(name="book_flight")])


# ---------------------------------------------------------------------------
# BatchResult model tests
# ---------------------------------------------------------------------------
class TestBatchResult:
    def test_empty(self) -> None:
        br = BatchResult(runs=[])
        assert br.total == 0
        assert br.passed is True
        assert br.pass_rate == 1.0
        assert br.match_rate == 1.0

    def test_all_passed(self) -> None:
        runs = [
            SingleRunResult(
                prompt="p1",
                run_index=i,
                eval_result=EvalResult(
                    passed=True,
                    expected=[ToolCall(name="t")],
                    actual=[ToolCall(name="t")],
                ),
            )
            for i in range(3)
        ]
        br = BatchResult(runs=runs)
        assert br.total == 3
        assert br.passed is True
        assert br.pass_rate == 1.0
        assert br.passed_count == 3
        assert br.failed_count == 0

    def test_partial_pass(self) -> None:
        runs = [
            SingleRunResult(
                prompt="p1",
                run_index=0,
                eval_result=EvalResult(passed=True, expected=[], actual=[]),
            ),
            SingleRunResult(
                prompt="p1",
                run_index=1,
                eval_result=EvalResult(passed=False, expected=[ToolCall(name="t")], actual=[]),
            ),
        ]
        br = BatchResult(runs=runs)
        assert br.total == 2
        assert br.passed is False
        assert br.pass_rate == 0.5
        assert br.passed_count == 1
        assert br.failed_count == 1

    def test_summary_grouped_by_prompt(self) -> None:
        runs = [
            SingleRunResult(
                prompt="prompt_a",
                run_index=0,
                eval_result=EvalResult(passed=True, expected=[], actual=[]),
            ),
            SingleRunResult(
                prompt="prompt_b",
                run_index=0,
                eval_result=EvalResult(passed=False, expected=[ToolCall(name="t")], actual=[]),
            ),
        ]
        br = BatchResult(runs=runs)
        s = br.summary()
        assert "prompt_a" in s
        assert "prompt_b" in s
        assert "FAILED" in s


# ---------------------------------------------------------------------------
# run_concurrent tests
# ---------------------------------------------------------------------------
class TestRunConcurrent:
    """Unit tests for run_concurrent — no real APIs."""

    async def test_single_prompt_single_run(self) -> None:
        """Degenerate case: 1 prompt, 1 run — should produce a BatchResult with 1 entry."""
        synth = FakeSynthesizer()
        agent = FakeAgent()
        evaluator = ExactEvaluator(ignore_extra_args=True)

        result = await run_concurrent(
            prompts="hello",
            synthesizer=synth,
            agent=agent,
            evaluator=evaluator,
            expect=[ToolCall(name="book_flight")],
            runs=1,
        )

        assert isinstance(result, BatchResult)
        assert result.total == 1
        assert result.passed is True
        assert len(synth.calls) == 1

    async def test_single_prompt_multiple_runs(self) -> None:
        """Scenario 1: same prompt, N runs."""
        synth = FakeSynthesizer()
        agent = FakeAgent()
        evaluator = ExactEvaluator(ignore_extra_args=True)

        result = await run_concurrent(
            prompts="hello",
            synthesizer=synth,
            agent=agent,
            evaluator=evaluator,
            expect=[ToolCall(name="book_flight")],
            runs=5,
        )

        assert isinstance(result, BatchResult)
        assert result.total == 5
        assert result.passed is True
        assert result.pass_rate == 1.0
        assert agent.call_count == 5

    async def test_multiple_prompts_single_run(self) -> None:
        """Scenario 2: different prompts, 1 run each."""
        synth = FakeSynthesizer()
        agent = FakeAgent()
        evaluator = ExactEvaluator(ignore_extra_args=True)

        prompts = ["prompt_a", "prompt_b", "prompt_c"]
        result = await run_concurrent(
            prompts=prompts,
            synthesizer=synth,
            agent=agent,
            evaluator=evaluator,
            expect=[ToolCall(name="book_flight")],
            runs=1,
        )

        assert result.total == 3
        assert result.passed is True
        assert set(synth.calls) == {"prompt_a", "prompt_b", "prompt_c"}

    async def test_multiple_prompts_multiple_runs(self) -> None:
        """Scenario 3: M prompts × N runs."""
        synth = FakeSynthesizer()
        agent = FakeAgent()
        evaluator = ExactEvaluator(ignore_extra_args=True)

        result = await run_concurrent(
            prompts=["p1", "p2"],
            synthesizer=synth,
            agent=agent,
            evaluator=evaluator,
            expect=[ToolCall(name="book_flight")],
            runs=3,
        )

        assert result.total == 6  # 2 prompts × 3 runs
        assert result.passed is True
        assert agent.call_count == 6

    async def test_partial_failure(self) -> None:
        """Some runs fail, BatchResult reflects partial pass."""
        synth = FakeSynthesizer()
        agent = FailingAgent()
        evaluator = ExactEvaluator(ignore_extra_args=True)

        result = await run_concurrent(
            prompts="test",
            synthesizer=synth,
            agent=agent,
            evaluator=evaluator,
            expect=[ToolCall(name="book_flight")],
            runs=4,
        )

        assert result.total == 4
        assert not result.passed  # not all passed
        assert 0 < result.pass_rate < 1.0

    async def test_max_concurrency(self) -> None:
        """Verify max_concurrency limits simultaneous runs."""
        concurrent_count = 0
        max_seen = 0

        class TrackedAgent:
            async def run(self, audio: Audio) -> AgentResponse:
                nonlocal concurrent_count, max_seen
                concurrent_count += 1
                max_seen = max(max_seen, concurrent_count)
                await asyncio.sleep(0.05)
                concurrent_count -= 1
                return AgentResponse(tool_calls=[ToolCall(name="book_flight")])

        synth = FakeSynthesizer()
        evaluator = ExactEvaluator(ignore_extra_args=True)

        result = await run_concurrent(
            prompts="test",
            synthesizer=synth,
            agent=TrackedAgent(),
            evaluator=evaluator,
            expect=[ToolCall(name="book_flight")],
            runs=10,
            max_concurrency=2,
        )

        assert result.total == 10
        assert result.passed is True
        assert max_seen <= 2

    async def test_runs_are_concurrent(self) -> None:
        """Verify that runs actually execute concurrently (not sequentially)."""
        synth = FakeSynthesizer()
        agent = FakeAgent(delay=0.1)
        evaluator = ExactEvaluator(ignore_extra_args=True)

        import time

        start = time.monotonic()
        result = await run_concurrent(
            prompts="test",
            synthesizer=synth,
            agent=agent,
            evaluator=evaluator,
            expect=[ToolCall(name="book_flight")],
            runs=5,
        )
        elapsed = time.monotonic() - start

        assert result.total == 5
        # 5 runs at 0.1s each sequentially = 0.5s. Concurrent should be ~0.1s.
        # Use generous threshold but well under sequential time.
        assert elapsed < 0.4, f"Runs should be concurrent, took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Integration test placeholder (requires real API)
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestRunConcurrentIntegration:
    """Integration tests for concurrent pipeline — real Gemini API."""

    async def test_single_prompt_multiple_runs(self, google_synth, gemini_live_agent) -> None:
        result = await run_concurrent(
            prompts="Book a flight from New York to Los Angeles",
            synthesizer=google_synth,
            agent=gemini_live_agent,
            evaluator=ExactEvaluator(ignore_extra_args=True),
            expect=[ToolCall(name="book_flight")],
            runs=3,
        )

        print(f"\n{result.summary()}")
        assert result.total == 3
        assert result.pass_rate >= 0.5, f"At least half should pass:\n{result.summary()}"

    async def test_multiple_prompts(self, google_synth, gemini_live_agent) -> None:
        result = await run_concurrent(
            prompts=[
                "Book a flight from Chicago to Miami",
                "I need to fly from Boston to Denver",
            ],
            synthesizer=google_synth,
            agent=gemini_live_agent,
            evaluator=ExactEvaluator(ignore_extra_args=True),
            expect=[ToolCall(name="book_flight")],
            runs=1,
        )

        print(f"\n{result.summary()}")
        assert result.total == 2
        assert result.pass_rate >= 0.5

    async def test_multiple_prompts_multiple_runs(self, google_synth, gemini_live_agent) -> None:
        result = await run_concurrent(
            prompts=[
                "Book a flight from Tokyo to Sydney",
                "I want to fly from Berlin to Rome",
            ],
            synthesizer=google_synth,
            agent=gemini_live_agent,
            evaluator=ExactEvaluator(ignore_extra_args=True),
            expect=[ToolCall(name="book_flight")],
            runs=2,
            max_concurrency=2,
        )

        print(f"\n{result.summary()}")
        assert result.total == 4  # 2 prompts × 2 runs
        assert result.pass_rate >= 0.5
