"""Integration tests for the russo pipeline — synthesize → agent → evaluate.

Uses the real Gemini Live API via Vertex AI.
Run with: pytest tests/test_pipeline.py --integration -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import russo
from russo._pipeline import run
from russo._types import EvalResult, ToolCall
from russo.adapters.gemini import GeminiLiveAgent
from russo.evaluators.exact import ExactEvaluator

AUDIO_OUT = Path(__file__).parent / "audio_output"


@pytest.mark.integration
class TestPipelineIntegration:
    """Full pipeline: GoogleSynthesizer → GeminiLiveAgent → ExactEvaluator."""

    async def test_book_flight_pass(self, google_synth: Any, gemini_live_agent: GeminiLiveAgent) -> None:
        """Pipeline should PASS when the agent returns the expected tool call."""
        result = await run(
            prompt="Book a flight from New York to Los Angeles",
            synthesizer=google_synth,
            agent=gemini_live_agent,
            evaluator=ExactEvaluator(ignore_extra_args=True),
            expect=[ToolCall(name="book_flight")],
        )

        print(f"\n{result.summary()}")
        assert result.passed, f"Pipeline should pass:\n{result.summary()}"
        assert result.match_rate == 1.0

    async def test_book_flight_checks_args(self, google_synth: Any, gemini_live_agent: GeminiLiveAgent) -> None:
        """Pipeline should PASS with ignore_extra_args and partial arg match."""
        result = await run(
            prompt="Book a flight from Chicago to Miami",
            synthesizer=google_synth,
            agent=gemini_live_agent,
            evaluator=ExactEvaluator(ignore_extra_args=True),
            expect=[ToolCall(name="book_flight")],
        )

        print(f"\n{result.summary()}")
        assert result.passed
        # Verify the actual args contain city names
        actual = result.actual[0]
        assert "from_city" in actual.arguments
        assert "to_city" in actual.arguments

    async def test_wrong_tool_name_fails(self, google_synth: Any, gemini_live_agent: GeminiLiveAgent) -> None:
        """Pipeline should FAIL when we expect a tool the agent didn't call."""
        result = await run(
            prompt="Book a flight from London to Paris",
            synthesizer=google_synth,
            agent=gemini_live_agent,
            evaluator=ExactEvaluator(ignore_extra_args=True),
            expect=[ToolCall(name="get_weather")],
        )

        print(f"\n{result.summary()}")
        assert not result.passed, "Should fail — agent calls book_flight, not get_weather"

    async def test_save_audio_for_inspection(self, google_synth: Any, gemini_live_agent: GeminiLiveAgent) -> None:
        """Synthesize audio, save to disk, then run through the pipeline."""
        prompt = "Please book me a flight from San Francisco to Seattle"

        # Step 1: synthesize & save
        audio = await google_synth.synthesize(prompt)
        saved = audio.save(AUDIO_OUT / "pipeline_book_flight.wav")
        print(f"\n  Audio saved to: {saved}")
        assert saved.exists()
        assert saved.stat().st_size > 0

        # Step 2: run the agent on the saved audio
        response = await gemini_live_agent.run(audio)
        assert len(response.tool_calls) >= 1
        assert response.tool_calls[0].name == "book_flight"

        # Step 3: evaluate
        evaluator = ExactEvaluator(ignore_extra_args=True)
        result = evaluator.evaluate(
            expected=[ToolCall(name="book_flight")],
            actual=response.tool_calls,
        )
        print(f"  {result.summary()}")
        assert result.passed

    async def test_pipeline_with_russo_run(self, google_synth: Any, gemini_live_agent: GeminiLiveAgent) -> None:
        """Verify the top-level russo.run() entry-point works end to end."""
        result = await russo.run(
            prompt="I need to fly from Boston to Denver",
            synthesizer=google_synth,
            agent=gemini_live_agent,
            evaluator=ExactEvaluator(ignore_extra_args=True),
            expect=[russo.ToolCall(name="book_flight")],
        )

        print(f"\n{result.summary()}")
        assert isinstance(result, EvalResult)
        assert result.passed

    async def test_save_multiple_prompts(self, google_synth: Any, gemini_live_agent: GeminiLiveAgent) -> None:
        """Synthesize two different prompts, save both, verify both produce correct tool calls."""
        prompts = [
            ("Book a flight from Tokyo to Sydney", "pipeline_tokyo_sydney.wav"),
            ("I want to fly from Berlin to Rome", "pipeline_berlin_rome.wav"),
        ]

        for prompt, filename in prompts:
            audio = await google_synth.synthesize(prompt)
            saved = audio.save(AUDIO_OUT / filename)
            print(f"\n  [{filename}] saved ({saved.stat().st_size} bytes)")

            result = await run(
                prompt=prompt,
                synthesizer=google_synth,
                agent=gemini_live_agent,
                evaluator=ExactEvaluator(ignore_extra_args=True),
                expect=[ToolCall(name="book_flight")],
            )

            print(f"  [{filename}] {result.summary()}")
            assert result.passed, f"Failed for prompt: {prompt!r}\n{result.summary()}"
