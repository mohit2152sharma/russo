"""Pipeline orchestrator — chains Synthesizer -> Agent -> Evaluator."""

from __future__ import annotations

from russo._protocols import Agent, Evaluator, Synthesizer
from russo._types import EvalResult, ToolCall


async def run(
    *,
    prompt: str,
    synthesizer: Synthesizer,
    agent: Agent,
    evaluator: Evaluator,
    expect: list[ToolCall],
) -> EvalResult:
    """Run the full russo pipeline.

    1. Synthesize audio from the text prompt.
    2. Pass audio to the agent under test.
    3. Evaluate the agent's tool calls against expectations.

    Args:
        prompt: The text prompt to synthesize into audio.
        synthesizer: Converts text to audio.
        agent: The agent under test.
        evaluator: Compares expected vs actual tool calls.
        expect: The expected tool calls.

    Returns:
        EvalResult with pass/fail and per-call match details.
    """
    audio = await synthesizer.synthesize(prompt)
    response = await agent.run(audio)
    return evaluator.evaluate(expected=expect, actual=response.tool_calls)
