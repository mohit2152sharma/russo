"""Pipeline orchestrator — chains Synthesizer -> Agent -> Evaluator."""

from __future__ import annotations

import asyncio

from russo._protocols import Agent, Evaluator, Synthesizer
from russo._types import BatchResult, EvalResult, SingleRunResult, ToolCall


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


async def run_concurrent(
    *,
    prompts: str | list[str],
    synthesizer: Synthesizer,
    agent: Agent,
    evaluator: Evaluator,
    expect: list[ToolCall],
    runs: int = 1,
    max_concurrency: int | None = None,
) -> BatchResult:
    """Run the pipeline concurrently for multiple prompts and/or multiple runs.

    Three scenarios:
        - Single prompt, N runs:  ``prompts="text", runs=N``
        - Multiple prompts, 1 run each:  ``prompts=["a", "b", "c"]``
        - Multiple prompts, N runs each:  ``prompts=["a", "b"], runs=N``

    Args:
        prompts: One or more text prompts to test.
        synthesizer: Converts text to audio.
        agent: The agent under test.
        evaluator: Compares expected vs actual tool calls.
        expect: The expected tool calls (same for every prompt).
        runs: Number of times to run each prompt (default 1).
        max_concurrency: Cap on simultaneous pipeline runs (``None`` = unlimited).

    Returns:
        BatchResult with per-run details and aggregate statistics.
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def _single_run(prompt: str, run_index: int) -> SingleRunResult:
        async def _do() -> SingleRunResult:
            result = await run(
                prompt=prompt,
                synthesizer=synthesizer,
                agent=agent,
                evaluator=evaluator,
                expect=expect,
            )
            return SingleRunResult(prompt=prompt, run_index=run_index, eval_result=result)

        if semaphore:
            async with semaphore:
                return await _do()
        return await _do()

    tasks = [_single_run(prompt, i) for prompt in prompts for i in range(runs)]
    results = await asyncio.gather(*tasks)
    return BatchResult(runs=list(results))
