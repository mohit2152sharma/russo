"""Concurrent pipeline execution — run_concurrent() example.

Demonstrates the three concurrent execution scenarios:
  1. Single prompt, multiple runs  (reliability / flakiness testing)
  2. Multiple prompts, single run  (variant testing)
  3. Multiple prompts × multiple runs  (full matrix)

Uses fake agents so it runs without API keys.
"""

import asyncio

import russo
from russo.evaluators import ExactEvaluator


# ---------------------------------------------------------------------------
# Fakes — replace with real adapters in production
# ---------------------------------------------------------------------------
class FakeSynthesizer:
    async def synthesize(self, text: str) -> russo.Audio:
        return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)


@russo.agent
async def fake_agent(audio: russo.Audio) -> russo.AgentResponse:
    return russo.AgentResponse(
        tool_calls=[russo.ToolCall(name="book_flight", arguments={"from_city": "NYC", "to_city": "LA"})]
    )


async def main():
    synth = FakeSynthesizer()
    evaluator = ExactEvaluator(ignore_extra_args=True)
    expect = [russo.tool_call("book_flight")]

    # ----- Scenario 1: Single prompt, 5 runs -----
    print("=== Scenario 1: Single prompt, 5 runs ===")
    result = await russo.run_concurrent(
        prompts="Book a flight from NYC to LA",
        synthesizer=synth,
        agent=fake_agent,
        evaluator=evaluator,
        expect=expect,
        runs=5,
    )
    print(result.summary())
    assert result.passed
    print(f"  pass_rate={result.pass_rate:.0%}  match_rate={result.match_rate:.0%}\n")

    # ----- Scenario 2: Multiple prompts, 1 run each -----
    print("=== Scenario 2: Multiple prompts, 1 run each ===")
    result = await russo.run_concurrent(
        prompts=[
            "Book a flight from NYC to LA",
            "I need to fly from Chicago to Miami",
            "Please book me a flight from Boston to Denver",
        ],
        synthesizer=synth,
        agent=fake_agent,
        evaluator=evaluator,
        expect=expect,
    )
    print(result.summary())
    assert result.total == 3
    print()

    # ----- Scenario 3: Multiple prompts × multiple runs -----
    print("=== Scenario 3: 2 prompts × 3 runs (6 total) ===")
    result = await russo.run_concurrent(
        prompts=[
            "Book a flight from Berlin to Rome",
            "Fly me from Tokyo to Sydney",
        ],
        synthesizer=synth,
        agent=fake_agent,
        evaluator=evaluator,
        expect=expect,
        runs=3,
        max_concurrency=4,  # at most 4 simultaneous pipeline runs
    )
    print(result.summary())
    assert result.total == 6
    print()

    # ----- Inspecting individual results -----
    print("=== Inspecting individual run results ===")
    for run in result.runs:
        icon = "+" if run.eval_result.passed else "-"
        print(f"  [{icon}] prompt={run.prompt!r}  run_index={run.run_index}")

    print("\nAll scenarios passed!")


if __name__ == "__main__":
    asyncio.run(main())
