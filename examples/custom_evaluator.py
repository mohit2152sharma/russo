"""Custom evaluator example — implement the Evaluator protocol.

russo uses structural subtyping (protocols), so you don't need to inherit
from anything. Just implement the right method signature:

    def evaluate(self, expected: list[ToolCall], actual: list[ToolCall]) -> EvalResult

This example builds a "fuzzy" evaluator that:
  - Matches tool names case-insensitively
  - Only checks that expected arguments are a subset of actual arguments
"""

import asyncio

import russo
from russo._types import EvalResult, ToolCall, ToolCallMatch


class FuzzyEvaluator:
    """Evaluator that does case-insensitive name matching and subset argument checking."""

    def evaluate(self, expected: list[ToolCall], actual: list[ToolCall]) -> EvalResult:
        matches: list[ToolCallMatch] = []
        remaining = list(actual)

        for exp in expected:
            match = self._find_match(exp, remaining)
            matches.append(match)
            if match.matched and match.actual in remaining:
                remaining.remove(match.actual)

        passed = all(m.matched for m in matches)
        return EvalResult(passed=passed, expected=expected, actual=actual, matches=matches)

    def _find_match(self, expected: ToolCall, candidates: list[ToolCall]) -> ToolCallMatch:
        for candidate in candidates:
            if self._is_match(expected, candidate):
                return ToolCallMatch(expected=expected, actual=candidate, matched=True)

        return ToolCallMatch(
            expected=expected,
            matched=False,
            details=f"No fuzzy match found for {expected.name}",
        )

    def _is_match(self, expected: ToolCall, actual: ToolCall) -> bool:
        # Case-insensitive name matching
        if expected.name.lower() != actual.name.lower():
            return False
        # Expected args must be a subset of actual args
        return all(actual.arguments.get(k) == v for k, v in expected.arguments.items())


# --- Demo ---


@russo.agent
async def agent_with_extra_args(audio: russo.Audio) -> russo.AgentResponse:
    return russo.AgentResponse(
        tool_calls=[
            russo.ToolCall(
                name="Book_Flight",  # different casing
                arguments={
                    "from_city": "NYC",
                    "to_city": "LA",
                    "airline": "Delta",  # extra argument
                    "class": "economy",  # extra argument
                },
            ),
        ]
    )


class FakeSynthesizer:
    async def synthesize(self, text: str) -> russo.Audio:
        return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)


async def main():
    result = await russo.run(
        prompt="Book a flight from NYC to LA",
        synthesizer=FakeSynthesizer(),
        agent=agent_with_extra_args,
        evaluator=FuzzyEvaluator(),  # our custom evaluator
        expect=[
            russo.tool_call("book_flight", from_city="NYC", to_city="LA"),
        ],
    )

    print(result.summary())
    russo.assert_tool_calls(result)
    print("\nFuzzy evaluator matched despite case difference and extra args!")


if __name__ == "__main__":
    asyncio.run(main())
