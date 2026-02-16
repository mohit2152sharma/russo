"""Minimal russo.run() example.

Demonstrates the core pipeline: synthesize audio -> send to agent -> evaluate tool calls.
Uses a fake agent so it runs without API keys.
"""

import asyncio

import russo
from russo.evaluators import ExactEvaluator


# A fake agent that always returns a book_flight tool call.
# In real usage you'd use GeminiLiveAgent, OpenAIAgent, etc.
@russo.agent
async def fake_agent(audio: russo.Audio) -> russo.AgentResponse:
    return russo.AgentResponse(
        tool_calls=[
            russo.ToolCall(name="book_flight", arguments={"from_city": "Berlin", "to_city": "Rome"}),
        ]
    )


# A fake synthesizer for this demo (avoids TTS API calls).
class FakeSynthesizer:
    async def synthesize(self, text: str) -> russo.Audio:
        return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)


async def main():
    result = await russo.run(
        prompt="Book a flight from Berlin to Rome for tomorrow",
        synthesizer=FakeSynthesizer(),
        agent=fake_agent,
        evaluator=ExactEvaluator(),
        expect=[
            russo.tool_call("book_flight", from_city="Berlin", to_city="Rome"),
        ],
    )

    # Print human-readable summary
    print(result.summary())
    # PASSED (100% match rate)
    #   [+] book_flight({'from_city': 'Berlin', 'to_city': 'Rome'}) -> book_flight(...)

    # Programmatic checks
    assert result.passed
    assert result.match_rate == 1.0

    # Or use the assertion helper (raises ToolCallAssertionError on failure)
    russo.assert_tool_calls(result)

    print("\nAll checks passed!")


if __name__ == "__main__":
    asyncio.run(main())
