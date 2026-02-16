"""Custom agent example using the @russo.agent decorator.

The decorator turns any async function with the signature
    (Audio) -> AgentResponse
into a valid Agent that russo.run() can use.

This is useful when your agent is a function call, a REST API wrapper,
or anything that doesn't fit the built-in adapters.
"""

import asyncio

import russo
from russo.evaluators import ExactEvaluator


@russo.agent
async def my_travel_agent(audio: russo.Audio) -> russo.AgentResponse:
    """Simulate calling your own backend with the audio bytes."""
    # In a real scenario you'd do something like:
    #   response = await httpx.post("https://my-api/voice", content=audio.data)
    #   parsed = response.json()
    #   tool_calls = [russo.ToolCall(name=tc["name"], arguments=tc["args"]) for tc in parsed["tool_calls"]]

    # For this demo, return a hard-coded response:
    return russo.AgentResponse(
        tool_calls=[
            russo.ToolCall(name="search_hotels", arguments={"city": "Paris", "checkin": "2026-03-01"}),
        ]
    )


class FakeSynthesizer:
    async def synthesize(self, text: str) -> russo.Audio:
        return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)


async def main():
    result = await russo.run(
        prompt="Find hotels in Paris for March first",
        synthesizer=FakeSynthesizer(),
        agent=my_travel_agent,
        evaluator=ExactEvaluator(),
        expect=[
            russo.tool_call("search_hotels", city="Paris", checkin="2026-03-01"),
        ],
    )

    print(result.summary())
    russo.assert_tool_calls(result)
    print("\nCustom agent test passed!")


if __name__ == "__main__":
    asyncio.run(main())
