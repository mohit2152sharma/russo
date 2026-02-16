"""OpenAI adapter examples — OpenAIAgent and OpenAIRealtimeAgent.

Shows how to test OpenAI models for tool-call accuracy using:
  - OpenAIAgent: Chat Completions with audio input (gpt-4o-audio-preview)
  - OpenAIRealtimeAgent: Realtime API over WebSocket (gpt-4o-realtime-preview)

Requires: pip install "russo[openai]"
          export OPENAI_API_KEY="your-api-key"
"""

import asyncio
import os

import russo
from russo.evaluators import ExactEvaluator
from russo.synthesizers import GoogleSynthesizer

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI format)
# ---------------------------------------------------------------------------
BOOK_FLIGHT_TOOL = {
    "type": "function",
    "function": {
        "name": "book_flight",
        "description": "Book a flight between two cities.",
        "parameters": {
            "type": "object",
            "properties": {
                "from_city": {"type": "string", "description": "Departure city"},
                "to_city": {"type": "string", "description": "Arrival city"},
            },
            "required": ["from_city", "to_city"],
        },
    },
}

# For the Realtime API, tool format is slightly different
BOOK_FLIGHT_REALTIME_TOOL = {
    "type": "function",
    "name": "book_flight",
    "description": "Book a flight between two cities.",
    "parameters": {
        "type": "object",
        "properties": {
            "from_city": {"type": "string", "description": "Departure city"},
            "to_city": {"type": "string", "description": "Arrival city"},
        },
        "required": ["from_city", "to_city"],
    },
}


# ---------------------------------------------------------------------------
# Example 1: OpenAIAgent (Chat Completions with audio)
# ---------------------------------------------------------------------------
async def example_openai_agent():
    """Test an OpenAI model via Chat Completions."""
    from openai import AsyncOpenAI

    from russo.adapters import OpenAIAgent

    client = AsyncOpenAI()  # reads OPENAI_API_KEY from env

    agent = OpenAIAgent(
        client=client,
        model="gpt-4o-audio-preview",
        tools=[BOOK_FLIGHT_TOOL],
        system_prompt="You are a travel assistant. When the user asks to book a flight, call the book_flight function.",
    )

    # GoogleSynthesizer for TTS (or use any Synthesizer)
    synthesizer = GoogleSynthesizer(api_key=os.environ["GOOGLE_API_KEY"])

    result = await russo.run(
        prompt="Book a flight from Berlin to Rome for tomorrow",
        synthesizer=synthesizer,
        agent=agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("book_flight", from_city="Berlin", to_city="Rome")],
    )

    print("OpenAIAgent result:")
    print(result.summary())


# ---------------------------------------------------------------------------
# Example 2: OpenAIRealtimeAgent (Realtime API)
# ---------------------------------------------------------------------------
async def example_openai_realtime():
    """Test an OpenAI model via the Realtime API."""
    from openai import AsyncOpenAI

    from russo.adapters import OpenAIRealtimeAgent

    client = AsyncOpenAI()

    agent = OpenAIRealtimeAgent(
        client=client,
        model="gpt-4o-realtime-preview",
        tools=[BOOK_FLIGHT_REALTIME_TOOL],
        response_timeout=30.0,
    )

    synthesizer = GoogleSynthesizer(api_key=os.environ["GOOGLE_API_KEY"])

    result = await russo.run(
        prompt="Book a flight from NYC to LA",
        synthesizer=synthesizer,
        agent=agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
    )

    print("OpenAIRealtimeAgent result:")
    print(result.summary())


# ---------------------------------------------------------------------------
# Example 3: OpenAIRealtimeAgent with pre-existing connection
# ---------------------------------------------------------------------------
async def example_openai_realtime_preconnected():
    """Reuse a pre-existing Realtime connection for multiple tests."""
    from openai import AsyncOpenAI

    from russo.adapters import OpenAIRealtimeAgent

    client = AsyncOpenAI()

    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as conn:
        agent = OpenAIRealtimeAgent(connection=conn)

        synthesizer = GoogleSynthesizer(api_key=os.environ["GOOGLE_API_KEY"])

        result = await russo.run(
            prompt="Book a flight from London to Paris",
            synthesizer=synthesizer,
            agent=agent,
            evaluator=ExactEvaluator(),
            expect=[russo.tool_call("book_flight", from_city="London", to_city="Paris")],
        )

        print("Pre-connected Realtime result:")
        print(result.summary())


if __name__ == "__main__":
    asyncio.run(example_openai_agent())
    # asyncio.run(example_openai_realtime())
    # asyncio.run(example_openai_realtime_preconnected())
