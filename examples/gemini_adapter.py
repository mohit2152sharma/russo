"""Gemini adapter examples — GeminiAgent and GeminiLiveAgent.

Shows how to test Gemini models for tool-call accuracy using both:
  - GeminiAgent: standard generate_content (request/response)
  - GeminiLiveAgent: Live API over WebSocket (streaming/real-time)

Requires: pip install russo
          export GOOGLE_API_KEY="your-api-key"
"""

import asyncio
import os

import russo
from russo.evaluators import ExactEvaluator
from russo.synthesizers import GoogleSynthesizer

# ---------------------------------------------------------------------------
# Tool declarations (Gemini format)
# ---------------------------------------------------------------------------
BOOK_FLIGHT_TOOL = {
    "function_declarations": [
        {
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
    ]
}


# ---------------------------------------------------------------------------
# Example 1: GeminiAgent (standard request/response)
# ---------------------------------------------------------------------------
async def example_gemini_agent():
    """Test a Gemini model via generate_content."""
    from google import genai

    from russo.adapters import GeminiAgent

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    agent = GeminiAgent(
        client=client,
        model="gemini-2.0-flash",
        tools=[BOOK_FLIGHT_TOOL],
        system_instruction=(
            "You are a travel assistant. When the user asks to book a flight, "
            "call the book_flight function."
        ),
    )

    synthesizer = GoogleSynthesizer(api_key=os.environ["GOOGLE_API_KEY"])

    result = await russo.run(
        prompt="Book a flight from Berlin to Rome for tomorrow",
        synthesizer=synthesizer,
        agent=agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("book_flight", from_city="Berlin", to_city="Rome")],
    )

    print("GeminiAgent result:")
    print(result.summary())


# ---------------------------------------------------------------------------
# Example 2: GeminiLiveAgent (streaming Live API)
# ---------------------------------------------------------------------------
async def example_gemini_live_agent():
    """Test a Gemini model via the Live API (real-time streaming)."""
    from google import genai

    from russo.adapters import GeminiLiveAgent

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    agent = GeminiLiveAgent(
        client=client,
        model="gemini-live-2.5-flash-preview",  # Google AI model name
        tools=[BOOK_FLIGHT_TOOL],
        system_instruction=(
            "You are a travel assistant. When the user asks to book a flight, "
            "call the book_flight function."
        ),
    )

    synthesizer = GoogleSynthesizer(api_key=os.environ["GOOGLE_API_KEY"])

    result = await russo.run(
        prompt="Book a flight from NYC to LA",
        synthesizer=synthesizer,
        agent=agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
    )

    print("GeminiLiveAgent result:")
    print(result.summary())


# ---------------------------------------------------------------------------
# Example 3: Vertex AI variant
# ---------------------------------------------------------------------------
async def example_vertex_ai():
    """Same agents, but authenticated via Vertex AI (ADC / service account)."""
    from google import genai

    from russo.adapters import GeminiLiveAgent

    # Vertex AI client — uses Application Default Credentials
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "my-project"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )

    agent = GeminiLiveAgent(
        client=client,
        model="gemini-2.0-flash-live-preview-04-09",  # Vertex AI model name
        tools=[BOOK_FLIGHT_TOOL],
    )

    synthesizer = GoogleSynthesizer(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "my-project"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )

    result = await russo.run(
        prompt="Book a flight from London to Paris",
        synthesizer=synthesizer,
        agent=agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("book_flight", from_city="London", to_city="Paris")],
    )

    print("Vertex AI result:")
    print(result.summary())


if __name__ == "__main__":
    # Run whichever example you have credentials for:
    asyncio.run(example_gemini_live_agent())
    # asyncio.run(example_gemini_agent())
    # asyncio.run(example_vertex_ai())
