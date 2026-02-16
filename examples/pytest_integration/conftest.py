"""pytest conftest.py — configure russo fixtures for your test suite.

Override these fixtures to plug in your real synthesizer and agent.
The russo pytest plugin auto-discovers these fixture names:
  - russo_synthesizer
  - russo_agent
  - russo_evaluator (optional, defaults to ExactEvaluator)
  - russo_audio_cache (optional, session-scoped)
"""

import os

import pytest

import russo
from russo.evaluators import ExactEvaluator


# ---------------------------------------------------------------------------
# Synthesizer fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def russo_synthesizer():
    """Create the TTS synthesizer used for all russo tests.

    Uses GoogleSynthesizer if GOOGLE_API_KEY is set, otherwise falls back
    to a fake synthesizer for offline testing.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        from russo.synthesizers import GoogleSynthesizer

        return GoogleSynthesizer(api_key=api_key)

    # Fallback: fake synthesizer for CI / offline use
    class FakeSynthesizer:
        async def synthesize(self, text: str) -> russo.Audio:
            return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)

    return FakeSynthesizer()


# ---------------------------------------------------------------------------
# Agent fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def russo_agent():
    """Create the agent under test.

    Replace this with your real agent. Examples:

        from russo.adapters import GeminiLiveAgent
        return GeminiLiveAgent(client=..., model="...", tools=[...])

        from russo.adapters import OpenAIAgent
        return OpenAIAgent(client=..., tools=[...])
    """

    # Demo: a fake agent that returns hard-coded tool calls
    @russo.agent
    async def fake_agent(audio: russo.Audio) -> russo.AgentResponse:
        return russo.AgentResponse(
            tool_calls=[
                russo.ToolCall(name="book_flight", arguments={"from_city": "NYC", "to_city": "LA"}),
            ]
        )

    return fake_agent


# ---------------------------------------------------------------------------
# Evaluator fixture (optional — defaults to ExactEvaluator)
# ---------------------------------------------------------------------------
@pytest.fixture
def russo_evaluator():
    """Override to use a custom evaluator."""
    return ExactEvaluator()
