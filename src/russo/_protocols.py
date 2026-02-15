"""Protocol definitions for russo extension points.

All extension points use typing.Protocol (structural subtyping).
Users never need to inherit — if the object has the right methods, it works.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from russo._types import AgentResponse, Audio, EvalResult, ToolCall


@runtime_checkable
class Synthesizer(Protocol):
    """Converts text into audio.

    Implement this to plug in any TTS provider (Google, OpenAI, ElevenLabs, etc.).
    """

    async def synthesize(self, text: str) -> Audio: ...


@runtime_checkable
class Agent(Protocol):
    """The agent under test.

    Takes audio input and returns a response containing tool calls.
    """

    async def run(self, audio: Audio) -> AgentResponse: ...


@runtime_checkable
class Evaluator(Protocol):
    """Compares expected tool calls against actual tool calls.

    Implement this for custom matching logic (exact, semantic, partial, etc.).
    """

    def evaluate(self, expected: list[ToolCall], actual: list[ToolCall]) -> EvalResult: ...


@runtime_checkable
class ResponseParser(Protocol):
    """Parses a provider-specific raw response into a normalized AgentResponse.

    Implement this for each LLM provider format (Gemini, OpenAI, Anthropic, etc.).
    """

    def parse(self, raw_response: Any) -> AgentResponse: ...
