"""russo — testing framework for LLM tool-call accuracy."""

from russo import adapters, evaluators, parsers, synthesizers  # noqa: F401
from russo._assertions import ToolCallAssertionError, assert_tool_calls
from russo._cache import AudioCache, CachedSynthesizer
from russo._helpers import agent, tool_call
from russo._pipeline import run
from russo._types import AgentResponse, Audio, EvalResult, ToolCall, ToolCallMatch

__all__ = [
    # Data types
    "Audio",
    "ToolCall",
    "AgentResponse",
    "EvalResult",
    "ToolCallMatch",
    # Cache
    "AudioCache",
    "CachedSynthesizer",
    # Helpers
    "tool_call",
    "agent",
    # Pipeline
    "run",
    # Assertions
    "assert_tool_calls",
    "ToolCallAssertionError",
]
