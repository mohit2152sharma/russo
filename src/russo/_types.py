"""Core data types for russo.

All data flowing through the pipeline is a Pydantic model,
giving us validation, serialization, and rich repr for free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class Audio(BaseModel):
    """Audio data with format metadata."""

    data: bytes
    format: Literal["wav", "mp3", "pcm", "ogg"] = "wav"
    sample_rate: int = 24000
    channels: int = 1
    sample_width: int = 2  # bytes per sample (16-bit = 2)

    def save(self, path: str | Path) -> Path:
        """Save audio to a file. Wraps raw PCM in a WAV container if needed.

        Usage:
            audio.save("output.wav")
        """
        import wave

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if p.suffix.lower() == ".wav":
            with wave.open(str(p), "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.sample_width)
                wf.setframerate(self.sample_rate)
                wf.writeframes(self.data)
        else:
            # For non-WAV formats, write raw bytes
            p.write_bytes(self.data)
        return p


class ToolCall(BaseModel):
    """A normalized tool/function call representation.

    Provider-agnostic — parsers convert provider-specific formats into this.
    """

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCall):
            return NotImplemented
        return self.name == other.name and self.arguments == other.arguments

    def __hash__(self) -> int:
        return hash((self.name, tuple(sorted(self.arguments.items()))))


class AgentResponse(BaseModel):
    """Normalized response from an agent, containing extracted tool calls."""

    tool_calls: list[ToolCall] = Field(default_factory=list)
    raw: Any | None = None
    """The raw, unparsed response from the provider (for debugging)."""

    model_config = {"arbitrary_types_allowed": True}


class ToolCallMatch(BaseModel):
    """Result of comparing a single expected tool call against actuals."""

    expected: ToolCall
    actual: ToolCall | None = None
    matched: bool = False
    details: str = ""


class EvalResult(BaseModel):
    """Full evaluation result for a test scenario."""

    passed: bool
    expected: list[ToolCall]
    actual: list[ToolCall]
    matches: list[ToolCallMatch] = Field(default_factory=list)

    @property
    def match_rate(self) -> float:
        """Fraction of expected tool calls that matched."""
        if not self.expected:
            return 1.0
        matched = sum(1 for m in self.matches if m.matched)
        return matched / len(self.expected)

    def summary(self) -> str:
        """Human-readable summary of the evaluation."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"{status} ({self.match_rate:.0%} match rate)"]
        for m in self.matches:
            icon = "+" if m.matched else "-"
            actual_str = (
                f" -> {m.actual.name}({m.actual.arguments})"
                if m.actual
                else " -> (no match)"
            )
            lines.append(
                f"  [{icon}] {m.expected.name}({m.expected.arguments}){actual_str}"
            )
            if m.details:
                lines.append(f"      {m.details}")
        return "\n".join(lines)
