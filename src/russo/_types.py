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


class SingleRunResult(BaseModel):
    """Result of a single pipeline run within a batch."""

    prompt: str
    run_index: int
    eval_result: EvalResult


class BatchResult(BaseModel):
    """Aggregated results from running the pipeline multiple times.

    Covers three scenarios:
    - Single prompt, N runs (reliability testing)
    - Multiple prompts, 1 run each (variant testing)
    - Multiple prompts, N runs each (full matrix)
    """

    runs: list[SingleRunResult] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.runs)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.runs if r.eval_result.passed)

    @property
    def failed_count(self) -> int:
        return self.total - self.passed_count

    @property
    def passed(self) -> bool:
        """True only if every single run passed."""
        return all(r.eval_result.passed for r in self.runs)

    @property
    def pass_rate(self) -> float:
        """Fraction of runs that passed."""
        if not self.runs:
            return 1.0
        return self.passed_count / self.total

    @property
    def match_rate(self) -> float:
        """Average match rate across all runs."""
        if not self.runs:
            return 1.0
        return sum(r.eval_result.match_rate for r in self.runs) / self.total

    def summary(self) -> str:
        """Human-readable summary grouped by prompt."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"{status} ({self.pass_rate:.0%} pass rate, {self.total} runs)",
            f"  Passed: {self.passed_count}/{self.total}",
        ]

        prompts: dict[str, list[SingleRunResult]] = {}
        for r in self.runs:
            prompts.setdefault(r.prompt, []).append(r)

        for prompt, results in prompts.items():
            prompt_passed = sum(1 for r in results if r.eval_result.passed)
            lines.append(f"  Prompt: {prompt!r}")
            lines.append(f"    {prompt_passed}/{len(results)} passed")
            for r in results:
                icon = "+" if r.eval_result.passed else "-"
                lines.append(
                    f"    [{icon}] run {r.run_index}: {r.eval_result.match_rate:.0%} match"
                )

        return "\n".join(lines)
