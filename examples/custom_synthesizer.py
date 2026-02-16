"""Custom synthesizer example — implement the Synthesizer protocol.

Just implement:
    async def synthesize(self, text: str) -> Audio

This example builds two custom synthesizers:
  1. FileSynthesizer — reads pre-recorded audio files from disk
  2. SilenceSynthesizer — generates silent audio (useful for testing the pipeline itself)

These are useful for offline testing without incurring TTS API costs.
"""

import asyncio
import hashlib
from pathlib import Path

import russo
from russo.evaluators import ExactEvaluator


class FileSynthesizer:
    """Reads pre-recorded audio files from a directory.

    Files are looked up by a SHA-256 hash of the prompt text.
    Useful when you have a fixed set of test prompts with pre-recorded audio.
    """

    def __init__(self, audio_dir: str | Path, format: str = "wav", sample_rate: int = 24000) -> None:
        self.audio_dir = Path(audio_dir)
        self.format = format
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> russo.Audio:
        key = hashlib.sha256(text.encode()).hexdigest()[:16]
        audio_path = self.audio_dir / f"{key}.{self.format}"
        if not audio_path.exists():
            raise FileNotFoundError(f"No pre-recorded audio for prompt (key={key}): {text!r}")
        return russo.Audio(
            data=audio_path.read_bytes(),
            format=self.format,
            sample_rate=self.sample_rate,
        )


class SilenceSynthesizer:
    """Generates silent audio of a fixed duration.

    Great for integration tests where the audio content doesn't matter
    (e.g. testing that the pipeline wiring works correctly).
    """

    def __init__(self, duration_seconds: float = 1.0, sample_rate: int = 24000) -> None:
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> russo.Audio:
        num_samples = int(self.sample_rate * self.duration_seconds)
        silent_pcm = b"\x00\x00" * num_samples  # 16-bit silence
        return russo.Audio(data=silent_pcm, format="wav", sample_rate=self.sample_rate)


# --- Demo ---

@russo.agent
async def fake_agent(audio: russo.Audio) -> russo.AgentResponse:
    return russo.AgentResponse(
        tool_calls=[russo.ToolCall(name="get_weather", arguments={"city": "Tokyo"})]
    )


async def main():
    # Use the silence synthesizer for this demo
    result = await russo.run(
        prompt="What's the weather in Tokyo?",
        synthesizer=SilenceSynthesizer(duration_seconds=0.5),
        agent=fake_agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("get_weather", city="Tokyo")],
    )

    print(result.summary())
    russo.assert_tool_calls(result)
    print("\nCustom synthesizer test passed!")


if __name__ == "__main__":
    asyncio.run(main())
