# Custom Synthesizer

Build your own synthesizer for offline testing, pre-recorded audio, or alternative TTS providers. Like all russo extension points, you just implement the right method -- no inheritance required.

!!! tip "Source file"
    [`examples/custom_synthesizer.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/custom_synthesizer.py)

## The Synthesizer protocol

Any class with this method is a valid synthesizer:

```python
async def synthesize(self, text: str) -> Audio
```

## Example 1: File-based synthesizer

Reads pre-recorded audio files from a directory, keyed by a hash of the prompt text. Useful when you have a fixed set of test prompts with recorded audio.

```python
import hashlib
from pathlib import Path
import russo


class FileSynthesizer:
    def __init__(
        self,
        audio_dir: str | Path,
        format: str = "wav",
        sample_rate: int = 24000,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.format = format
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> russo.Audio:
        key = hashlib.sha256(text.encode()).hexdigest()[:16]
        audio_path = self.audio_dir / f"{key}.{self.format}"
        if not audio_path.exists():
            raise FileNotFoundError(
                f"No pre-recorded audio for prompt (key={key}): {text!r}"
            )
        return russo.Audio(
            data=audio_path.read_bytes(),
            format=self.format,
            sample_rate=self.sample_rate,
        )
```

**Usage:**

```python
synth = FileSynthesizer(audio_dir="test_audio/")
```

Pre-record your audio files named by their SHA-256 hash prefix (e.g. `a1b2c3d4e5f6g7h8.wav`).

## Example 2: Silence synthesizer

Generates silent audio of a fixed duration. Great for integration tests where the audio content doesn't matter -- you just want to verify the pipeline wiring works.

```python
class SilenceSynthesizer:
    def __init__(
        self, duration_seconds: float = 1.0, sample_rate: int = 24000
    ) -> None:
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> russo.Audio:
        num_samples = int(self.sample_rate * self.duration_seconds)
        silent_pcm = b"\x00\x00" * num_samples  # 16-bit silence
        return russo.Audio(
            data=silent_pcm, format="wav", sample_rate=self.sample_rate
        )
```

**Usage:**

```python
from russo.evaluators import ExactEvaluator

result = await russo.run(
    prompt="What's the weather in Tokyo?",
    synthesizer=SilenceSynthesizer(duration_seconds=0.5),
    agent=my_agent,
    evaluator=ExactEvaluator(),
    expect=[russo.tool_call("get_weather", city="Tokyo")],
)
```

Run the demo:

```bash
python examples/custom_synthesizer.py
```

Expected output:

```
PASSED (100% match rate)
  [+] get_weather({'city': 'Tokyo'}) -> get_weather({'city': 'Tokyo'})

Custom synthesizer test passed!
```

## When to use a custom synthesizer

- **Offline testing** -- no TTS API calls, no cost, no latency
- **Pre-recorded audio** -- use real human recordings for higher fidelity
- **Alternative TTS** -- plug in ElevenLabs, Azure Speech, Amazon Polly, etc.
- **CI pipelines** -- use `SilenceSynthesizer` to test pipeline wiring without credentials
