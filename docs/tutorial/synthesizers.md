# Synthesizers

Synthesizers convert text prompts into audio. This is the first step of the russo pipeline.

## Protocol

```python
class Synthesizer(Protocol):
    async def synthesize(self, text: str) -> Audio: ...
```

## Built-in: Google TTS

```python
from russo.synthesizers import GoogleSynthesizer

synth = GoogleSynthesizer(
    api_key="...",
    voice="Kore",                              # optional
    model="gemini-2.5-flash-preview-tts",      # optional
)

audio = await synth.synthesize("Book a flight from Berlin to Rome")
audio.save("output.wav")  # save to file
```

### Authentication Modes

=== "API Key (Google AI)"

    ```python
    synth = GoogleSynthesizer(api_key="AIza...")
    ```

=== "Vertex AI"

    ```python
    synth = GoogleSynthesizer(
        project="my-gcp-project",
        location="us-central1",
    )
    ```

## Custom Synthesizers

Implement the protocol — no inheritance needed:

```python
class ElevenLabsSynthesizer:
    def __init__(self, api_key: str, voice_id: str = "default"):
        self.api_key = api_key
        self.voice_id = voice_id

    async def synthesize(self, text: str) -> russo.Audio:
        # Call ElevenLabs API
        audio_bytes = await eleven_labs_tts(text, self.voice_id, self.api_key)
        return russo.Audio(data=audio_bytes, format="mp3")
```

## Caching

Wrap any synthesizer with `CachedSynthesizer` to avoid repeated TTS calls:

```python
from russo import CachedSynthesizer

cached = CachedSynthesizer(
    GoogleSynthesizer(api_key="..."),
    cache_key_extra={"voice": "Kore"},  # invalidate cache on config change
)
```

See [Caching](caching.md) for details.

## API Reference

See [`GoogleSynthesizer`](../reference/synthesizers/google.md) for the full API docs.
