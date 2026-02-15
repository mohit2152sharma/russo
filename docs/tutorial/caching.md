# Caching

TTS calls are slow and cost money. russo's caching layer saves synthesized audio to disk so repeated test runs skip the TTS call entirely.

## How It Works

Audio is cached by a deterministic hash of the prompt text (plus optional metadata like voice/model). On subsequent runs, if the cache has a hit, the synthesizer is bypassed.

```
First run:  prompt → TTS API → audio → cache → agent
Next runs:  prompt → cache hit → audio → agent   (TTS skipped)
```

## CachedSynthesizer

Wrap any synthesizer with `CachedSynthesizer`:

```python
from russo import CachedSynthesizer
from russo.synthesizers import GoogleSynthesizer

synth = CachedSynthesizer(GoogleSynthesizer(api_key="..."))
```

### Include Config in Cache Key

If you change the voice or model, you want the cache to invalidate:

```python
synth = CachedSynthesizer(
    GoogleSynthesizer(api_key="...", voice="Kore"),
    cache_key_extra={"voice": "Kore", "model": "gemini-2.5-flash-preview-tts"},
)
```

### Custom Cache Directory

```python
from russo import AudioCache, CachedSynthesizer

cache = AudioCache(Path("/tmp/russo_audio_cache"))
synth = CachedSynthesizer(GoogleSynthesizer(api_key="..."), cache=cache)
```

### Disable at Runtime

```python
synth = CachedSynthesizer(GoogleSynthesizer(api_key="..."), enabled=False)
```

## AudioCache

The low-level cache stores audio as file pairs:

- `<key>.audio` — raw audio bytes
- `<key>.meta` — JSON metadata (format, sample rate, prompt)

```python
from russo import AudioCache

cache = AudioCache()               # default: .russo_cache/
cache = AudioCache(Path("custom")) # custom directory

# Manual operations
cache.get("abc123")       # Audio | None
cache.put("abc123", audio, prompt="hello")
cache.size()              # number of cached entries
cache.clear()             # remove all entries
```

## pytest Integration

The pytest plugin automatically wraps your synthesizer with caching. Control it via CLI:

```bash
# Caching is ON by default
pytest

# Disable caching
pytest --russo-no-cache

# Clear cache before running
pytest --russo-clear-cache

# Custom cache directory
pytest --russo-cache-dir /tmp/my_cache
```

## API Reference

See [`AudioCache`](../reference/core/cache.md) and [`CachedSynthesizer`](../reference/core/cache.md) for the full API docs.
