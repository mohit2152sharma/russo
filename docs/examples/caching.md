# Caching

TTS API calls are slow and expensive. `CachedSynthesizer` wraps any synthesizer with a file-system cache so repeated test runs skip the TTS call entirely.

!!! tip "Source file"
    [`examples/caching.py`](https://github.com/mohit2152sharma/russo/blob/main/examples/caching.py)

## How it works

The cache key is a SHA-256 hash of the prompt text (plus optional extras). Identical prompts always return cached audio without calling the TTS API.

Each cached entry is a pair of files on disk:

```
.russo_cache/
  a1b2c3d4e5f6.audio   # raw audio bytes
  a1b2c3d4e5f6.meta    # JSON metadata (format, sample_rate, prompt)
```

## Example 1: Basic caching

Wrap any synthesizer with `CachedSynthesizer`:

```python
from russo import CachedSynthesizer
from russo.synthesizers import GoogleSynthesizer

inner = GoogleSynthesizer(api_key="...")
synth = CachedSynthesizer(inner)
```

First call with a prompt: **cache miss** -- calls the TTS API and stores the result. Second call with the same prompt: **cache hit** -- returns audio from disk instantly.

```python
import russo
from russo.evaluators import ExactEvaluator

# First run: TTS API is called
await russo.run(
    prompt="Book a flight from NYC to LA",
    synthesizer=synth,
    agent=my_agent,
    evaluator=ExactEvaluator(),
    expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
)
print(f"Cache size: {synth.cache.size()}")  # 1

# Second run with same prompt: TTS is skipped
await russo.run(
    prompt="Book a flight from NYC to LA",
    synthesizer=synth,
    ...
)
# No TTS call was made!
```

## Example 2: Custom cache directory

By default, cached audio goes to `.russo_cache/` in the current directory. You can customize this:

```python
from pathlib import Path
from russo import AudioCache, CachedSynthesizer

custom_cache = AudioCache(Path("/tmp/russo_audio_cache"))
synth = CachedSynthesizer(inner, cache=custom_cache)
```

## Example 3: Cache key extras

Include synthesizer config in the cache key so changing the voice or model automatically invalidates the cache:

```python
synth = CachedSynthesizer(
    inner,
    cache_key_extra={
        "voice": "Kore",
        "model": "gemini-2.5-flash-preview-tts",
    },
)
```

Same prompt + same config = cache hit. Same prompt + different voice = cache miss.

## Example 4: Disable caching at runtime

Temporarily bypass the cache without changing your setup:

```python
synth = CachedSynthesizer(inner, enabled=False)
# All calls go to the inner synthesizer directly
```

## Cache management

```python
# Check how many entries are cached
synth.cache.size()

# Clear all cached entries
synth.cache.clear()
```

## pytest integration

The russo pytest plugin supports caching out of the box via CLI options:

```bash
pytest --russo-cache                # enable audio cache (default)
pytest --russo-no-cache             # disable caching
pytest --russo-clear-cache          # clear cache before running
pytest --russo-cache-dir .my_cache  # custom cache directory
```

Run the demo:

```bash
python examples/caching.py
```

Expected output:

```
=== Example 1: Basic caching ===
  [TTS] Synthesizing: 'Book a flight from NYC to LA'  (call #1)
  Cache size after first run: 1
  Total TTS calls: 1 (should be 1 — second was cached)

=== Example 2: Custom cache directory ===
  Cache dir: /tmp/russo_audio_cache

=== Example 3: Cache key extras ===
  Cache key for same prompt with different config will differ

=== Example 4: Disable caching ===
  [TTS] Synthesizing: 'test'  (call #2)
  TTS calls with caching disabled: 2

Done!
```
