"""Caching example — AudioCache and CachedSynthesizer.

TTS API calls are slow and expensive. CachedSynthesizer wraps any synthesizer
with a file-system cache so repeated test runs skip the TTS call entirely.

The cache key is a SHA-256 hash of the prompt text (+ optional extras),
so identical prompts always return cached audio.
"""

import asyncio
from pathlib import Path

import russo
from russo import AudioCache, CachedSynthesizer
from russo.evaluators import ExactEvaluator


class FakeSynthesizer:
    """Simulates a slow TTS call."""

    call_count: int = 0

    async def synthesize(self, text: str) -> russo.Audio:
        self.call_count += 1
        print(f"  [TTS] Synthesizing: {text!r}  (call #{self.call_count})")
        return russo.Audio(data=b"\x00" * 4800, format="wav", sample_rate=24000)


@russo.agent
async def fake_agent(audio: russo.Audio) -> russo.AgentResponse:
    return russo.AgentResponse(
        tool_calls=[russo.ToolCall(name="book_flight", arguments={"from_city": "NYC", "to_city": "LA"})]
    )


async def main():
    inner = FakeSynthesizer()

    # --- Example 1: Basic caching (default .russo_cache/ directory) ---
    print("=== Example 1: Basic caching ===")
    synth = CachedSynthesizer(inner)
    synth.cache.clear()  # start fresh for the demo

    # First call: cache miss -> calls the inner synthesizer
    result1 = await russo.run(
        prompt="Book a flight from NYC to LA",
        synthesizer=synth,
        agent=fake_agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
    )
    print(f"  Cache size after first run: {synth.cache.size()}")

    # Second call with same prompt: cache hit -> no TTS call
    result2 = await russo.run(
        prompt="Book a flight from NYC to LA",
        synthesizer=synth,
        agent=fake_agent,
        evaluator=ExactEvaluator(),
        expect=[russo.tool_call("book_flight", from_city="NYC", to_city="LA")],
    )
    print(f"  Total TTS calls: {inner.call_count} (should be 1 — second was cached)")

    # --- Example 2: Custom cache directory ---
    print("\n=== Example 2: Custom cache directory ===")
    custom_cache = AudioCache(Path("/tmp/russo_audio_cache"))
    synth2 = CachedSynthesizer(inner, cache=custom_cache)
    print(f"  Cache dir: {synth2.cache.cache_dir}")

    # --- Example 3: Cache key extras (invalidate on config change) ---
    print("\n=== Example 3: Cache key extras ===")
    synth3 = CachedSynthesizer(
        inner,
        cache_key_extra={"voice": "Kore", "model": "gemini-2.5-flash-preview-tts"},
    )
    # If you change voice/model, the cache key changes -> cache miss
    print("  Cache key for same prompt with different config will differ")

    # --- Example 4: Disable caching at runtime ---
    print("\n=== Example 4: Disable caching ===")
    synth4 = CachedSynthesizer(inner, enabled=False)
    await synth4.synthesize("test")  # always calls inner
    print(f"  TTS calls with caching disabled: {inner.call_count}")

    # --- Cleanup ---
    synth.cache.clear()
    custom_cache.clear()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
