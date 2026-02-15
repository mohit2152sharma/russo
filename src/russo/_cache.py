"""Audio caching for synthesized prompts.

Saves synthesized audio to disk keyed by a hash of the prompt text,
so repeated test runs skip the TTS call entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from russo._protocols import Synthesizer
from russo._types import Audio

logger = logging.getLogger("russo.cache")

_DEFAULT_CACHE_DIR = Path(".russo_cache")


class AudioCache:
    """File-system cache for synthesized audio.

    Each entry is a pair of files:
        <key>.audio  — raw audio bytes
        <key>.meta   — JSON with format, sample_rate, prompt

    Usage:
        cache = AudioCache()                     # .russo_cache/
        cache = AudioCache(Path("my_cache"))     # custom dir
        cache.get("abc123")                      # Audio | None
        cache.put("abc123", audio)
        cache.clear()
    """

    def __init__(self, cache_dir: Path = _DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = cache_dir

    def _ensure_dir(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_key(self, prompt: str, **extra: Any) -> str:
        """Deterministic key from prompt text + optional extra metadata.

        Extra kwargs (e.g. voice, model) are included so a change in
        synthesizer config invalidates the cache automatically.
        """
        blob = json.dumps({"prompt": prompt, **extra}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def get(self, key: str) -> Audio | None:
        """Load cached audio, or None if not cached."""
        audio_path = self.cache_dir / f"{key}.audio"
        meta_path = self.cache_dir / f"{key}.meta"
        if not audio_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
            data = audio_path.read_bytes()
            logger.debug("Cache hit: %s", key)
            return Audio(
                data=data, format=meta["format"], sample_rate=meta["sample_rate"]
            )
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Corrupt cache entry %s, removing: %s", key, exc)
            self._remove_entry(key)
            return None

    def put(self, key: str, audio: Audio, *, prompt: str = "") -> None:
        """Write audio + metadata to cache."""
        self._ensure_dir()
        audio_path = self.cache_dir / f"{key}.audio"
        meta_path = self.cache_dir / f"{key}.meta"
        audio_path.write_bytes(audio.data)
        meta = {
            "format": audio.format,
            "sample_rate": audio.sample_rate,
            "prompt": prompt,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.debug("Cached: %s (%d bytes)", key, len(audio.data))

    def clear(self) -> None:
        """Remove all cached entries."""
        if not self.cache_dir.exists():
            return
        count = 0
        for f in self.cache_dir.iterdir():
            if f.suffix in (".audio", ".meta"):
                f.unlink()
                count += 1
        logger.info("Cleared %d cache files from %s", count, self.cache_dir)

    def size(self) -> int:
        """Number of cached audio entries."""
        if not self.cache_dir.exists():
            return 0
        return sum(1 for f in self.cache_dir.iterdir() if f.suffix == ".audio")

    def _remove_entry(self, key: str) -> None:
        for suffix in (".audio", ".meta"):
            p = self.cache_dir / f"{key}{suffix}"
            p.unlink(missing_ok=True)


class CachedSynthesizer:
    """Wraps any Synthesizer with local audio caching.

    Satisfies the Synthesizer protocol — drop-in replacement.

    Usage:
        synth = CachedSynthesizer(GoogleSynthesizer(...))

        # Disable caching at runtime
        synth = CachedSynthesizer(GoogleSynthesizer(...), enabled=False)

        # Custom cache directory
        synth = CachedSynthesizer(
            GoogleSynthesizer(...),
            cache=AudioCache(Path("/tmp/my_cache")),
        )

        # Include synthesizer config in cache key (invalidates on config change)
        synth = CachedSynthesizer(
            GoogleSynthesizer(voice="Kore", model="gemini-2.5-flash-preview-tts"),
            cache_key_extra={"voice": "Kore", "model": "gemini-2.5-flash-preview-tts"},
        )

        # Clear cache
        synth.cache.clear()
    """

    def __init__(
        self,
        synthesizer: Synthesizer,
        *,
        cache: AudioCache | None = None,
        enabled: bool = True,
        cache_key_extra: dict[str, Any] | None = None,
    ) -> None:
        self.inner = synthesizer
        self.cache = cache or AudioCache()
        self.enabled = enabled
        self.cache_key_extra = cache_key_extra or {}

    async def synthesize(self, text: str) -> Audio:
        """Synthesize with cache lookup/store."""
        if not self.enabled:
            return await self.inner.synthesize(text)

        key = self.cache.cache_key(text, **self.cache_key_extra)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        audio = await self.inner.synthesize(text)
        self.cache.put(key, audio, prompt=text)
        return audio
