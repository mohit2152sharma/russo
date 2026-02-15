"""Tests for synthesizers — GoogleSynthesizer, CachedSynthesizer, and the Synthesizer protocol."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from russo._cache import AudioCache, CachedSynthesizer
from russo._protocols import Synthesizer
from russo._types import Audio
from russo.synthesizers.google import GoogleSynthesizer
from tests.conftest import FakeSynthesizer, make_gemini_tts_response


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------
class TestSynthesizerProtocol:
    """Verify that our synthesizers satisfy the Synthesizer protocol."""

    def test_google_synthesizer_is_synthesizer(self) -> None:
        """GoogleSynthesizer should satisfy the Synthesizer protocol."""
        assert issubclass(GoogleSynthesizer, Synthesizer)

    def test_cached_synthesizer_is_synthesizer(self) -> None:
        """CachedSynthesizer should satisfy the Synthesizer protocol."""
        assert issubclass(CachedSynthesizer, Synthesizer)

    def test_fake_synthesizer_is_synthesizer(self) -> None:
        """Any object with async synthesize(text) -> Audio satisfies Synthesizer."""
        assert isinstance(FakeSynthesizer(), Synthesizer)

    def test_plain_object_not_synthesizer(self) -> None:
        """An object without synthesize() should not satisfy Synthesizer."""
        assert not isinstance(object(), Synthesizer)


# ---------------------------------------------------------------------------
# GoogleSynthesizer
# ---------------------------------------------------------------------------
class TestGoogleSynthesizer:
    def test_init_defaults(self) -> None:
        with patch("russo.synthesizers.google.genai"):
            synth = GoogleSynthesizer()
        assert synth.voice == "Kore"
        assert synth.model == "gemini-2.5-flash-preview-tts"
        assert synth.audio_format == "wav"
        assert synth.sample_rate == 24000

    def test_init_custom(self) -> None:
        with patch("russo.synthesizers.google.genai") as mock_genai:
            synth = GoogleSynthesizer(
                voice="Puck",
                model="gemini-2.0-flash",
                audio_format="mp3",
                sample_rate=16000,
                api_key="test-key",
            )
        assert synth.voice == "Puck"
        assert synth.model == "gemini-2.0-flash"
        assert synth.audio_format == "mp3"
        assert synth.sample_rate == 16000
        mock_genai.Client.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_synthesize_returns_audio(self) -> None:
        """synthesize() should call the Gemini API and return Audio with extracted data."""
        expected_audio = b"synthesized-audio-bytes"
        mock_response = make_gemini_tts_response(expected_audio)

        with patch("russo.synthesizers.google.genai") as mock_genai:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_genai.Client.return_value = mock_client

            synth = GoogleSynthesizer(voice="Kore", api_key="test-key")
            audio = await synth.synthesize("Hello world")

        assert isinstance(audio, Audio)
        assert audio.data == expected_audio
        assert audio.format == "wav"
        assert audio.sample_rate == 24000
        mock_client.aio.models.generate_content.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_synthesize_empty_response(self) -> None:
        """If Gemini returns no candidates, we get empty audio data."""
        mock_response = MagicMock()
        mock_response.candidates = []

        with patch("russo.synthesizers.google.genai") as mock_genai:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_genai.Client.return_value = mock_client

            synth = GoogleSynthesizer(api_key="test-key")
            audio = await synth.synthesize("Hello")

        assert audio.data == b""

    @pytest.mark.asyncio
    async def test_synthesize_multi_part_response(self) -> None:
        """Audio data from multiple parts should be concatenated."""
        part1 = MagicMock()
        part1.inline_data = MagicMock()
        part1.inline_data.data = b"chunk1-"

        part2 = MagicMock()
        part2.inline_data = MagicMock()
        part2.inline_data.data = b"chunk2"

        content = MagicMock()
        content.parts = [part1, part2]
        candidate = MagicMock()
        candidate.content = content
        mock_response = MagicMock()
        mock_response.candidates = [candidate]

        with patch("russo.synthesizers.google.genai") as mock_genai:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_genai.Client.return_value = mock_client

            synth = GoogleSynthesizer(api_key="test-key")
            audio = await synth.synthesize("Hello")

        assert audio.data == b"chunk1-chunk2"

    @pytest.mark.asyncio
    async def test_synthesize_passes_correct_config(self) -> None:
        """Verify the model, voice, and modality config are sent to Gemini."""
        mock_response = make_gemini_tts_response(b"audio")

        with patch("russo.synthesizers.google.genai") as mock_genai:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_genai.Client.return_value = mock_client

            synth = GoogleSynthesizer(voice="Puck", model="gemini-2.0-flash", api_key="k")
            await synth.synthesize("Test prompt")

        call_kwargs = mock_client.aio.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "gemini-2.0-flash"
        assert call_kwargs.kwargs["contents"] == "Test prompt"
        config = call_kwargs.kwargs["config"]
        assert "AUDIO" in config.response_modalities
        assert config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Puck"


# ---------------------------------------------------------------------------
# CachedSynthesizer
# ---------------------------------------------------------------------------
class TestCachedSynthesizer:
    @pytest.mark.asyncio
    async def test_cache_miss_then_hit(self, fake_synth: FakeSynthesizer, tmp_path: Path) -> None:
        """First call hits inner synthesizer, second call serves from cache."""
        cache = AudioCache(tmp_path / "cache")
        synth = CachedSynthesizer(fake_synth, cache=cache)

        audio1 = await synth.synthesize("hello")
        audio2 = await synth.synthesize("hello")

        assert fake_synth.calls == ["hello"]  # inner called only once
        assert audio1.data == audio2.data
        assert cache.size() == 1

    @pytest.mark.asyncio
    async def test_different_prompts_cached_separately(self, fake_synth: FakeSynthesizer, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path / "cache")
        synth = CachedSynthesizer(fake_synth, cache=cache)

        await synth.synthesize("hello")
        await synth.synthesize("world")

        assert fake_synth.calls == ["hello", "world"]
        assert cache.size() == 2

    @pytest.mark.asyncio
    async def test_disabled_skips_cache(self, fake_synth: FakeSynthesizer, tmp_path: Path) -> None:
        """When disabled, every call hits the inner synthesizer."""
        cache = AudioCache(tmp_path / "cache")
        synth = CachedSynthesizer(fake_synth, cache=cache, enabled=False)

        await synth.synthesize("hello")
        await synth.synthesize("hello")

        assert fake_synth.calls == ["hello", "hello"]
        assert cache.size() == 0

    @pytest.mark.asyncio
    async def test_toggle_enabled_at_runtime(self, fake_synth: FakeSynthesizer, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path / "cache")
        synth = CachedSynthesizer(fake_synth, cache=cache, enabled=True)

        await synth.synthesize("hello")  # cached
        synth.enabled = False
        await synth.synthesize("hello")  # bypasses cache

        assert fake_synth.calls == ["hello", "hello"]

    @pytest.mark.asyncio
    async def test_cache_key_extra_differentiates(self, fake_synth: FakeSynthesizer, tmp_path: Path) -> None:
        """Different cache_key_extra should produce separate cache entries."""
        cache = AudioCache(tmp_path / "cache")
        synth_a = CachedSynthesizer(fake_synth, cache=cache, cache_key_extra={"voice": "Kore"})
        synth_b = CachedSynthesizer(fake_synth, cache=cache, cache_key_extra={"voice": "Puck"})

        await synth_a.synthesize("hello")
        await synth_b.synthesize("hello")

        assert fake_synth.calls == ["hello", "hello"]  # both miss
        assert cache.size() == 2

    @pytest.mark.asyncio
    async def test_clear_cache_forces_re_synthesis(self, fake_synth: FakeSynthesizer, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path / "cache")
        synth = CachedSynthesizer(fake_synth, cache=cache)

        await synth.synthesize("hello")
        assert fake_synth.calls == ["hello"]

        cache.clear()
        assert cache.size() == 0

        await synth.synthesize("hello")
        assert fake_synth.calls == ["hello", "hello"]


# ---------------------------------------------------------------------------
# AudioCache (unit tests)
# ---------------------------------------------------------------------------
class TestAudioCache:
    def test_get_miss(self, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path / "cache")
        assert cache.get("nonexistent") is None

    def test_put_and_get(self, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path / "cache")
        audio = Audio(data=b"some-audio", format="mp3", sample_rate=16000)
        cache.put("key1", audio, prompt="test")

        result = cache.get("key1")
        assert result is not None
        assert result.data == b"some-audio"
        assert result.format == "mp3"
        assert result.sample_rate == 16000

    def test_size(self, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path / "cache")
        assert cache.size() == 0

        cache.put("a", Audio(data=b"1", format="wav"))
        cache.put("b", Audio(data=b"2", format="wav"))
        assert cache.size() == 2

    def test_clear(self, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path / "cache")
        cache.put("a", Audio(data=b"1", format="wav"))
        cache.put("b", Audio(data=b"2", format="wav"))

        cache.clear()
        assert cache.size() == 0
        assert cache.get("a") is None

    def test_clear_on_empty(self, tmp_path: Path) -> None:
        """Clearing a non-existent cache dir should not raise."""
        cache = AudioCache(tmp_path / "does-not-exist")
        cache.clear()  # no error

    def test_cache_key_deterministic(self) -> None:
        cache = AudioCache()
        k1 = cache.cache_key("hello")
        k2 = cache.cache_key("hello")
        assert k1 == k2

    def test_cache_key_varies_with_prompt(self) -> None:
        cache = AudioCache()
        assert cache.cache_key("hello") != cache.cache_key("world")

    def test_cache_key_varies_with_extra(self) -> None:
        cache = AudioCache()
        assert cache.cache_key("hello", voice="Kore") != cache.cache_key("hello", voice="Puck")

    def test_corrupt_meta_handled(self, tmp_path: Path) -> None:
        """Corrupt metadata file should return None and clean up."""
        cache = AudioCache(tmp_path / "cache")
        cache.put("bad", Audio(data=b"data", format="wav"))

        # Corrupt the meta file
        meta_path = cache.cache_dir / "bad.meta"
        meta_path.write_text("not json{{{")

        result = cache.get("bad")
        assert result is None
        # Entry should be cleaned up
        assert not (cache.cache_dir / "bad.audio").exists()
        assert not meta_path.exists()


# ---------------------------------------------------------------------------
# Integration tests (real API, skipped unless --integration)
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestGoogleSynthesizerIntegration:
    """These tests hit the real Google Gemini API.

    Run with: pytest tests/test_synthesizer.py --integration
    Requires: GOOGLE_API_KEY env var set.
    """

    @pytest.mark.asyncio
    async def test_synthesize_returns_audio_bytes(self, google_synth) -> None:
        """Real TTS call should return non-empty audio data.

        Audio is saved to tests/audio_output/ so you can listen to it.
        """
        audio = await google_synth.synthesize("Hello, this is a test.")

        assert isinstance(audio, Audio)
        assert len(audio.data) > 0, "Expected non-empty audio data from Gemini TTS"
        assert audio.format == "wav"
        assert audio.sample_rate == 24000

        # Save for manual listening (wraps raw PCM in a proper WAV container)
        out_dir = Path(__file__).parent / "audio_output"
        saved = audio.save(out_dir / "hello_test.wav")
        print(f"\n  Audio saved to: {saved}")

    @pytest.mark.asyncio
    async def test_synthesize_different_prompts(self, google_synth) -> None:
        """Different prompts should produce different audio."""
        audio_a = await google_synth.synthesize("Book a flight to New York")
        audio_b = await google_synth.synthesize("What is the weather in London")

        assert len(audio_a.data) > 0
        assert len(audio_b.data) > 0
        # Audio content should differ (not a guarantee, but overwhelmingly likely)
        assert audio_a.data != audio_b.data

    @pytest.mark.asyncio
    async def test_synthesize_with_cache_integration(self, google_synth, tmp_path: Path) -> None:
        """CachedSynthesizer should cache the real API response on disk."""
        cache = AudioCache(tmp_path / "integration_cache")
        synth = CachedSynthesizer(google_synth, cache=cache)

        audio1 = await synth.synthesize("Cache integration test prompt")
        assert cache.size() == 1
        assert len(audio1.data) > 0

        # Second call should serve from cache (no API hit)
        audio2 = await synth.synthesize("Cache integration test prompt")
        assert audio1.data == audio2.data

    @pytest.mark.asyncio
    async def test_synthesize_custom_voice(self, google_synth) -> None:
        """Synthesize with a different voice — reuses the session-scoped auth config."""
        import os

        from russo.synthesizers.google import GoogleSynthesizer

        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            synth = GoogleSynthesizer(api_key=api_key, voice="Puck")
        else:
            synth = GoogleSynthesizer(
                vertexai=True,
                project=os.environ.get("GOOGLE_CLOUD_PROJECT") or None,
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
                voice="Puck",
            )

        audio = await synth.synthesize("Testing a different voice.")

        assert isinstance(audio, Audio)
        assert len(audio.data) > 0
