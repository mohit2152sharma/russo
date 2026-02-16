"""AudioManager — shared audio processing for russo adapters."""

from __future__ import annotations

import io
import struct
import wave

from russo._types import Audio
from russo.audio.constants import AudioMime, Gemini, OpenAI


class AudioManager:
    """Centralized audio processing for adapter input preparation."""

    @staticmethod
    def has_wav_header(data: bytes) -> bool:
        """True if data starts with a RIFF/WAV header."""
        return len(data) >= 4 and data[:4] == b"RIFF"

    @staticmethod
    def extract_pcm(audio: Audio) -> bytes:
        """Extract raw PCM frames, stripping WAV headers if present."""
        if audio.format == "wav" and len(audio.data) > 44 and AudioManager.has_wav_header(audio.data):
            with wave.open(io.BytesIO(audio.data), "rb") as wf:
                return wf.readframes(wf.getnframes())
        return audio.data

    @staticmethod
    def resample_pcm_16bit(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
        """Resample 16-bit little-endian PCM (linear interpolation)."""
        if from_rate == to_rate:
            return pcm
        n_in = len(pcm) // 2
        samples = struct.unpack(f"<{n_in}h", pcm)
        n_out = int(round(n_in * to_rate / from_rate))
        if n_out == 0:
            return b""
        out: list[int] = []
        for i in range(n_out):
            src_idx = i * from_rate / to_rate
            lo = int(src_idx) % n_in
            hi = min(lo + 1, n_in - 1)
            frac = src_idx - int(src_idx)
            val = int(samples[lo] * (1 - frac) + samples[hi] * frac)
            out.append(max(-32768, min(32767, val)))
        return struct.pack(f"<{len(out)}h", *out)

    @staticmethod
    def prepare_for_generate_content(audio: Audio) -> tuple[bytes, str]:
        """Return (bytes, mime_type) for Gemini generate_content API.

        Wraps raw PCM in WAV container when needed.
        """
        if audio.format == "wav":
            if not AudioManager.has_wav_header(audio.data):
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(audio.channels)
                    wf.setsampwidth(audio.sample_width)
                    wf.setframerate(audio.sample_rate)
                    wf.writeframes(audio.data)
                return buf.getvalue(), "audio/wav"
            return audio.data, "audio/wav"
        return audio.data, AudioMime.for_format(audio.format)

    @staticmethod
    def prepare_for_live(audio: Audio) -> tuple[bytes, str]:
        """Return (pcm_bytes, mime_type) for Gemini Live API — 16 kHz PCM required."""
        pcm = AudioManager.extract_pcm(audio)
        if audio.sample_rate != Gemini.LIVE_INPUT_SAMPLE_RATE:
            pcm = AudioManager.resample_pcm_16bit(pcm, audio.sample_rate, Gemini.LIVE_INPUT_SAMPLE_RATE)
        return pcm, Gemini.LIVE_INPUT_MIME

    @staticmethod
    def prepare_for_openai_realtime(audio: Audio) -> bytes:
        """Return PCM bytes for OpenAI Realtime API — 24 kHz mono required."""
        pcm = AudioManager.extract_pcm(audio)
        if audio.sample_rate != OpenAI.REALTIME_INPUT_SAMPLE_RATE:
            pcm = AudioManager.resample_pcm_16bit(pcm, audio.sample_rate, OpenAI.REALTIME_INPUT_SAMPLE_RATE)
        return pcm
