"""Audio constants organized by provider.

Each provider namespace holds the sample rates, MIME types, and format
requirements for that provider's APIs.
"""

from __future__ import annotations

from enum import Enum


class AudioMime(str, Enum):
    """MIME types for audio formats."""

    MP3 = "audio/mp3"
    PCM = "audio/l16"
    OGG = "audio/ogg"
    WAV = "audio/wav"

    @classmethod
    def for_format(cls, format: str) -> str:
        """Return MIME type for format string, or 'audio/{format}' if unknown."""
        try:
            return cls[format.upper()].value
        except KeyError:
            return f"audio/{format}"


class Gemini:
    """Gemini / Vertex AI audio constants."""

    # Live API: 16-bit PCM at 16 kHz
    # https://cloud.google.com/vertex-ai/generative-ai/docs/live-api/send-audio-video-streams
    LIVE_INPUT_SAMPLE_RATE = 16000
    LIVE_INPUT_MIME = "audio/pcm;rate=16000"
    LIVE_RESPONSE_MODALITIES = ["AUDIO"]  # Native-audio models require AUDIO


class OpenAI:
    """OpenAI audio constants."""

    # Realtime API: pcm16 at 24 kHz mono (default)
    REALTIME_INPUT_SAMPLE_RATE = 24000
