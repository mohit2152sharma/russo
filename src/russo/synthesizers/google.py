"""Google Gemini TTS synthesizer."""

from __future__ import annotations

from typing import Literal

from google import genai
from google.genai import types

from russo._types import Audio


class GoogleSynthesizer:
    """Synthesizes audio from text using Google Gemini's TTS.

    Supports two auth modes:

    1. **Google AI API** (api_key):
        synth = GoogleSynthesizer(api_key="AIza...")

    2. **Vertex AI** (ADC / GOOGLE_APPLICATION_CREDENTIALS):
        synth = GoogleSynthesizer(vertexai=True, project="my-proj", location="us-central1")
    """

    def __init__(
        self,
        *,
        voice: str = "Kore",
        model: str = "gemini-2.5-flash-preview-tts",
        api_key: str | None = None,
        vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
        audio_format: Literal["wav", "mp3", "pcm", "ogg"] = "wav",
        sample_rate: int = 24000,
    ) -> None:
        self.voice = voice
        self.model = model
        self.audio_format = audio_format
        self.sample_rate = sample_rate

        if api_key:
            self._client = genai.Client(api_key=api_key)
        elif vertexai:
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=location or "us-central1",
            )
        else:
            # Fall back: let the SDK resolve from env (GOOGLE_API_KEY, ADC, etc.)
            self._client = genai.Client()

    async def synthesize(self, text: str) -> Audio:
        """Convert text to audio using Gemini TTS."""
        # Use explicit Content with text Part so the backend treats this as text-to-speech
        # (raw string can be interpreted as non-audio request and trigger 1007).
        contents = types.Content(
            role="user",
            parts=[types.Part.from_text(text=text)],
        )
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice),
                    ),
                ),
            ),
        )
        audio_data = b""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    audio_data += part.inline_data.data

        return Audio(
            data=audio_data,
            format=self.audio_format,
            sample_rate=self.sample_rate,
        )
