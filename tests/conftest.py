"""Shared fixtures for russo tests."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from russo._types import Audio


# ---------------------------------------------------------------------------
# CLI option & marker for integration tests
# ---------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests that hit real APIs (requires GOOGLE_API_KEY).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: marks tests that call real external APIs (skipped unless --integration is passed)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--integration"):
        return  # user asked for integration tests, don't skip
    skip_integration = pytest.mark.skip(reason="needs --integration flag to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


# ---------------------------------------------------------------------------
# Fake / unit-test fixtures
# ---------------------------------------------------------------------------
class FakeSynthesizer:
    """In-memory synthesizer that records calls and returns deterministic audio."""

    def __init__(self, audio_data: bytes = b"fake-audio-data") -> None:
        self.audio_data = audio_data
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> Audio:
        self.calls.append(text)
        return Audio(data=self.audio_data, format="wav", sample_rate=24000)


@pytest.fixture
def fake_synth() -> FakeSynthesizer:
    return FakeSynthesizer()


@pytest.fixture
def sample_audio() -> Audio:
    return Audio(data=b"test-audio-bytes", format="wav", sample_rate=24000)


# ---------------------------------------------------------------------------
# Integration fixtures (real API)
# ---------------------------------------------------------------------------
@pytest.fixture
def google_api_key() -> str | None:
    """Return the API key if set, or None when using ADC."""
    return os.environ.get("GOOGLE_API_KEY") or None


@pytest.fixture
def google_synth(google_api_key: str | None):
    """Real GoogleSynthesizer pointing at Gemini TTS.

    Auth resolution:
      1. GOOGLE_API_KEY env var → Google AI API (api_key)
      2. ADC (gcloud cli, GOOGLE_APPLICATION_CREDENTIALS, etc.) → Vertex AI
         The SDK resolves project/location from ADC or env vars automatically.

    Skips if no credentials can be found at all.
    """
    from russo.synthesizers.google import GoogleSynthesizer

    if google_api_key:
        return GoogleSynthesizer(api_key=google_api_key)

    # Try Vertex AI with ADC — covers gcloud auth, service accounts, workload identity, etc.
    # The google-genai SDK resolves project from ADC / GOOGLE_CLOUD_PROJECT automatically.
    try:
        return GoogleSynthesizer(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT") or None,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    except Exception as exc:
        pytest.skip(
            f"No usable Google credentials found "
            f"(set GOOGLE_API_KEY or run 'gcloud auth application-default login'): {exc}"
        )


@pytest.fixture
def gemini_client(google_api_key: str | None):
    """Real google.genai.Client for adapter integration tests.

    Auth resolution mirrors google_synth — prefers GOOGLE_API_KEY, falls back to Vertex AI ADC.
    """
    from google import genai

    if google_api_key:
        return genai.Client(api_key=google_api_key)

    try:
        return genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT") or None,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    except Exception as exc:
        pytest.skip(
            f"No usable Google credentials found "
            f"(set GOOGLE_API_KEY or run 'gcloud auth application-default login'): {exc}"
        )


BOOK_FLIGHT_TOOL = {
    "function_declarations": [
        {
            "name": "book_flight",
            "description": "Book a flight between two cities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_city": {"type": "string", "description": "Departure city"},
                    "to_city": {"type": "string", "description": "Arrival city"},
                },
                "required": ["from_city", "to_city"],
            },
        }
    ]
}


@pytest.fixture
def gemini_live_agent(gemini_client: Any):
    """GeminiLiveAgent pre-configured with a book_flight tool for integration tests."""
    from russo.adapters.gemini import GeminiLiveAgent

    return GeminiLiveAgent(
        client=gemini_client,
        model="gemini-2.0-flash-live-preview-04-09",
        tools=[BOOK_FLIGHT_TOOL],
        system_instruction=(
            "You are a travel assistant. When the user asks to book a flight, "
            "call the book_flight function with from_city and to_city."
        ),
    )


def make_gemini_response(function_calls: list[dict[str, Any]]) -> MagicMock:
    """Build a mock that mimics a google-genai GenerateContentResponse with function_call parts."""
    parts = []
    for fc in function_calls:
        part = MagicMock()
        part.inline_data = None
        part.function_call = MagicMock()
        part.function_call.name = fc["name"]
        part.function_call.args = fc.get("args", {})
        parts.append(part)

    content = MagicMock()
    content.parts = parts
    candidate = MagicMock()
    candidate.content = content
    response = MagicMock()
    response.candidates = [candidate]
    return response


def make_gemini_tts_response(audio_data: bytes) -> MagicMock:
    """Build a mock that mimics a google-genai TTS response with inline audio data."""
    inline_data = MagicMock()
    inline_data.data = audio_data

    part = MagicMock()
    part.inline_data = inline_data

    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    response = MagicMock()
    response.candidates = [candidate]
    return response
