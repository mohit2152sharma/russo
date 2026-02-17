"""WebSocket server wrapping Gemini for tool calling.

Accepts audio over WebSocket, sends it to Gemini with tool declarations,
and returns parsed tool calls. Used for end-to-end testing with russo's
WebSocketAgent adapter.

Uses the ``websockets`` library (already a russo dependency) — no FastAPI/uvicorn needed.

Run standalone:
    python examples/websocket_testing/server.py --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import signal
import wave

import websockets

logger = logging.getLogger("websocket_testing.server")

# ---------------------------------------------------------------------------
# Tool declarations (Gemini format)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "function_declarations": [
            {
                "name": "book_flight",
                "description": "Book a flight between two cities.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_city": {"type": "string", "description": "Departure city"},
                        "to_city": {"type": "string", "description": "Destination city"},
                    },
                    "required": ["from_city", "to_city"],
                },
            },
            {
                "name": "search_hotels",
                "description": "Search for hotels in a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City to search hotels in"},
                        "checkin_date": {"type": "string", "description": "Check-in date"},
                    },
                    "required": ["city"],
                },
            },
            {
                "name": "get_weather",
                "description": "Get current weather information for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City to get weather for"},
                    },
                    "required": ["city"],
                },
            },
        ]
    }
]

SYSTEM_INSTRUCTION = (
    "You are a travel assistant. When the user asks you to perform an action, "
    "call the appropriate tool. Available tools: book_flight (for booking flights), "
    "search_hotels (for finding hotels), get_weather (for weather information). "
    "Always call the most relevant tool based on the user's request."
)


def _ensure_wav(audio_bytes: bytes, sample_rate: int = 24000) -> tuple[bytes, str]:
    """Wrap raw PCM in a WAV container if needed. Return (data, mime_type)."""
    if len(audio_bytes) >= 4 and audio_bytes[:4] == b"RIFF":
        return audio_bytes, "audio/wav"
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    return buf.getvalue(), "audio/wav"


def _expand_creds_path() -> None:
    """Expand ~ in GOOGLE_APPLICATION_CREDENTIALS if present."""
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if creds and "~" in creds:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser(creds)


def _gcp_project() -> str | None:
    """Resolve GCP project from common env vars."""
    return os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_PROJECT_ID")


def _make_client():
    """Create a genai.Client using env-based auth (API key or Vertex AI)."""
    from google import genai

    _expand_creds_path()
    project = _gcp_project()
    if project:
        return genai.Client(
            vertexai=True,
            project=project,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    return genai.Client()


async def _handle_connection(ws: websockets.ServerConnection) -> None:
    """Handle a single audio -> tool-call exchange."""
    from google.genai import types

    try:
        raw = json.loads(await ws.recv())
        audio_b64 = raw.get("audio", "")
        audio_bytes = base64.b64decode(audio_b64)

        data, mime_type = _ensure_wav(audio_bytes)

        client = _make_client()
        contents = [types.Part.from_bytes(data=data, mime_type=mime_type)]
        config = types.GenerateContentConfig(
            tools=TOOLS,
            system_instruction=SYSTEM_INSTRUCTION,
        )

        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=config,
        )

        tool_calls = []
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.function_call:
                            fc = part.function_call
                            tool_calls.append(
                                {
                                    "name": fc.name,
                                    "arguments": dict(fc.args) if fc.args else {},
                                }
                            )

        await ws.send(json.dumps({"tool_calls": tool_calls}))

    except Exception:
        logger.exception("Error processing audio")
        await ws.send(json.dumps({"tool_calls": [], "error": "Internal server error"}))


async def serve(port: int) -> None:
    """Start the WebSocket server on the given port."""
    stop = asyncio.get_running_loop().create_future()

    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set_result, None)

    async with websockets.serve(_handle_connection, "localhost", port):
        logger.info("WebSocket server listening on ws://localhost:%d", port)
        await stop


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket agent server")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(args.port))
