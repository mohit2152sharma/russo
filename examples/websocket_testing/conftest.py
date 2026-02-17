"""pytest conftest.py — server lifecycle + russo fixtures for WebSocket testing.

Starts a websockets server as a subprocess, wires up GoogleSynthesizer,
WebSocketAgent, and ExactEvaluator for end-to-end russo tests.

Requires Google AI or Vertex AI credentials in the environment.
"""

from __future__ import annotations

import os
import pathlib
import socket
import subprocess
import sys
import time

import pytest

from russo.adapters import WebSocketAgent
from russo.evaluators import ExactEvaluator
from russo.synthesizers import GoogleSynthesizer

_SERVER_SCRIPT = str(pathlib.Path(__file__).parent / "server.py")

# Expand ~ in GOOGLE_APPLICATION_CREDENTIALS — the google-auth library
# doesn't do this, so a value like "~/.config/gcloud/..." would fail.
_creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
if _creds_path and "~" in _creds_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser(_creds_path)


def _gcp_project() -> str | None:
    """Resolve GCP project from common env vars."""
    return os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_PROJECT_ID")


def _find_free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout: float = 15.0) -> None:
    """Poll until the server accepts connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.3)
    msg = f"Server on port {port} did not start within {timeout}s"
    raise TimeoutError(msg)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def travel_agent_server():
    """Start the WebSocket server and yield the port number."""
    port = _find_free_port()
    proc = subprocess.Popen(
        [sys.executable, _SERVER_SCRIPT, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_server(port)
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# russo fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def russo_synthesizer():
    """Google TTS synthesizer — uses Vertex AI if project is set, else API key.

    Function-scoped so each test gets a fresh genai.Client (its internal
    asyncio.Lock binds to the current event loop and can't be reused across
    the per-test loops that pytest-asyncio creates by default).  The russo
    plugin wraps this in CachedSynthesizer, so TTS calls are still cached.
    """
    project = _gcp_project()
    if project:
        return GoogleSynthesizer(
            vertexai=True,
            project=project,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    return GoogleSynthesizer()


@pytest.fixture
def russo_agent(travel_agent_server):
    """WebSocketAgent pointing at the local WebSocket server."""
    port = travel_agent_server
    return WebSocketAgent(url=f"ws://localhost:{port}")


@pytest.fixture
def russo_evaluator():
    """Exact match evaluator."""
    return ExactEvaluator()
