"""Built-in agent adapters for different invocation styles."""

from russo.adapters.callable import CallableAgent
from russo.adapters.gemini import GeminiAgent, GeminiLiveAgent
from russo.adapters.http import HttpAgent
from russo.adapters.openai import OpenAIAgent, OpenAIRealtimeAgent
from russo.adapters.websocket import WebSocketAgent

__all__ = [
    "CallableAgent",
    "GeminiAgent",
    "GeminiLiveAgent",
    "HttpAgent",
    "OpenAIAgent",
    "OpenAIRealtimeAgent",
    "WebSocketAgent",
]
