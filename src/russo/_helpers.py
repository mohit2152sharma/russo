"""Convenience helpers for building test scenarios."""

from __future__ import annotations

import functools
from collections.abc import Callable, Coroutine
from typing import Any

from russo._types import AgentResponse, Audio, ToolCall


def tool_call(name: str, **arguments: Any) -> ToolCall:
    """Shorthand for creating a ToolCall.

    Usage:
        russo.tool_call("book_flight", from_city="NYC", to_city="LA")
    """
    return ToolCall(name=name, arguments=arguments)


def agent(fn: Callable[[Audio], Coroutine[Any, Any, AgentResponse]]) -> _CallableAgent:
    """Decorator to turn an async function into an Agent.

    Usage:
        @russo.agent
        async def my_agent(audio: russo.Audio) -> russo.AgentResponse:
            result = await call_my_api(audio.data)
            return russo.AgentResponse(tool_calls=[...])
    """
    return _CallableAgent(fn)


class _CallableAgent:
    """Wraps an async callable as an Agent (satisfies the Agent protocol)."""

    def __init__(self, fn: Callable[[Audio], Coroutine[Any, Any, AgentResponse]]) -> None:
        self._fn = fn
        functools.update_wrapper(self, fn)

    async def run(self, audio: Audio) -> AgentResponse:
        return await self._fn(audio)
