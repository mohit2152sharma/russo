"""Callable agent adapter — wraps async functions as Agents.

Re-exports from _helpers for organizational clarity.
The actual implementation lives in russo._helpers to keep the
public API flat (russo.agent decorator).
"""

from russo._helpers import _CallableAgent, agent

__all__ = ["CallableAgent", "agent"]

# Public alias
CallableAgent = _CallableAgent
