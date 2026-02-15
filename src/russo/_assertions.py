"""Custom assertion helpers for russo test results."""

from __future__ import annotations

from russo._types import EvalResult


class ToolCallAssertionError(AssertionError):
    """Rich assertion error with detailed tool call diff."""

    def __init__(self, result: EvalResult, message: str = "") -> None:
        self.result = result
        detail = result.summary()
        full_message = f"{message}\n{detail}" if message else detail
        super().__init__(full_message)


def assert_tool_calls(
    result: EvalResult,
    *,
    message: str = "",
) -> None:
    """Assert that an EvalResult passed.

    Raises a ToolCallAssertionError with a rich diff if it didn't.

    Usage:
        result = await russo.run(...)
        russo.assert_tool_calls(result)

        # Or with a custom message
        russo.assert_tool_calls(result, message="Flight booking should work")
    """
    if not result.passed:
        raise ToolCallAssertionError(result, message)
