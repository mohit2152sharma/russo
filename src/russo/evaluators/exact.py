"""Exact-match evaluator for tool calls."""

from __future__ import annotations

from russo._types import EvalResult, ToolCall, ToolCallMatch


class ExactEvaluator:
    """Evaluates tool calls by exact name + arguments match.

    Supports optional config for relaxed matching:
    - match_order: If True, tool calls must appear in the same order.
    - ignore_extra_args: If True, actual calls may contain extra arguments.
    - ignore_extra_calls: If True, extra actual calls don't cause failure.

    Usage:
        evaluator = ExactEvaluator()
        result = evaluator.evaluate(expected=[...], actual=[...])
    """

    def __init__(
        self,
        *,
        match_order: bool = False,
        ignore_extra_args: bool = False,
        ignore_extra_calls: bool = True,
    ) -> None:
        self.match_order = match_order
        self.ignore_extra_args = ignore_extra_args
        self.ignore_extra_calls = ignore_extra_calls

    def evaluate(self, expected: list[ToolCall], actual: list[ToolCall]) -> EvalResult:
        """Compare expected tool calls against actual ones."""
        if not expected:
            return EvalResult(passed=True, expected=expected, actual=actual, matches=[])

        matches: list[ToolCallMatch] = []
        remaining_actual = list(actual)

        for i, exp in enumerate(expected):
            match = self._find_match(exp, remaining_actual, index=i if self.match_order else None)
            matches.append(match)
            if match.matched and match.actual in remaining_actual:
                remaining_actual.remove(match.actual)

        all_matched = all(m.matched for m in matches)

        extra_calls_ok = self.ignore_extra_calls or len(remaining_actual) == 0
        passed = all_matched and extra_calls_ok

        if not extra_calls_ok and all_matched:
            for leftover in remaining_actual:
                matches.append(
                    ToolCallMatch(
                        expected=ToolCall(name="(none)", arguments={}),
                        actual=leftover,
                        matched=False,
                        details=f"Unexpected extra tool call: {leftover.name}",
                    )
                )

        return EvalResult(passed=passed, expected=expected, actual=actual, matches=matches)

    def _find_match(
        self, expected: ToolCall, candidates: list[ToolCall], *, index: int | None
    ) -> ToolCallMatch:
        """Find a matching actual tool call for the expected one."""
        if index is not None:
            # Order-sensitive: must match at the exact position
            if index < len(candidates):
                candidate = candidates[index]
                if self._is_match(expected, candidate):
                    return ToolCallMatch(expected=expected, actual=candidate, matched=True)
                return ToolCallMatch(
                    expected=expected,
                    actual=candidate,
                    matched=False,
                    details=f"Position {index}: expected {expected.name}({expected.arguments}), "
                    f"got {candidate.name}({candidate.arguments})",
                )
            return ToolCallMatch(
                expected=expected,
                matched=False,
                details=f"Position {index}: no actual call at this index",
            )

        # Order-insensitive: find first match in candidates
        for candidate in candidates:
            if self._is_match(expected, candidate):
                return ToolCallMatch(expected=expected, actual=candidate, matched=True)

        # No match found — find closest for diagnostics
        if candidates:
            best = min(candidates, key=lambda c: self._distance(expected, c))
            return ToolCallMatch(
                expected=expected,
                actual=best,
                matched=False,
                details=self._diff_details(expected, best),
            )
        return ToolCallMatch(
            expected=expected, matched=False, details="No actual tool calls to match against"
        )

    def _is_match(self, expected: ToolCall, actual: ToolCall) -> bool:
        """Check if an actual tool call matches the expected one."""
        if expected.name != actual.name:
            return False
        if self.ignore_extra_args:
            # All expected args must be present in actual (actual may have more)
            return all(
                actual.arguments.get(k) == v for k, v in expected.arguments.items()
            )
        return expected.arguments == actual.arguments

    def _distance(self, expected: ToolCall, actual: ToolCall) -> int:
        """Simple distance metric for diagnostic ranking."""
        d = 0
        if expected.name != actual.name:
            d += 10
        for k, v in expected.arguments.items():
            if k not in actual.arguments:
                d += 2
            elif actual.arguments[k] != v:
                d += 1
        return d

    def _diff_details(self, expected: ToolCall, actual: ToolCall) -> str:
        """Generate a human-readable diff between expected and actual."""
        diffs: list[str] = []
        if expected.name != actual.name:
            diffs.append(f"name: expected '{expected.name}', got '{actual.name}'")
        for k, v in expected.arguments.items():
            if k not in actual.arguments:
                diffs.append(f"arg '{k}': missing (expected {v!r})")
            elif actual.arguments[k] != v:
                diffs.append(f"arg '{k}': expected {v!r}, got {actual.arguments[k]!r}")
        return "; ".join(diffs) if diffs else "unknown mismatch"
