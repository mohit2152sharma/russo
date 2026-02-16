"""Terminal reporter for russo test results."""

from __future__ import annotations

from russo._types import BatchResult, EvalResult

ResultType = EvalResult | BatchResult


class TerminalReporter:
    """Collects russo results and formats them for terminal output."""

    def __init__(self) -> None:
        self.results: list[tuple[str, ResultType]] = []

    def add(self, test_name: str, result: ResultType) -> None:
        """Record a result for a test."""
        self.results.append((test_name, result))

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for _, r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    def summary(self) -> str:
        """Generate a summary table for terminal output."""
        if not self.results:
            return "No russo tests collected."

        lines: list[str] = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("RUSSO TEST RESULTS")
        lines.append("=" * 60)

        max_name_len = max(len(name) for name, _ in self.results)

        for name, result in self.results:
            status = "PASS" if result.passed else "FAIL"
            rate = f"{result.match_rate:.0%}"
            suffix = ""
            if isinstance(result, BatchResult):
                suffix = f"  [{result.passed_count}/{result.total} runs]"
            lines.append(f"  {status}  {name:<{max_name_len}}  ({rate} match){suffix}")

        lines.append("-" * 60)
        lines.append(f"  Total: {self.total}  Passed: {self.passed}  Failed: {self.failed}")
        lines.append("=" * 60)

        failures = [(name, r) for name, r in self.results if not r.passed]
        if failures:
            lines.append("")
            lines.append("FAILURES:")
            for name, result in failures:
                lines.append(f"\n  {name}:")
                lines.append(f"  {result.summary()}")

        return "\n".join(lines)
