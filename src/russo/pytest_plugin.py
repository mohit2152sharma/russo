"""pytest plugin for russo — auto-discovered via the pytest11 entry point.

Provides:
- `russo` marker for declarative test scenarios
- `russo_result` fixture that runs the full pipeline
- Terminal summary via `pytest_terminal_summary` hook
- `--russo-report` CLI option for HTML report output
"""

from __future__ import annotations

from typing import Any

import pytest

from russo._cache import AudioCache, CachedSynthesizer
from russo._pipeline import run
from russo._types import EvalResult, ToolCall
from russo.evaluators.exact import ExactEvaluator
from russo.report.terminal import TerminalReporter

# ---------------------------------------------------------------------------
# Global state for collecting results
# ---------------------------------------------------------------------------
_reporter = TerminalReporter()


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("russo", "russo tool-call testing")
    group.addoption(
        "--russo-report",
        action="store",
        default=None,
        metavar="PATH",
        help="Write russo HTML report to PATH.",
    )
    group.addoption(
        "--russo-cache",
        action="store_true",
        default=True,
        dest="russo_cache",
        help="Enable audio cache for synthesized prompts (default: on).",
    )
    group.addoption(
        "--russo-no-cache",
        action="store_false",
        dest="russo_cache",
        help="Disable audio cache — always call the synthesizer.",
    )
    group.addoption(
        "--russo-clear-cache",
        action="store_true",
        default=False,
        help="Clear the audio cache before running tests.",
    )
    group.addoption(
        "--russo-cache-dir",
        action="store",
        default=".russo_cache",
        metavar="DIR",
        help="Directory for cached audio files (default: .russo_cache).",
    )


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "russo(prompt, expect, **kwargs): Mark a test as a russo tool-call scenario.",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Clear audio cache if --russo-clear-cache was passed."""
    if session.config.getoption("russo_clear_cache", default=False):
        from pathlib import Path

        cache_dir = session.config.getoption("russo_cache_dir", default=".russo_cache")
        cache = AudioCache(Path(cache_dir))
        n = cache.size()
        cache.clear()
        # Use write_line via terminal writer if available
        tw = session.config.get_terminal_writer()
        tw.line(f"russo: cleared {n} cached audio entries from {cache_dir}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def russo_audio_cache(request: pytest.FixtureRequest) -> AudioCache:
    """Session-scoped audio cache. Override in conftest.py to customize."""
    from pathlib import Path

    cache_dir = request.config.getoption("russo_cache_dir", default=".russo_cache")
    return AudioCache(Path(cache_dir))


@pytest.fixture
def russo_evaluator() -> ExactEvaluator:
    """Default evaluator — exact match. Override in conftest.py to customize."""
    return ExactEvaluator()


@pytest.fixture
async def russo_result(request: pytest.FixtureRequest) -> EvalResult | None:
    """Run the russo pipeline based on the @pytest.mark.russo marker.

    Reads marker kwargs, resolves synthesizer/agent/evaluator fixtures,
    runs the pipeline, and returns the EvalResult.

    Returns None if the test has no russo marker (allows manual usage).
    """
    marker = request.node.get_closest_marker("russo")
    if marker is None:
        return None

    # Extract marker arguments
    prompt: str = marker.kwargs.get("prompt", "")
    if not prompt and marker.args:
        prompt = marker.args[0]

    expect_raw: list[Any] = marker.kwargs.get("expect", [])
    expect: list[ToolCall] = [tc if isinstance(tc, ToolCall) else ToolCall(**tc) for tc in expect_raw]

    # Resolve fixtures
    synthesizer = request.getfixturevalue("russo_synthesizer")
    agent = request.getfixturevalue("russo_agent")

    # Wrap synthesizer with caching (unless it's already cached or caching is off)
    cache_enabled = request.config.getoption("russo_cache", default=True)
    if cache_enabled and not isinstance(synthesizer, CachedSynthesizer):
        cache = request.getfixturevalue("russo_audio_cache")
        synthesizer = CachedSynthesizer(synthesizer, cache=cache)
    elif not cache_enabled and isinstance(synthesizer, CachedSynthesizer):
        synthesizer.enabled = False

    try:
        evaluator = request.getfixturevalue("russo_evaluator")
    except pytest.FixtureLookupError:
        evaluator = ExactEvaluator()

    result = await run(
        prompt=prompt,
        synthesizer=synthesizer,
        agent=agent,
        evaluator=evaluator,
        expect=expect,
    )

    # Collect for reporting
    _reporter.add(request.node.nodeid, result)

    return result


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------
def pytest_terminal_summary(
    terminalreporter: Any,
    exitstatus: int,  # noqa: ARG001
    config: pytest.Config,
) -> None:
    """Print russo results summary at the end of the test run."""
    if _reporter.total == 0:
        return

    terminalreporter.write_line(_reporter.summary())

    # Write HTML report if requested
    report_path = config.getoption("--russo-report", default=None)
    if report_path:
        _write_html_report(report_path)
        terminalreporter.write_line(f"\nRusso HTML report written to: {report_path}")


def _write_html_report(path: str) -> None:
    """Generate a simple HTML report of russo test results."""
    rows = ""
    for name, result in _reporter.results:
        status_class = "pass" if result.passed else "fail"
        status = "PASS" if result.passed else "FAIL"
        rate = f"{result.match_rate:.0%}"
        details = result.summary().replace("\n", "<br>").replace(" ", "&nbsp;")
        rows += f"""
        <tr class="{status_class}">
            <td>{name}</td>
            <td>{status}</td>
            <td>{rate}</td>
            <td><pre>{details}</pre></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Russo Test Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        .pass {{ background-color: #e6ffe6; }}
        .fail {{ background-color: #ffe6e6; }}
        pre {{ margin: 0; white-space: pre-wrap; font-size: 0.85em; }}
        h1 {{ color: #333; }}
        .summary {{ margin: 1rem 0; font-size: 1.1em; }}
    </style>
</head>
<body>
    <h1>Russo Test Report</h1>
    <div class="summary">
        Total: {_reporter.total} |
        Passed: {_reporter.passed} |
        Failed: {_reporter.failed}
    </div>
    <table>
        <thead>
            <tr>
                <th>Test</th>
                <th>Status</th>
                <th>Match Rate</th>
                <th>Details</th>
            </tr>
        </thead>
        <tbody>{rows}
        </tbody>
    </table>
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Session cleanup
# ---------------------------------------------------------------------------
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    """Reset global reporter state between sessions (relevant for xdist, etc.)."""
    global _reporter  # noqa: PLW0603
    _reporter = TerminalReporter()
