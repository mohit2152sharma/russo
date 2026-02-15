# Installation

## Requirements

- Python 3.12+

## Install from PyPI

```bash
pip install russo
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add russo
```

## Optional Dependencies

russo has optional extras for different LLM providers:

=== "OpenAI"

    ```bash
    pip install "russo[openai]"
    ```

    Adds support for `OpenAIAgent` and `OpenAIRealtimeAgent`.

=== "WebSocket"

    ```bash
    pip install "russo[ws]"
    ```

    Adds support for `WebSocketAgent` (generic WebSocket connections).

=== "All"

    ```bash
    pip install "russo[all]"
    ```

    Installs all optional dependencies.

## For Development

```bash
git clone https://github.com/mohit2152sharma/russo.git
cd russo
uv sync --all-extras
```

This installs all dependencies including dev tools (ruff, pytest, basedpyright, etc.) and documentation tools (mkdocs-material).

## Verify Installation

```python
import russo
print(russo.__doc__)
# russo — testing framework for LLM tool-call accuracy.
```
