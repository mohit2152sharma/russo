"""Tests for built-in response parsers."""

from __future__ import annotations

from russo._types import AgentResponse
from russo.parsers.mapping import JsonResponseParser


class TestJsonResponseParserDefaults:
    """Default field names: tool_calls / name / arguments, multi-call list."""

    def test_standard_format(self) -> None:
        raw = {"tool_calls": [{"name": "book_flight", "arguments": {"from": "NYC", "to": "LA"}}]}
        result = JsonResponseParser().parse(raw)
        assert isinstance(result, AgentResponse)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "book_flight"
        assert result.tool_calls[0].arguments == {"from": "NYC", "to": "LA"}

    def test_multiple_tool_calls(self) -> None:
        raw = {
            "tool_calls": [
                {"name": "book_flight", "arguments": {"from": "NYC", "to": "LA"}},
                {"name": "book_hotel", "arguments": {"city": "LA"}},
            ]
        }
        result = JsonResponseParser().parse(raw)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[1].name == "book_hotel"

    def test_empty_tool_calls_list(self) -> None:
        raw = {"tool_calls": []}
        result = JsonResponseParser().parse(raw)
        assert result.tool_calls == []

    def test_key_missing_returns_empty(self) -> None:
        raw = {"response": "Hello"}
        result = JsonResponseParser().parse(raw)
        assert result.tool_calls == []

    def test_raw_preserved(self) -> None:
        raw = {"tool_calls": [{"name": "fn", "arguments": {}}]}
        result = JsonResponseParser().parse(raw)
        assert result.raw is raw


class TestJsonResponseParserCustomKeys:
    """Custom top-level key and per-call field names."""

    def test_custom_tool_calls_key(self) -> None:
        raw = {"toolCall": [{"name": "search", "arguments": {"q": "hello"}}]}
        parser = JsonResponseParser(tool_calls_key="toolCall")
        result = parser.parse(raw)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"

    def test_custom_name_key(self) -> None:
        raw = {"tool_calls": [{"function": "get_weather", "arguments": {"city": "Tokyo"}}]}
        parser = JsonResponseParser(name_key="function")
        result = parser.parse(raw)
        assert result.tool_calls[0].name == "get_weather"

    def test_custom_arguments_key(self) -> None:
        raw = {"tool_calls": [{"name": "fn", "params": {"x": 1}}]}
        parser = JsonResponseParser(arguments_key="params")
        result = parser.parse(raw)
        assert result.tool_calls[0].arguments == {"x": 1}

    def test_all_custom_keys(self) -> None:
        raw = {"calls": [{"fn": "do_thing", "args": {"a": "b"}}]}
        parser = JsonResponseParser(tool_calls_key="calls", name_key="fn", arguments_key="args")
        result = parser.parse(raw)
        assert result.tool_calls[0].name == "do_thing"
        assert result.tool_calls[0].arguments == {"a": "b"}


class TestJsonResponseParserNestedPath:
    """Dot-notation key paths for nested response structures."""

    def test_one_level_nesting(self) -> None:
        raw = {"result": {"tool_calls": [{"name": "fn", "arguments": {}}]}}
        parser = JsonResponseParser(tool_calls_key="result.tool_calls")
        result = parser.parse(raw)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "fn"

    def test_two_level_nesting(self) -> None:
        raw = {"response": {"data": {"calls": [{"name": "fn", "arguments": {"k": "v"}}]}}}
        parser = JsonResponseParser(tool_calls_key="response.data.calls")
        result = parser.parse(raw)
        assert result.tool_calls[0].arguments == {"k": "v"}

    def test_missing_intermediate_key(self) -> None:
        raw = {"response": {}}
        parser = JsonResponseParser(tool_calls_key="response.data.calls")
        result = parser.parse(raw)
        assert result.tool_calls == []


class TestJsonResponseParserSingle:
    """single=True: the key points to one tool call dict, not a list."""

    def test_single_tool_call_object(self) -> None:
        raw = {"toolCall": {"name": "book_flight", "arguments": {"from": "NYC", "to": "LA"}}}
        parser = JsonResponseParser(tool_calls_key="toolCall", single=True)
        result = parser.parse(raw)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "book_flight"

    def test_single_with_custom_keys(self) -> None:
        raw = {"call": {"fn": "search", "params": {"q": "test"}}}
        parser = JsonResponseParser(tool_calls_key="call", name_key="fn", arguments_key="params", single=True)
        result = parser.parse(raw)
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"q": "test"}

    def test_single_missing_key(self) -> None:
        raw = {"other": "value"}
        parser = JsonResponseParser(tool_calls_key="toolCall", single=True)
        result = parser.parse(raw)
        assert result.tool_calls == []


class TestJsonResponseParserWebSocketList:
    """When aggregated WebSocket messages produce a list, search each item."""

    def test_tool_call_in_first_message(self) -> None:
        raw = [
            {"tool_calls": [{"name": "fn", "arguments": {"x": 1}}]},
            {"text": "done"},
        ]
        result = JsonResponseParser().parse(raw)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "fn"

    def test_tool_call_in_later_message(self) -> None:
        raw = [
            {"status": "processing"},
            {"tool_calls": [{"name": "fn", "arguments": {}}]},
        ]
        result = JsonResponseParser().parse(raw)
        assert len(result.tool_calls) == 1

    def test_no_tool_calls_in_any_message(self) -> None:
        raw = [{"text": "hello"}, {"text": "world"}]
        result = JsonResponseParser().parse(raw)
        assert result.tool_calls == []

    def test_raw_is_original_list(self) -> None:
        raw = [{"tool_calls": [{"name": "fn", "arguments": {}}]}]
        result = JsonResponseParser().parse(raw)
        assert result.raw is raw


class TestJsonResponseParserArgumentTypes:
    """Arguments may arrive as a dict or as a JSON-encoded string."""

    def test_arguments_as_dict(self) -> None:
        raw = {"tool_calls": [{"name": "fn", "arguments": {"key": "val"}}]}
        result = JsonResponseParser().parse(raw)
        assert result.tool_calls[0].arguments == {"key": "val"}

    def test_arguments_as_json_string(self) -> None:
        raw = {"tool_calls": [{"name": "fn", "arguments": '{"key": "val"}'}]}
        result = JsonResponseParser().parse(raw)
        assert result.tool_calls[0].arguments == {"key": "val"}

    def test_arguments_missing_defaults_to_empty(self) -> None:
        raw = {"tool_calls": [{"name": "fn"}]}
        result = JsonResponseParser().parse(raw)
        assert result.tool_calls[0].arguments == {}

    def test_invalid_json_string_defaults_to_empty(self) -> None:
        raw = {"tool_calls": [{"name": "fn", "arguments": "not-json"}]}
        result = JsonResponseParser().parse(raw)
        assert result.tool_calls[0].arguments == {}
