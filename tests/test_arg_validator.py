"""
tests/test_arg_validator.py

Unit tests for agents/core/arg_validator.py
Uses real LangChain tool definitions — no LLM calls, no network.

Run with:
    pytest tests/test_arg_validator.py -v
"""

import pytest
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from agents.core.arg_validator import (
    get_tool_schema,
    validate_and_fix_args,
)


# ── Fixtures — minimal fake tools with typed schemas ──────────────────────────

class SearchSchema(BaseModel):
    query: str = Field(..., description="Search query string")
    max_results: int = Field(default=5, description="Max number of results")

class SearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web"
    args_schema: type[BaseModel] = SearchSchema

    def _run(self, query: str, max_results: int = 5) -> str:
        return f"Results for: {query}"

    async def _arun(self, query: str, max_results: int = 5) -> str:
        return self._run(query, max_results)


class CalcSchema(BaseModel):
    expression: str = Field(..., description="Math expression to evaluate")

class CalcTool(BaseTool):
    name: str = "calculator"
    description: str = "Evaluate math expressions"
    args_schema: type[BaseModel] = CalcSchema

    def _run(self, expression: str) -> str:
        return str(eval(expression))

    async def _arun(self, expression: str) -> str:
        return self._run(expression)


class MultiTypeSchema(BaseModel):
    name: str = Field(..., description="Name string")
    count: int = Field(..., description="Integer count")
    enabled: bool = Field(default=True, description="Boolean flag")
    tags: list = Field(default_factory=list, description="List of tags")

class MultiTypeTool(BaseTool):
    name: str = "multi_type_tool"
    description: str = "Tool with multiple arg types"
    args_schema: type[BaseModel] = MultiTypeSchema

    def _run(self, **kwargs) -> str:
        return "ok"

    async def _arun(self, **kwargs) -> str:
        return "ok"


class NoSchemaTool(BaseTool):
    name: str = "no_schema_tool"
    description: str = "Tool with no args schema"

    def _run(self, **kwargs) -> str:
        return "ok"

    async def _arun(self, **kwargs) -> str:
        return "ok"


@pytest.fixture
def search_tool():
    return SearchTool()

@pytest.fixture
def calc_tool():
    return CalcTool()

@pytest.fixture
def multi_tool():
    return MultiTypeTool()

@pytest.fixture
def no_schema_tool():
    return NoSchemaTool()


# ── get_tool_schema ───────────────────────────────────────────────────────────

class TestGetToolSchema:

    def test_returns_fields_for_tool_with_schema(self, search_tool):
        schema = get_tool_schema(search_tool)
        assert "query" in schema
        assert "max_results" in schema

    def test_required_field_marked_correctly(self, search_tool):
        schema = get_tool_schema(search_tool)
        assert schema["query"]["required"] is True

    def test_optional_field_marked_correctly(self, search_tool):
        schema = get_tool_schema(search_tool)
        assert schema["max_results"]["required"] is False

    def test_description_extracted(self, search_tool):
        schema = get_tool_schema(search_tool)
        assert "Search query" in schema["query"]["description"]

    def test_returns_empty_dict_for_no_schema_tool(self, no_schema_tool):
        schema = get_tool_schema(no_schema_tool)
        assert schema == {}

    def test_type_extracted(self, calc_tool):
        schema = get_tool_schema(calc_tool)
        assert schema["expression"]["type"] == "string"

    def test_multi_type_schema(self, multi_tool):
        schema = get_tool_schema(multi_tool)
        assert schema["name"]["type"] == "string"
        assert schema["count"]["type"] == "integer"
        assert schema["enabled"]["type"] == "boolean"


# ── validate_and_fix_args — happy path ───────────────────────────────────────

class TestValidateHappyPath:

    def test_valid_args_pass_through_unchanged(self, search_tool):
        args = {"query": "AI news", "max_results": 3}
        fixed, warnings = validate_and_fix_args(search_tool, args)
        assert fixed["query"] == "AI news"
        assert fixed["max_results"] == 3
        assert warnings == []

    def test_no_schema_tool_returns_args_unchanged(self, no_schema_tool):
        args = {"anything": "goes"}
        fixed, warnings = validate_and_fix_args(no_schema_tool, args)
        assert fixed == args
        assert warnings == []


# ── validate_and_fix_args — unknown args stripped ────────────────────────────

class TestStripUnknownArgs:

    def test_unknown_arg_is_stripped(self, calc_tool):
        args = {"expression": "2+2", "unknown_param": "value"}
        fixed, warnings = validate_and_fix_args(calc_tool, args)
        assert "unknown_param" not in fixed
        assert "expression" in fixed

    def test_warning_issued_for_stripped_arg(self, calc_tool):
        args = {"expression": "2+2", "bad_arg": "oops"}
        _, warnings = validate_and_fix_args(calc_tool, args)
        assert any("bad_arg" in w for w in warnings)

    def test_multiple_unknown_args_all_stripped(self, calc_tool):
        args = {"expression": "2+2", "a": 1, "b": 2, "c": 3}
        fixed, warnings = validate_and_fix_args(calc_tool, args)
        assert list(fixed.keys()) == ["expression"]
        assert len([w for w in warnings if "Stripped" in w]) == 3


# ── validate_and_fix_args — type coercion ────────────────────────────────────

class TestTypeCoercion:

    def test_int_float_coerced_to_int(self, search_tool):
        args = {"query": "hello", "max_results": 5.0}
        fixed, warnings = validate_and_fix_args(search_tool, args)
        assert fixed["max_results"] == 5
        assert isinstance(fixed["max_results"], int)
        assert any("int" in w for w in warnings)

    def test_non_string_coerced_to_string(self, calc_tool):
        args = {"expression": 100}
        fixed, warnings = validate_and_fix_args(calc_tool, args)
        assert fixed["expression"] == "100"
        assert isinstance(fixed["expression"], str)

    def test_string_bool_coerced_true(self, multi_tool):
        args = {"name": "test", "count": 1, "enabled": "true"}
        fixed, warnings = validate_and_fix_args(multi_tool, args)
        assert fixed["enabled"] is True

    def test_string_bool_coerced_false(self, multi_tool):
        args = {"name": "test", "count": 1, "enabled": "false"}
        fixed, warnings = validate_and_fix_args(multi_tool, args)
        assert fixed["enabled"] is False

    def test_string_bool_yes_coerced_true(self, multi_tool):
        args = {"name": "test", "count": 1, "enabled": "yes"}
        fixed, warnings = validate_and_fix_args(multi_tool, args)
        assert fixed["enabled"] is True

    def test_string_list_coerced(self, multi_tool):
        args = {"name": "test", "count": 1, "tags": '["a", "b"]'}
        fixed, warnings = validate_and_fix_args(multi_tool, args)
        assert fixed["tags"] == ["a", "b"]
        assert isinstance(fixed["tags"], list)


# ── validate_and_fix_args — missing required args ────────────────────────────

class TestMissingRequiredArgs:

    def test_missing_required_arg_reported(self, calc_tool):
        args = {}
        _, warnings = validate_and_fix_args(calc_tool, args)
        assert any("Missing required" in w for w in warnings)
        assert any("expression" in w for w in warnings)

    def test_missing_optional_arg_not_reported(self, search_tool):
        """Omitting an optional arg should not produce a warning."""
        args = {"query": "test"}   # max_results is optional
        _, warnings = validate_and_fix_args(search_tool, args)
        missing = [w for w in warnings if "Missing required" in w]
        assert missing == []

    def test_all_missing_required_args_reported(self, multi_tool):
        args = {}  # name and count are required
        _, warnings = validate_and_fix_args(multi_tool, args)
        missing = [w for w in warnings if "Missing required" in w]
        assert len(missing) >= 2


# ── Combined scenarios ────────────────────────────────────────────────────────

class TestCombinedScenarios:

    def test_strip_unknown_and_coerce_type(self, search_tool):
        args = {
            "query": "test",
            "max_results": 3.0,     # needs coercion
            "extra_param": "oops",  # needs stripping
        }
        fixed, warnings = validate_and_fix_args(search_tool, args)
        assert "extra_param" not in fixed
        assert fixed["max_results"] == 3
        assert isinstance(fixed["max_results"], int)

    def test_warnings_do_not_affect_valid_fields(self, search_tool):
        args = {"query": "hello", "max_results": 5, "junk": "ignore"}
        fixed, _ = validate_and_fix_args(search_tool, args)
        assert fixed["query"] == "hello"
        assert fixed["max_results"] == 5
