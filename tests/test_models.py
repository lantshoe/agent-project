"""
tests/test_models.py

Unit tests for agents/core/models.py
No LLM calls, no network, no filesystem — pure logic tests.

Run with:
    pytest tests/test_models.py -v
"""

import time
import pytest
from agents.core.models import (
    BudgetState,
    StepRecord,
    ToolResult,
    VerifierDecision,
    PlanStep,
    EvaluationResult,
)


class TestBudgetState:

    def test_default_values(self):
        b = BudgetState()
        assert b.max_steps == 20
        assert b.max_seconds == 300
        assert b.steps_used == 0

    def test_custom_values(self):
        b = BudgetState(max_steps=5, max_seconds=60)
        assert b.max_steps == 5
        assert b.max_seconds == 60

    def test_not_exhausted_initially(self):
        b = BudgetState()
        assert not b.is_exhausted
        assert not b.is_over_steps
        assert not b.is_over_time

    def test_steps_remaining(self):
        b = BudgetState(max_steps=10)
        assert b.steps_remaining == 10
        b2 = b.increment()
        assert b2.steps_remaining == 9

    def test_increment_is_immutable(self):
        """increment() returns a new instance — original is unchanged."""
        b = BudgetState(max_steps=5)
        b2 = b.increment()
        assert b.steps_used == 0      # original unchanged
        assert b2.steps_used == 1     # new instance updated

    def test_increment_chain(self):
        b = BudgetState(max_steps=3)
        b = b.increment().increment().increment()
        assert b.steps_used == 3
        assert b.is_over_steps
        assert b.is_exhausted

    def test_steps_remaining_floors_at_zero(self):
        b = BudgetState(max_steps=2)
        b = b.increment().increment().increment()
        assert b.steps_remaining == 0  # not negative

    def test_is_over_time(self):
        b = BudgetState(max_seconds=1)
        time.sleep(2)
        assert b.is_over_time
        assert b.is_exhausted

    def test_seconds_elapsed_increases(self):
        b = BudgetState()
        time.sleep(0.05)
        assert b.seconds_elapsed >= 0.05

    def test_summary_format(self):
        b = BudgetState(max_steps=10, max_seconds=60)
        b = b.increment().increment()
        summary = b.summary()
        assert "2/10" in summary
        assert "60" in summary



class TestToolResult:

    def test_from_raw_json_string(self):
        raw = '{"status": "success", "output": "done"}'
        result = ToolResult.from_raw(raw)
        assert result.status == "success"
        assert result.output["output"] == "done"

    def test_from_raw_plain_text(self):
        """Plain text (e.g. web_search output) should be treated as success."""
        result = ToolResult.from_raw("Here are the search results...")
        assert result.status == "success"
        assert result.output == "Here are the search results..."
        assert result.error is None

    def test_from_raw_dict(self):
        result = ToolResult.from_raw({"status": "success", "data": [1, 2, 3]})
        assert result.status == "success"
        assert result.output["data"] == [1, 2, 3]

    def test_from_raw_dict_with_error(self):
        result = ToolResult.from_raw({"status": "error", "error": "File not found"})
        assert result.status == "error"
        assert result.error == "File not found"

    def test_from_raw_mcp_block_list(self):
        """MCP tools return content blocks — should be unwrapped."""
        raw = [
            {"type": "text", "text": '{"status": "success", "output": "file written"}'},
        ]
        result = ToolResult.from_raw(raw)
        assert result.status == "success"

    def test_from_raw_mcp_multiple_blocks(self):
        raw = [
            {"type": "text", "text": "line one"},
            {"type": "text", "text": "line two"},
        ]
        result = ToolResult.from_raw(raw)
        assert result.status == "success"
        assert "line one" in result.output
        assert "line two" in result.output

    def test_from_raw_empty_string(self):
        result = ToolResult.from_raw("")
        assert result.status == "success"
        assert result.output == ""

    def test_from_raw_dict_missing_status_defaults_to_success(self):
        result = ToolResult.from_raw({"data": "something"})
        assert result.status == "success"

    def test_from_raw_json_error_status(self):
        raw = '{"status": "error", "error": "timeout"}'
        result = ToolResult.from_raw(raw)
        assert result.status == "error"
        assert result.error == "timeout"

    def test_from_raw_non_string_non_dict(self):
        """Integers, booleans, etc. should return a success with str output."""
        result = ToolResult.from_raw(42)
        assert result.status == "success"
        assert result.output == "42"



class TestStepRecord:

    def test_succeeded_true_on_success(self):
        step = StepRecord(tool="web_search", status="success")
        assert step.succeeded is True

    def test_succeeded_false_on_error(self):
        step = StepRecord(tool="web_search", status="error", error="timeout")
        assert step.succeeded is False

    def test_default_args_is_empty_dict(self):
        step = StepRecord(tool="calculator", status="success")
        assert step.args == {}

    def test_output_stored(self):
        step = StepRecord(tool="calculator", status="success", output="2+2=4")
        assert step.output == "2+2=4"

    def test_error_stored(self):
        step = StepRecord(tool="write_file", status="error", error="permission denied")
        assert step.error == "permission denied"

    def test_model_dump_roundtrip(self):
        step = StepRecord(
            tool="calculator",
            args={"expression": "2+2"},
            status="success",
            output="4",
        )
        d = step.model_dump()
        restored = StepRecord(**d)
        assert restored.tool == step.tool
        assert restored.args == step.args
        assert restored.status == step.status



class TestVerifierDecision:

    def test_should_retry_true(self):
        d = VerifierDecision(action="retry", reason="failed", hint="check args")
        assert d.should_retry is True
        assert d.should_stop is False

    def test_should_stop_true_on_escalate(self):
        d = VerifierDecision(action="escalate", reason="critical failure")
        assert d.should_stop is True
        assert d.should_retry is False

    def test_continue_neither_retry_nor_stop(self):
        d = VerifierDecision(action="continue")
        assert d.should_retry is False
        assert d.should_stop is False

    def test_skip_neither_retry_nor_stop(self):
        d = VerifierDecision(action="skip", reason="non-critical")
        assert d.should_retry is False
        assert d.should_stop is False

    def test_invalid_action_raises(self):
        with pytest.raises(Exception):
            VerifierDecision(action="unknown_action")

    def test_hint_defaults_to_empty_string(self):
        d = VerifierDecision(action="continue")
        assert d.hint == ""

    def test_reason_defaults_to_empty_string(self):
        d = VerifierDecision(action="skip")
        assert d.reason == ""




class TestPlanStep:

    def test_basic_creation(self):
        step = PlanStep(step=1, tool="web_search", reason="search for info")
        assert step.step == 1
        assert step.tool == "web_search"
        assert step.depends_on == []

    def test_depends_on_stored(self):
        step = PlanStep(step=2, tool="write_file", reason="save results", depends_on=[1])
        assert step.depends_on == [1]

    def test_model_dump(self):
        step = PlanStep(step=1, tool="calculator", reason="compute", depends_on=[])
        d = step.model_dump()
        assert d["tool"] == "calculator"
        assert d["depends_on"] == []



class TestEvaluationResult:

    def test_basic_creation(self):
        e = EvaluationResult(quality="good", reason="all steps succeeded")
        assert e.quality == "good"
        assert e.suggestions == []

    def test_suggestions_stored(self):
        e = EvaluationResult(quality="partial", reason="one step failed", suggestions=["retry step 2"])
        assert len(e.suggestions) == 1
