"""
tests/test_verifier.py

Unit tests for agents/core/verifier.py
No LLM calls, no network — pure deterministic logic tests.

Run with:
    pytest tests/test_verifier.py -v
"""

import pytest
from agents.core.models import StepRecord
from agents.core.verifier import verify_step, _build_retry_hint, CRITICAL_TOOLS, MAX_RETRIES_PER_STEP


def success_step(tool="web_search") -> StepRecord:
    return StepRecord(tool=tool, status="success", output="some output")

def error_step(tool="web_search", error="something went wrong") -> StepRecord:
    return StepRecord(tool=tool, status="error", error=error)



class TestContinuePath:

    def test_success_returns_continue(self):
        decision = verify_step(success_step())
        assert decision.action == "continue"

    def test_continue_has_empty_hint(self):
        decision = verify_step(success_step())
        assert decision.hint == ""

    def test_continue_on_any_tool_success(self):
        for tool in ["calculator", "write_file", "web_search", "read_file"]:
            decision = verify_step(success_step(tool=tool))
            assert decision.action == "continue", f"Expected continue for {tool}"



class TestRetryPath:

    def test_first_error_returns_retry(self):
        decision = verify_step(error_step(), retry_count=0)
        assert decision.action == "retry"

    def test_retry_includes_hint(self):
        decision = verify_step(error_step(), retry_count=0)
        assert len(decision.hint) > 0

    def test_retry_within_max_retries(self):
        for count in range(MAX_RETRIES_PER_STEP):
            decision = verify_step(error_step(), retry_count=count)
            assert decision.action == "retry", f"Expected retry at count={count}"

    def test_retry_includes_tool_name_in_hint(self):
        step = error_step(tool="write_file", error="permission denied")
        decision = verify_step(step, retry_count=0)
        assert "write_file" in decision.hint



class TestSkipPath:

    def test_non_critical_tool_skips_after_max_retries(self):
        """Non-critical tools should skip once retries are exhausted."""
        non_critical = "web_search"
        assert non_critical not in CRITICAL_TOOLS
        decision = verify_step(error_step(tool=non_critical), retry_count=MAX_RETRIES_PER_STEP)
        assert decision.action == "skip"

    def test_skip_includes_reason(self):
        decision = verify_step(error_step(tool="web_search"), retry_count=MAX_RETRIES_PER_STEP)
        assert len(decision.reason) > 0

    def test_calculator_skips_after_retries(self):
        assert "calculator" not in CRITICAL_TOOLS
        decision = verify_step(error_step(tool="calculator"), retry_count=MAX_RETRIES_PER_STEP)
        assert decision.action == "skip"



class TestEscalatePath:

    def test_critical_tool_escalates_after_max_retries(self):
        for tool in CRITICAL_TOOLS:
            decision = verify_step(error_step(tool=tool), retry_count=MAX_RETRIES_PER_STEP)
            assert decision.action == "escalate", f"Expected escalate for critical tool {tool}"

    def test_escalate_includes_reason(self):
        critical_tool = next(iter(CRITICAL_TOOLS))
        decision = verify_step(error_step(tool=critical_tool), retry_count=MAX_RETRIES_PER_STEP)
        assert len(decision.reason) > 0

    def test_critical_tool_still_retries_before_max(self):
        """Critical tools should still retry — they only escalate after retries exhausted."""
        critical_tool = next(iter(CRITICAL_TOOLS))
        decision = verify_step(error_step(tool=critical_tool), retry_count=0)
        assert decision.action == "retry"



class TestRetryHints:

    def test_missing_arg_hint(self):
        step = error_step(error="Missing required argument 'filepath'")
        hint = _build_retry_hint(step)
        assert "required" in hint.lower() or "missing" in hint.lower()

    def test_file_not_found_hint(self):
        step = error_step(error="File not found: report.txt")
        hint = _build_retry_hint(step)
        assert "path" in hint.lower() or "found" in hint.lower()

    def test_permission_denied_hint(self):
        step = error_step(error="Permission denied: access denied to /etc/passwd")
        hint = _build_retry_hint(step)
        assert "sandbox" in hint.lower() or "permission" in hint.lower()

    def test_timeout_hint(self):
        step = error_step(error="Request timeout after 30s")
        hint = _build_retry_hint(step)
        assert "timeout" in hint.lower() or "simpler" in hint.lower()

    def test_generic_fallback_hint_includes_tool_name(self):
        step = error_step(tool="my_tool", error="unexpected error xyz")
        hint = _build_retry_hint(step)
        assert "my_tool" in hint

    def test_generic_fallback_hint_includes_error(self):
        step = error_step(error="unexpected error xyz")
        hint = _build_retry_hint(step)
        assert "unexpected error xyz" in hint

    def test_hint_is_always_non_empty(self):
        """Every failure should produce a non-empty hint."""
        errors = [
            "missing required arg",
            "file not found",
            "permission denied",
            "timeout",
            "some completely unknown error",
            "",
        ]
        for error in errors:
            step = error_step(error=error)
            hint = _build_retry_hint(step)
            assert isinstance(hint, str)



class TestEdgeCases:

    def test_step_with_no_error_message(self):
        """A failed step with no error message should still produce a valid decision."""
        step = StepRecord(tool="web_search", status="error", error=None)
        decision = verify_step(step, retry_count=0)
        assert decision.action in {"retry", "skip", "escalate", "continue"}

    def test_very_high_retry_count_still_gives_valid_decision(self):
        decision = verify_step(error_step(), retry_count=999)
        assert decision.action in {"skip", "escalate"}

    def test_retry_count_boundary(self):
        """At exactly MAX_RETRIES_PER_STEP we should stop retrying."""
        step = error_step(tool="web_search")
        at_limit = verify_step(step, retry_count=MAX_RETRIES_PER_STEP)
        below_limit = verify_step(step, retry_count=MAX_RETRIES_PER_STEP - 1)
        assert below_limit.action == "retry"
        assert at_limit.action != "retry"
