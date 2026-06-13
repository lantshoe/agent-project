from agents.core.models import StepRecord, VerifierDecision
from agents.core.logger import get_logger


logger = get_logger("verifier")

# Tools where a failure is critical — escalate immediately rather than skip.
# These match the actual Excel MCP tool names used in this project.
CRITICAL_TOOLS = {"write_file", "create_workbook", "write_sheet"}

# How many times a single step is allowed to retry before we skip or escalate
MAX_RETRIES_PER_STEP = 2


def verify_step(
    step: StepRecord,
    retry_count: int = 0,
) -> VerifierDecision:
    """
    Inspects a completed StepRecord and decides what happens next.

    This runs after every tool execution, before the next LLM call.
    it is deterministic

    Decision logic:
    success: continue
    error and retry left and recoverable: retry with hint
    error and retry exhausted and critical tool: escalate
    error and retry exhausted and non critical: skip
    """
    if step.succeeded:
        logger.debug(f"[verifier] {step.tool}: succeeded")
        return VerifierDecision(action="continue", reason="step completed successfully")

    error_msg = step.error or "unknown error"
    logger.warning(f"[verifier] {step.tool} failed: {error_msg}")
    if retry_count < MAX_RETRIES_PER_STEP:
        hint = _build_retry_hint(step)
        logger.debug(f"[verifier] retry ({retry_count + 1}/{MAX_RETRIES_PER_STEP}): {hint}")
        return VerifierDecision(action="retry", hint=hint, reason=f"step failed with {error_msg}")
    if step.tool in CRITICAL_TOOLS:
        logger.error(f"[verifier] → escalate: {step.tool} is critical and failed after retries")
        return VerifierDecision(
            action="escalate",
            reason=f"{step.tool} failed after {retry_count} retries: {error_msg}",
        )
    logger.warning(f"[verifier] → skip: {step.tool} failed but is non-critical")
    return VerifierDecision(
        action="skip",
        reason=f"Skipping {step.tool} after {retry_count} retries: {error_msg}",
    )


def _build_retry_hint(step: StepRecord) -> str:
    """
    Builds a targeted hint for the LLM based on the failure pattern.
    The hint is injected into the next LLM call to guide correction.
    """
    error = (step.error or "").lower()

    if "missing required" in error:
        return (
            f"The previous call to '{step.tool}' was missing required arguments. "
            f"Check the tool schema and provide all required fields. Error: {step.error}"
        )

    if "not found" in error or "no such file" in error:
        return (
            f"'{step.tool}' could not find the target. "
            f"Check the path or identifier and try again. Error: {step.error}"
        )

    if "permission" in error or "access denied" in error:
        return (
            f"'{step.tool}' was denied access. "
            f"Check the path is inside the sandbox. Error: {step.error}"
        )

    if "timeout" in error:
        return (
            f"'{step.tool}' timed out. "
            f"Try a simpler or more targeted query. Error: {step.error}"
        )

    if "decompress" in error or "invalid block" in error or "corrupt" in error:
        return (
            f"'{step.tool}' failed because the Excel file may be corrupted "
            f"from a concurrent write. Try create_workbook again with a "
            f"fresh filepath, or call list_sheets first to check the "
            f"file's current state. Error: {step.error}"
        )

    # Generic fallback
    return (
        f"'{step.tool}' failed with: {step.error}. "
        f"Review the arguments used ({step.args}) and try a different approach."
    )