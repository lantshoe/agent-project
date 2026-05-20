from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field

class StepRecord(BaseModel):
    """
    Records the outcome of a single tool execution.
    """
    tool: str
    args: dict[str, Any]=Field(default_factory=dict)
    status: str
    output: Any=None
    error: str | None =None

    @property
    def succeeded(self) -> bool:
        return self.status == "success"


class PlanStep(BaseModel):
    """
    A single step in the planner's execution plan.
    """
    step: int
    tool: str
    reason: str
    depends_on: list[int] = Field(default_factory=list)

class EvaluationResult(BaseModel):
    """
    result of evaluating agent execution quality.
    """
    quality:str
    reason:str
    suggestions: list[str] = Field(default_factory=list)

class ToolResult(BaseModel):
    """
    Normalised output from any tool, regardless of return format.
    """
    status: str
    output: Any=None
    error: str | None=None

    @classmethod
    def from_raw(cls, raw: Any) -> "ToolResult":
        """
        Safely parse any tool output into a ToolResult.
        Handles JSON dicts, plain strings, and MCP content blocks.
        """
        import json
        if isinstance(raw, list):
            texts = []
            for block in raw:
                if isinstance(block, dict) and "text" in block:
                    texts.append(block["text"])
                elif hasattr(block, "text"):
                    texts.append(block.text)
            raw = "\n".join(texts) if texts else str(raw)

        if isinstance(raw, dict):
            return cls(
                status=raw.get("status", "success"),
                output=raw,
                error=raw.get("error"),
            )

        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return cls(
                    status=parsed.get("status", "success"),
                    output=parsed,
                    error=parsed.get("error"),
                )
            except json.JSONDecodeError:
                # Plain text output — treat as successful
                return cls(status="success", output=raw)

            # Fallback for any other type
        return cls(status="success", output=str(raw))

class BudgetState(BaseModel):
    """
    Tracks resource consumption during a single agent run.

    The budget guard checks this after every step
    and triggers a graceful exit when any limit is breached,
    rather than letting the agent loop forever or run up unbounded LLM costs.
    """
    max_steps: int = 20
    max_seconds: int = 300
    started_at: datetime = Field(default_factory=datetime.now)
    step_used: int = 0

    @property
    def steps_remaining(self) -> int:
        return max(0, self.max_steps - self.steps_used)

    @property
    def seconds_elapsed(self) -> float:
        return (datetime.now() - self.started_at).total_seconds()

    @property
    def is_over_steps(self) -> bool:
        return self.steps_used >= self.max_steps

    @property
    def is_over_time(self) -> bool:
        return self.seconds_elapsed >= self.max_seconds

    @property
    def is_exhausted(self) -> bool:
        return self.is_over_steps or self.is_over_time

    def increment(self) -> "BudgetState":
        """Returns a new BudgetState with steps_used incremented by 1."""
        return self.model_copy(update={"steps_used": self.steps_used + 1})

    def summary(self) -> str:
        return (
            f"steps {self.steps_used}/{self.max_steps} | "
            f"time {self.seconds_elapsed:.1f}s/{self.max_seconds:.0f}s"
        )


class VerifierDecision(BaseModel):
    """
    The verifier's decision after inspecting a tool result.
    After every tool execution, the verifier checks the StepRecord
    and decides what the agent should do next.

    continue: result is good, proceed to next call
    retry: result is wrong, but can try again, retry this step with a hint about what went wrong.
    skip: skip failed but not critical step
    escalate: unrecoverable failure, return error to user.
    """

    action: Literal["continue", "retry", "skip", "escalate"]
    reason: str = ""
    hint: str = ""

    @property
    def should_retry(self) -> bool:
        return self.action == "retry"

    @property
    def should_stop(self) -> bool:
        return self.action == "escalate"
