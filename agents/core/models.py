from __future__ import annotations
from typing import Any
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


