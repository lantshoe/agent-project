"""
after every agent run, writes a log

"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.core.logger import get_logger
from agents.core.models import StepRecord, BudgetState, VerifierDecision

logger = get_logger("run_logger")
LOG_DIR = Path(os.getenv("AGENT_LOG_DIR", "./logs/runs"))

class RunLogger:
    """
    Collects structured data during an agent run and writes
    a JSON summary when the run completes.

    Usage in agent.py:
        run_log = RunLogger(user_id=user_id, user_input=user_input)
        run_log.record_llm_call(input_tokens=310, output_tokens=95)
        run_log.record_step(step, verifier_decision, duration_ms=840)
        run_log.finish(final_answer=answer, budget=budget, evaluation=evaluation)
    """
    def __init__(self, user_id: str,  user_input: str):
        self.user_id = user_id
        self.user_input = user_input
        self.started_at = datetime.now(timezone.utc)
        self.run_id = self._make_run_id()

        self._steps:list[dict] = []
        self._llm_calls: list[dict] = []
        self._call_index = 0
        logger.debug(f"[run_logger] Run started: {self.run_id}")

    def record_llm_call(self, input_tokens:int=0, output_tokens:int=0):
        self._call_index += 1
        self._llm_calls.append({
            "call":self._call_index,
            "input":input_tokens,
            "output":output_tokens,
            "total": input_tokens + output_tokens,
        })

    def record_step(self, step:StepRecord, decision:VerifierDecision | None = None, duration_ms:int=0):
        """called after every tool execution and verifier decision"""
        self._steps.append({
            "index": len(self._steps) + 1,
            "tool": step.tool,
            "status": step.status,
            "verifier": decision.action if decision else "none",
            "retry_count":0,
            "duration_ms":duration_ms,
            "error":step.error,
        })

    def increment_retry(self, tool_name:str) -> None:
        for entry in reversed(self._steps):
            if entry["tool"] == tool_name:
                entry["retry_count"] += 1
                break
    def finish(self, final_answer:str, budget:BudgetState, evaluation:dict[str, Any]) -> Path:
        """
        write the complete run summary to disk
        return the path pf the written log file
        """
        finished_at = datetime.now(timezone.utc)
        duration_s = (finished_at - self.started_at).total_seconds()

        total_input = sum(c["input"] for c in self._llm_calls)
        total_output = sum(c["output"] for c in self._llm_calls)
        verifier_summary = {"continue": 0, "retry": 0, "skip": 0, "escalate": 0, "none": 0}
        for s in self._steps:
            key = s["verifier"]
            verifier_summary[key] = verifier_summary.get(key, 0) + 1
        verifier_summary.pop("none", None)

        payload = {
            "run_id": self.run_id,
            "user_id": self.user_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_s": round(duration_s, 2),
            "user_input": self.user_input,
            "final_answer": final_answer,
            "budget": {
                "steps_used": budget.steps_used,
                "max_steps": budget.max_steps,
                "seconds_used": round(budget.seconds_elapsed, 2),
                "max_seconds": budget.max_seconds,
                "exhausted": budget.is_exhausted,
            },
            "tokens": {
                "total_input": total_input,
                "total_output": total_output,
                "total": total_input + total_output,
                "by_call": self._llm_calls,
            },
            "steps": self._steps,
            "verifier_summary": verifier_summary,
            "evaluation": {
                "quality": evaluation.get("quality"),
                "efficiency_score": evaluation.get("efficiency_score"),
            },
        }

        log_path = self._write(payload)
        self._log_summary(payload)
        return log_path

    def _make_run_id(self) -> str:
        ts = self.started_at.strftime("%Y-%m-%d_%H-%M-%S")
        return f"{ts}_{self.user_id}"

    def _write(self, payload: dict) -> Path:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f"{self.run_id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        logger.debug(f"[run_logger] Written: {path}")
        return path

    def _log_summary(self, payload: dict) -> None:
        """Prints a compact one-line summary to the console logger."""
        b = payload["budget"]
        t = payload["tokens"]
        vs = payload["verifier_summary"]
        q = payload["evaluation"].get("quality", "?")

        logger.info(
            f"[run] {self.run_id} | "
            f"quality={q} | "
            f"steps={b['steps_used']}/{b['max_steps']} | "
            f"time={b['seconds_used']}s | "
            f"tokens={t['total']} ({t['total_input']}in/{t['total_output']}out) | "
            f"verifier={vs}"
        )