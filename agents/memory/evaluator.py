import json
from langchain_core.messages import SystemMessage, HumanMessage
from agents.core.llm import get_llm
from agents.core.logger import get_logger
from agents.core.skill_enum import Skill

logger = get_logger("evaluator")


EVALUATOR_SYSTEM_PROMPT = """You are an agent execution evaluator.
Given a task, the execution plan, completed steps, and final answer,
evaluate the execution quality and extract learnings.

Respond ONLY with a valid JSON object with these exact fields:
{
  "quality": "good" | "partial" | "failed",
  "efficiency_score": 0-10,
  "what_worked": ["list of things that worked well"],
  "what_failed": ["list of things that went wrong"],
  "learnings": ["actionable lessons for future similar tasks"],
  "suggested_plan": ["improved step sequence for this type of task"]
}
"""

EVALUATOR_USER_PROMPT = """Task: {user_input}
Planned steps:
{plan_text}

Executed steps:
{steps_text}

Final answer: {final_answer}
Evaluate this execution."""


def evaluate_execution_quality(user_input: str, plan: list, completed_steps: list, final_answer: str) -> dict:
    """
    LLM self evaluates the execution quality.
    Return structured evaluation with learning.
    """
    llm = get_llm(skill=Skill.REASONING)

    plan_text = "\n".join([
        f" step {s['step']}: {s['tool']}-{s['reason']}"
        for s in plan
    ]) if plan else "No plan"

    steps_text = "\n".join([
        f" {s.get('tool')}:  {'success' if s.get('status') == 'success' else 'failed'} "
        for s in completed_steps
    ]) if completed_steps else "No steps recorded"
    try:
        formatted = EVALUATOR_USER_PROMPT.format(user_input=user_input, plan_text=plan_text,steps_text = steps_text,final_answer=final_answer)
        response = llm.invoke([
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=formatted),
        ])
        content = response.content.strip()
        logger.debug(f"content: {content}")
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        evaluation = json.loads(content.strip())
        logger.debug(f"Execution quality: {evaluation.get('quality')} "
                     f"(score: {evaluation.get('efficiency_score')})")
        logger.debug(f"Evaluation quality: {evaluation} ")
        return evaluation
    except Exception as e:
        logger.error(f"Evaluator failed: {e}")
        return _rule_based_evaluation(completed_steps)


def _rule_based_evaluation(completed_steps: list) -> dict:
    """
    Fallback rule-based evaluator when LLM evaluation fails.
    """
    total   = len(completed_steps)
    success = sum(1 for s in completed_steps if s.get("status") == "success")
    failed  = total - success

    if failed == 0:
        quality = "good"
        score   = 9
    elif success > failed:
        quality = "partial"
        score   = 5
    else:
        quality = "failed"
        score   = 2

    return {
        "quality":          quality,
        "efficiency_score": score,
        "what_worked":      [s["tool"] for s in completed_steps if s.get("status") == "success"],
        "what_failed":      [s["tool"] for s in completed_steps if s.get("status") != "success"],
        "learnings":        [],
        "suggested_plan":   []
    }