"""
the supervisor agent - orchestrates specialist subagents to complete complex tasks
 that benefit from parallelism and specialization.

responsibilities:
    decompose user input into a typed delegationplan
    dispatch independent tasks in parallel via asyncio.gather
    handle partial failure - retry, skip or escalate per task
    aggregate subagent results into a final answer
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from agents.core.llm import get_llm
from agents.core.logger import get_logger
from agents.core.skill_enum import Skill
from agents.memory.evaluator import evaluate_execution_quality
from agents.memory.long_term import LongTermMemory
from agents.models.supervisor import (
    AgentType,
    DelegationPlan,
    SubagentResult,
    SupervisorState,
    Task,
)
import os
from pathlib import Path
from agents.execution.checkpointer import get_checkpointer_cm, make_run_id
from agents.myagent.base_agent import BaseAgent
from agents.myagent.base_agent import _extract_final_answer
from agents.myagent.data_agent import make_data_agent
from agents.myagent.research_agent import make_research_agent
from agents.tools.calculator import calculator
from agents.tools.mcp_client import get_mcp_config
from agents.tools.search import web_search

logger = get_logger("supervisor")

DECOMPOSE_SYSTEM_PROMPT = """You are a task supervisor. Given a user request,
decompose it into a list of tasks for specialist agents.

Available agent types:
- "research"   → web search, finding information, summarizing content
- "data"       → file operations, Excel, calculations, data processing
- "generalist" → anything that doesn't fit the above specialists

Each task must have:
- "task_id":    unique string like "task_1", "task_2"
- "agent_type": exactly one of "research", "data", "generalist"
- "instruction": clear, specific instruction for the subagent
- "depends_on": list of task_ids that must finish first ([] if independent)
- "priority":   integer 1-3 (3 = most critical)

RULES:
- Prefer parallel tasks (empty depends_on) when tasks are independent
- Only add depends_on when a task genuinely needs results from another
- Be specific in instructions — the subagent only sees its own instruction
- If a task needs output from a prior task, say so explicitly in the instruction
- If multiple tasks will write to the SAME Excel file, each task must write
  to a DIFFERENT sheet name to avoid concurrent writes to the same sheet.
  Specify the sheet name explicitly in each task's instruction.

Respond ONLY with a valid JSON object:
{
  "tasks": [...],
  "reasoning": "why you chose this decomposition",
  "strategy": "parallel" | "sequential" | "mixed"
}
"""

AGGREGATE_SYSTEM_PROMPT = """You are a supervisor synthesizing results
from specialist agents into a final answer for the user.

Write a clear, complete response that:
- Directly answers the user's original request
- Integrates all successful results naturally — include specific facts,
  figures, and names from the results, not just descriptions of what was done
- Acknowledges any tasks that failed or were partial
- Does not expose internal agent details to the user
"""

def decompose_task(user_input: str, memory_context: str = "") -> DelegationPlan:
    """
    call the llm to compose user input into a DelegationPlan.
    validates the output against typed models - bad agent types raise immediately.
    """
    llm = get_llm(skill=Skill.REASONING)
    memory_section = (
        f"Relevant context from past sessions:\n{memory_context}\n\n"
        if memory_context else ""
    )

    response = llm.invoke([
        SystemMessage(content=DECOMPOSE_SYSTEM_PROMPT),
        HumanMessage(content=f"{memory_section}User request: {user_input}"),
    ])

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        raw = json.loads(content.strip())

        tasks = [Task(**t) for t in raw["tasks"]]
        plan = DelegationPlan(
            tasks=tasks,
            reasoning=raw.get("reasoning", ""),
            strategy=raw.get("strategy", "parallel"),
        )
        logger.debug(f"[supervisor] plan: {plan.summary()}")
        logger.debug(f"[supervisor] reasoning: {plan.reasoning}")
        return plan
    except Exception as e:
        logger.error(f"[supervisor] Decomposition failed: {e} — falling back to generalist")
        return DelegationPlan(
            tasks=[Task(
                task_id="task_1",
                agent_type=AgentType.GENERALIST,
                instruction=user_input,
                priority=3,
            )],
            reasoning="Decomposition failed — delegating entire task to generalist",
            strategy="sequential",
        )

def make_agent_for_task(task: Task, mcp_tools: list) -> BaseAgent:
    if task.agent_type == AgentType.RESEARCH:
        return make_research_agent()
    if task.agent_type == AgentType.DATA:
        return make_data_agent(mcp_tools)

    GENERALIST_PROMPT = """
    You are a general-purpose agent. Use whatever tools are available to
    complete the task. Think step by step and use tools when needed.

    If your context includes "file_state", it tells you which files and
    sheets already exist. Use write_sheet (NOT create_workbook) to add a
    new sheet to an existing file — only use create_workbook if the file
    doesn't exist yet according to file_state.

    FILESYSTEM RULES:
    Sandbox root: agent_workspace
    All file paths must be relative and inside agent_workspace/.
    """
    return BaseAgent(
        tools=[web_search, calculator] + mcp_tools,
        system_prompt=GENERALIST_PROMPT,
        agent_type=AgentType.GENERALIST,
    )


async def dispatch_tasks(
        tasks: list[Task],
        mcp_tools: list,
        user_id: str,
        run_id: str,
        completed_results: list[SubagentResult],
        checkpointer
) -> list[SubagentResult]:
    """
    runs a batch of tasks concurrently with asyncio.gather()
    each task gets its own agent instance and its own context window.

    completed_results is a snapshot of results from prior rounds —
    passed explicitly (not read from shared state) to avoid the
    closure-timing bug where state.results was empty when read.
    """

    async def run_one(task: Task) -> SubagentResult:
        thread_id = f"{run_id}_{task.agent_type}_{task.task_id}"

        # Check if this task already completed in a previous run
        try:
            existing = await checkpointer.aget(
                {"configurable": {"thread_id": thread_id}}
            )
            if existing:
                channel_values = existing.get("channel_values", {})
                messages = channel_values.get("messages", [])
                completed_steps = channel_values.get("completed_steps", [])
                final_answer = _extract_final_answer(messages) if messages else ""
                has_real_answer = bool(final_answer and final_answer != "Task completed but no summary was generated.")
                has_steps = len(completed_steps) > 0

                if has_real_answer or has_steps:
                    logger.info(
                        f"[supervisor] ↩ SKIPPING task '{task.task_id}' "
                        f"({task.agent_type}) — completed in previous run | "
                        f"steps: {len(completed_steps)}"
                    )
                    return SubagentResult(
                        task_id=task.task_id,
                        agent_type=task.agent_type,
                        status="success",
                        answer=final_answer,
                        steps_taken=[s.tool for s in completed_steps],
                    )
                else:
                    logger.debug(
                        f"[supervisor] checkpoint exists for '{task.task_id}' "
                        f"but task didn't complete — resuming from checkpoint"
                    )
        except Exception as e:
            logger.debug(
                f"[supervisor] checkpoint check failed for '{task.task_id}': {e} — running fresh"
            )

        # No checkpoint found — inject dependency context and run normally
        for dep_id in task.depends_on:
            dep_result = next(
                (r for r in completed_results if r.task_id == dep_id), None
            )
            if dep_result and dep_result.usable:
                context_payload = {"summary": dep_result.answer}
                file_state = dep_result.metadata.get("file_state")
                if file_state:
                    context_payload["file_state"] = file_state
                task = task.with_context(dep_id, context_payload)

        agent = make_agent_for_task(task, mcp_tools)
        return await agent.run(task=task, user_id=user_id, run_id=run_id, checkpointer=checkpointer)

    logger.debug(f"[supervisor] Dispatching {len(tasks)} tasks in parallel: "
                 f"{[t.task_id for t in tasks]}")

    results = await asyncio.gather(
        *[run_one(t) for t in tasks],
        return_exceptions=True
    )

    clean = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"[supervisor] Task {tasks[i].task_id} raised: {r}")
            clean.append(SubagentResult(
                task_id=tasks[i].task_id,
                agent_type=tasks[i].agent_type,
                status="failed",
                answer="",
                error=str(r),
            ))
        else:
            clean.append(r)
    return clean

def handle_failures(
        results: list[SubagentResult],
        plan: DelegationPlan,
) -> tuple[list[SubagentResult], list[SubagentResult]]:
    """
    split results into usable and failed.
    decision logic:
        high priority task failed, log as critical
        low priority task failed, log as normal
        all task failed, escalate
    return usable and failed subagent results
    """
    usable = [r for r in results if r.usable]
    failed = [r for r in results if r.is_failed]
    for r in failed:
        task = plan.get_task(r.task_id)
        prio = task.priority if task else 1
        prefix = "CRITICAL" if prio >= 3 else "WARNING"
        logger.warning(
            f"[supervisor] {prefix}: task '{r.task_id}' "
            f"({r.agent_type}) failed: {r.error}"
        )

    if not usable:
        logger.error("[supervisor] All tasks failed — no usable results")
    return usable, failed

def aggregate_results(user_input: str, state: SupervisorState) -> str:
    """
    calls the LLM to synthesize all subagent results into a final answer.
    The LLM sees clean summaries - not raw tool call histories.
    """
    if not state.successful_results:
        return (
            "I was unable to complete your request. "
            f"All {len(state.failed_tasks)} tasks failed. "
            "Please try again or rephrase your request."
        )
    results_context = state.context_for_supervisor()
    failed_note = (
        f"\n\nNote: {len(state.failed_tasks)} task(s) failed and were skipped: "
        f"{', '.join(state.failed_tasks)}"
        if state.failed_tasks else ""
    )
    llm = get_llm(skill=Skill.REASONING)
    response = llm.invoke([
        SystemMessage(content=AGGREGATE_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Original user request: {user_input}\n\n"
            f"Results from specialist agents:\n{results_context}"
            f"{failed_note}\n\n"
            "Please synthesize these results into a final answer."
        )),
    ])
    return response.content


async def run_multiagent(
        user_input: str,
        job_id: str,
        user_id: str = "default",
        max_seconds: int = 600,
) -> str:
    run_id = job_id if job_id else make_run_id(user_id=user_id)
    start_time = datetime.now()
    db_path = str(Path(os.getenv("AGENT_CHECKPOINT_DIR", "./checkpoints")) / "agent_state.db")

    logger.debug(f"[supervisor] Run started: {run_id}")
    logger.debug(f"[supervisor] Input: {user_input[:1000]}")

    async with get_checkpointer_cm(db_path) as checkpointer:
        # Check resume vs new run
        try:
            sample_thread = f"{run_id}_AgentType.RESEARCH_task_1"
            existing = await checkpointer.aget(
                {"configurable": {"thread_id": sample_thread}}
            )
            if existing:
                logger.info(f"[supervisor] ▶ RESUMING job: {run_id}")
            else:
                logger.info(f"[supervisor] ▶ NEW run: {run_id}")
        except Exception:
            logger.info(f"[supervisor] ▶ NEW run: {run_id}")


        start_time = datetime.now()

        logger.debug(f"[supervisor] Run started: {run_id}")
        logger.debug(f"[supervisor] Input: {user_input[:1000]}")

        client = MultiServerMCPClient(get_mcp_config())
        mcp_tools = await client.get_tools()
        memory = LongTermMemory(user_id=user_id)
        memory_context = memory.format_for_llm(user_input)

        plan = decompose_task(user_input, memory_context)
        state = SupervisorState(run_id=run_id, user_input=user_input, plan=plan)

        remaining = list(plan.tasks)
        max_round = 10
        rounds = 0

        while remaining and rounds < max_round:
            rounds += 1
            ready = [
                t for t in remaining
                if all(dep in state.completed_task_ids for dep in t.depends_on)
            ]

            if not ready:
                logger.error(
                    "[supervisor] No tasks ready to run — "
                    "possible circular dependency. Stopping."
                )
                break

            batch_results = await dispatch_tasks(
                tasks=ready,
                mcp_tools=mcp_tools,
                user_id=user_id,
                run_id=run_id,
                completed_results=list(state.results),
                checkpointer=checkpointer,
            )

            usable, failed = handle_failures(batch_results, plan)
            for r in batch_results:
                state.add_result(r)

            completed_ids = {r.task_id for r in batch_results}
            remaining = [t for t in remaining if t.task_id not in completed_ids]
            logger.debug(
                f"[supervisor] Round {rounds}: "
                f"{len(usable)} succeeded, {len(failed)} failed, "
                f"{len(remaining)} remaining"
            )
            critical_failed = [
                r for r in failed
                if plan.get_task(r.task_id) and
                   plan.get_task(r.task_id).priority >= 3
            ]
            if critical_failed and not usable:
                logger.error("[supervisor] Critical tasks failed — stopping early")
                break

        final_answer = aggregate_results(user_input, state)
        duration_s = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"[supervisor] Run complete — {run_id} | "
            f"tasks={len(plan.tasks)} | "
            f"success_rate={state.success_rate:.0%} | "
            f"time={duration_s:.1f}s"
        )

        # Memory
        steps_as_dicts = [
            {"tool": step, "status": "success"}
            for r in state.successful_results
            for step in r.steps_taken
        ]
        plans = [
            {
                "step": i + 1,
                "tool": t.agent_type,
                "reason": t.instruction[:100],
            }
            for i, t in enumerate(plan.tasks)
        ]
        evaluation = evaluate_execution_quality(
            user_input=user_input,
            plan=plans,
            completed_steps=steps_as_dicts,
            final_answer=final_answer,
        )
        memory.store_episode(
            user_input=user_input,
            completed_steps=steps_as_dicts,
            final_answer=final_answer,
            evaluation=evaluation,
        )
        memory.update_profile({
            "last_task": user_input[:100],
            "agents_used": list({r.agent_type for r in state.results}),
            "task_quality": evaluation.get("quality"),
            "session_date": datetime.now().isoformat(),
        })

        return final_answer

def run_multiagent_sync(user_input: str,  job_id: str, user_id: str = "default",) -> str:
    """Sync wrapper for convenience."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, run_multiagent(user_input, job_id, user_id))
            return future.result()

    return asyncio.run(run_multiagent(user_input, job_id, user_id))


if __name__ == "__main__":
    response = run_multiagent_sync("""
    Research the top 3 AI companies by revenue in 2025,
    then create an Excel file called ai_companies.xlsx
    and write the research results to it.
    """, "5")
    print(response)