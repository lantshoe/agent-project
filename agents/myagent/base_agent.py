"""
shared ReAct loop used by every subagent in Phase 3

Extracted from agent.py,
"""


from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
import traceback

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agents.core.llm import get_llm
from agents.core.logger import get_logger
from agents.core.models import StepRecord, BudgetState
from agents.core.run_logger import RunLogger
from agents.core.skill_enum import Skill
from agents.core.tool_node import make_tool_node
from agents.core.verifier import verify_step
from agents.models.supervisor import AgentType, Task, SubagentResult

logger = get_logger("base_agent")

# Tools that create or write to files — used to build file_state
# so downstream tasks know what files/sheets already exist.
FILE_STATE_TOOLS = {"create_workbook", "write_sheet"}

def _merge_steps(old: list[StepRecord], new: list[StepRecord]) -> list[StepRecord]:
    return old + new

def _merge_budgets(old: list[BudgetState], new: list[BudgetState]) -> list[BudgetState]:
    return new if new else old

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    completed_steps: Annotated[list[StepRecord], _merge_steps]
    budget: Annotated[list[BudgetState], _merge_budgets]
    retry_counts: dict[str, int]


def _extract_file_state(completed_steps: list[StepRecord]) -> dict[str, list[str]]:
    """
    Scans completed steps for create_workbook/write_sheet calls and builds
    a map of {filepath: [sheet_names]} reflecting what currently exists.

    This lets downstream tasks know which files/sheets already exist so
    they use write_sheet (not create_workbook) to add new sheets, and
    don't waste steps re-discovering structure via list_sheets.
    """
    files: dict[str, set[str]] = {}
    for step in completed_steps:
        if step.tool in FILE_STATE_TOOLS and step.succeeded:
            filepath = step.args.get("filepath")
            sheet    = step.args.get("sheet_name")
            if not filepath:
                continue
            files.setdefault(filepath, set())
            if sheet:
                files[filepath].add(sheet)
    return {fp: sorted(sheets) for fp, sheets in files.items()}


class BaseAgent:
    """
    the shared ReAct loop every subagent built on

    subagent instantiate with their own tools and system prompt
    """
    def __init__(
            self,
            tools: list[BaseTool],
            system_prompt: str,
            agent_type: AgentType,
            max_steps: int = 15,
            max_seconds: int = 600,
    ):
        self.tools = tools
        self.system_prompt = system_prompt
        self.agent_type = agent_type
        self.max_steps = max_steps
        self.max_seconds = max_seconds

    async def run(self, task: Task, user_id: str = "default") -> SubagentResult:
        """
        run the ReAct loop for a single task,
        return a typed subagent result to supervisor can use directly.
        """
        start_at = datetime.now()
        run_log = RunLogger(
            user_id = f"{user_id}_{self.agent_type}_{task.task_id}",
            user_input=task.instruction
        )
        content = task.instruction
        if task.context:
            context_str = "\n".join(f" {k}: {v}" for k, v in task.context.items())
            content = f"{task.instruction} \n\n Context from prior tasks:\n{context_str}"

        initial_messages = [HumanMessage(content=content)]

        budget = BudgetState(
            max_steps = self.max_steps,
            max_seconds = self.max_seconds,
        )
        graph = self._build_graph(run_log)

        try:
            result = await graph.ainvoke({
                "messages": initial_messages,
                "completed_steps": [],
                "budget": [budget],
                "retry_counts": {},
            })
            final_answer: str = _extract_final_answer(result["messages"])

            completed_steps: list[StepRecord] = result.get("completed_steps", [])
            final_budget: BudgetState = result['budget'][-1]

            duration_s = (datetime.now() - start_at).total_seconds()

            if final_budget.is_exhausted:
                status = "partial"
            elif not completed_steps:
                status = "partial"
            elif any(not s.succeeded for s in completed_steps):
                status = "partial"
            else:
                status = "success"

            run_log.finish(
                final_answer,
                budget=final_budget,
                evaluation={"quality":status},
            )
            logger.debug(
                f"[{self.agent_type}:{task.task_id}] done — "
                f"status={status} | steps={final_budget.steps_used} | "
                f"time={duration_s:.1f}s"
            )

            file_state = _extract_file_state(completed_steps)
            if file_state:
                logger.debug(f"[{self.agent_type}:{task.task_id}] file_state: {file_state}")

            return SubagentResult(
                task_id=task.task_id,
                agent_type=self.agent_type,
                status=status,
                answer=final_answer,
                steps_taken=[s.tool for s in completed_steps],
                duration_s=duration_s,
                metadata={"file_state": file_state} if file_state else {},
            )
        except Exception as e:
            duration_s = (datetime.now() - start_at).total_seconds()
            logger.error(f"[{self.agent_type}:{task.task_id}] crashed: {e} \n"
                         f"{traceback.format_exc()}")
            return SubagentResult(
                task_id=task.task_id,
                agent_type=self.agent_type,
                status="failed",
                answer="",
                duration_s=duration_s,
                error=str(e),
            )


    def _build_graph(self, run_log: RunLogger):
        tools = self.tools
        system_prompt = self.system_prompt
        def call_llm(state: AgentState) -> dict:
            budget: BudgetState = state["budget"][-1]

            if budget.is_exhausted:
                reason = (
                    f"Maximum steps reached ({budget.max_steps})"
                    if budget.is_over_steps
                    else f"Time limit reached ({budget.max_seconds:.0f}s)"
                )
                logger.warning(f"[budget] Exhausted: {budget.summary()}")
                llm = get_llm(skill=Skill.SIMPLE)
                response = llm.invoke(
                    [SystemMessage(content=(
                        f"Budget limit reached: {reason}. "
                        "Summarise what was accomplished and what remains."
                    ))]
                    + state["messages"]
                )
                _record_tokens(response, run_log)
                return {"messages": [response]}

            llm = get_llm(skill=Skill.REASONING)
            llm_with_tools = llm.bind_tools(tools)
            response = llm_with_tools.invoke(
                [SystemMessage(content=system_prompt)] + state["messages"]
            )

            # Structural guard: create_workbook must never run alongside
            # other tools — it must complete first so subsequent writes
            # target an existing file. The LLM repeatedly batches these
            # together despite prompt instructions, so we enforce it here.
            if hasattr(response, "tool_calls") and len(response.tool_calls) > 1:
                tool_names = [tc["name"] for tc in response.tool_calls]
                if "create_workbook" in tool_names:
                    logger.debug(
                        f"[{self.agent_type}] create_workbook batched with "
                        f"{tool_names} — isolating to create_workbook only"
                    )
                    create_call = next(
                        tc for tc in response.tool_calls if tc["name"] == "create_workbook"
                    )
                    response = response.model_copy(update={"tool_calls": [create_call]})

            _record_tokens(response, run_log)
            logger.debug(
                f"[llm] tool_calls={[t['name'] for t in getattr(response, 'tool_calls', [])]}"
            )
            return {"messages": [response]}

        def run_verifier(state: AgentState) -> dict:
            completed = state["completed_steps"]
            if not completed:
                return {}

            latest_step = completed[-1]
            retry_counts = dict(state.get("retry_counts", {}))
            retry_count = retry_counts.get(latest_step.tool, 0)
            decision = verify_step(latest_step, retry_count)
            logger.debug(f"[verifier] {latest_step.tool} → {decision.action}")
            run_log.record_step(latest_step, decision)

            if decision.action == "retry":
                retry_counts[latest_step.tool] = retry_count + 1
                run_log.increment_retry(latest_step.tool)
                return {
                    "messages": [HumanMessage(content=(
                        f"The previous call to '{latest_step.tool}' failed. "
                        f"Hint: {decision.hint} Try again with corrected arguments."
                    ))],
                    "retry_counts": retry_counts,
                }

            if decision.action == "escalate":
                return {"messages": [HumanMessage(content=(
                    f"ESCALATION: '{latest_step.tool}' failed critically. "
                    f"Reason: {decision.reason}. Stop and explain the failure."
                ))]}

            if decision.action == "skip":
                return {"messages": [HumanMessage(content=(
                    f"'{latest_step.tool}' failed and was skipped. "
                    f"Reason: {decision.reason}. Continue with remaining tasks."
                ))]}

            return {}

        def increment_budget(state: AgentState) -> dict:
            budget: BudgetState = state["budget"][-1].increment()
            logger.debug(f"[budget] {budget.summary()}")
            return {"budget": [budget]}

        def should_use_tool(state: AgentState) -> str:
            if state['budget'][-1].is_exhausted:
                return END

            last_message = state["messages"][-1]
            tool_calls = getattr(last_message, "tool_calls", [])
            if tool_calls:
                logger.debug(f"Thought → Action: {[t['name'] for t in tool_calls]}")
                return "tools"
            logger.debug("Thought → Final Answer")
            return END

        graph = StateGraph(AgentState)
        graph.add_node("llm", call_llm)
        graph.add_node("tools", make_tool_node(tools))
        graph.add_node("verifier", run_verifier)
        graph.add_node("budget", increment_budget)
        graph.set_entry_point("llm")
        graph.add_conditional_edges("llm",should_use_tool)
        graph.add_edge("tools","budget")
        graph.add_edge("budget","verifier")
        graph.add_edge("verifier","llm")
        return graph.compile()

def _record_tokens(response, run_log):
    usage = getattr(response, "usage_metadata", None)
    run_log.record_llm_call(
        input_tokens = usage.get("input_tokens", 0) if usage else 0,
        output_tokens=usage.get("output_tokens", 0) if usage else 0
    )


def _extract_final_answer(messages: list) -> str:
    """
    Finds the last non-empty AIMessage content.
    Small models sometimes return empty content after tool results —
    in that case fall back to summarizing the last tool output.
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content.strip():
            return msg.content.strip()

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                import json
                data = json.loads(msg.content)
                output = data.get("output", "")
                if output:
                    return str(output)
            except Exception:
                if msg.content.strip():
                    return msg.content.strip()

    return "Task completed but no summary was generated."