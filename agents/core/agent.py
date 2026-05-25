import asyncio
from datetime import datetime
from typing import Annotated

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
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
from agents.memory.evaluator import evaluate_execution_quality
from agents.memory.long_term import LongTermMemory
from agents.tools.calculator import calculator
from agents.tools.mcp_client import get_mcp_config, get_sandbox_dir
from agents.tools.search import web_search

logger = get_logger("agent")
SANDBOX_DIR = get_sandbox_dir()

# System Prompt
SYSTEM_PROMPT = """
You are an autonomous agent that uses tools.

RULES:
- Use tools when necessary
- Only ONE tool call per step
- Wait for tool result before continuing

FILESYSTEM RULES:
Sandbox root: agent_workspace
All file paths must be relative and inside agent_workspace/.


IMPORTANT:
Tool execution is handled externally.
You only decide WHICH tool to call and WITH WHAT arguments.
"""


def _merge_steps(old: list[StepRecord], new: list[StepRecord]) -> list[StepRecord]:
    return old + new


def _merge_budgets(old: list[BudgetState], new: list[BudgetState]) -> list[BudgetState]:
    # only keep the newest one
    return new if new else old


# Agent State
class AgentState(TypedDict):
    """
    Everything the agents remembers during one run.
    add_message is a LangGraph reducer - it appends instead of overwriting.
    """
    messages: Annotated[list, add_messages]
    completed_steps: Annotated[list[StepRecord], _merge_steps]
    budget: Annotated[list[BudgetState], _merge_budgets]
    retry_counts: dict[str, int]


def clean_message_history(messages: list) -> list:
    """
    An AIMessage with tool_calls but no matching ToolMessage creates a
    malformed conversation that some LLMs reject. We strip both sides.

    Repair the potentially damaged tool calling conversation history
    to a single tool conversation that LLM can safely read

    Fix inconsistency between execution log and message history
    """

    # collect all tools id that have been executed
    # no matter successful or failed
    # each tool execution will generate a Toolmessage
    present_tool_call_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            present_tool_call_ids.add(msg.tool_call_id)

    cleaned = []
    for msg in messages:
        # the tools AI planed to call
        if isinstance(msg, AIMessage) and msg.tool_calls:
            matched = [tc for tc in msg.tool_calls if tc["id"] in present_tool_call_ids]
            if not matched:
                continue
            # when some tool calls dont have corresponding tool message
            if len(matched) != len(msg.tool_calls):
                msg = msg.model_copy(update={"tool_calls": matched[:1], "additional_kwargs": {}})

        if isinstance(msg, ToolMessage):
            if msg.tool_call_id not in present_tool_call_ids:
                continue
        cleaned.append(msg)
    return cleaned


def build_agent(tools:list, run_log: RunLogger):
    def call_llm(state: AgentState) -> dict:
        """
        single ReAct seasoning step.
        The LLM sees the full message history and decides what to do next.
        No planner, No executor, just think then act.
        """
        budget: BudgetState = state["budget"][-1]
        if budget.is_exhausted:
            reason = (
                f"Maximum steps reached {budget.max_steps} "
                if budget.is_over_steps
                else f"Maximum steps reached {budget.max_steps}"
            )
            logger.warning(f"[budget] exhausted {budget.summary()}]")
            llm = get_llm(skill=Skill.REASONING)
            llm_response = llm.invoke(
                [SystemMessage(content=(
                    f"You have reached the budget limit: {reason}. "
                    "Summarise what was accomplished so far and explain what remains"
                ))]
                + state["messages"]
            )
            record_usage(run_log, llm_response)
            return {"messages": [llm_response]}
        else:
            llm = get_llm(skill=Skill.REASONING)
            llm_with_tools = llm.bind_tools(tools)
            cleaned = clean_message_history(state["messages"])
            response_with_tool = llm_with_tools.invoke(
                [SystemMessage(content=SYSTEM_PROMPT)] + cleaned
            )
            logger.debug(f"[llm] tool_calls: {[t['name'] for t in getattr(response_with_tool, 'tool_calls', [])]}")
            record_usage(run_log, response_with_tool)
        return {"messages": [response_with_tool]}

    def run_verifier(state: AgentState) -> dict:
        """
        Run after tool execution.
        Inspect the latest StepRecord and decide continue or retry or skip or escalate.
        Inject a hint message into history on retry so the LLM corrects itself.
        """
        completed = state["completed_steps"]
        if not completed:
            return {}
        latest_step = completed[-1]
        retry_counts = dict(state.get("retry_counts", {}))
        retry_count = retry_counts.get(latest_step.tool, 0)

        decision = verify_step(latest_step, retry_count)
        logger.debug(f"[verifier] {latest_step.tool} → {decision.action}: {decision.reason}")
        run_log.record_step(latest_step, decision)

        if decision.action == "retry":
            retry_counts[latest_step.tool] = retry_count + 1
            hint_msg = SystemMessage(content=(
                f"The previous call to '{latest_step.tool}' failed. "
                f"Hint: {decision.hint} "
                f"Please try again with corrected arguments."
            ))
            run_log.increment_retry(latest_step.tool)
            return {
                "messages": [hint_msg],
                "retry_counts": retry_counts,
            }
        if decision.action == "escalate":
            stop_msg = SystemMessage(content=(
                f"Escalation: '{latest_step.tool}' failed critically after retries. "
                f"Reason: {decision.reason}."
                f"Stop execution and explain the failure to the user."
            ))
            return {
                "messages": [stop_msg],
            }
        if decision.action == "skip":
            skip_msg = SystemMessage(content=(
                f"'{latest_step.tool}' failed and was skipped. '"
                f"Reason: {decision.reason}."
                "Continue with the remaining tasks if any."
            ))
            return {
                "messages": [skip_msg],
            }
        # nothing to inject, continue
        return {}
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("verifier",run_verifier)
    graph.add_node("tools", make_tool_node(tools))
    graph.add_node("budget", increase_budget)

    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_use_tool)
    # tools -> budget -> verifier -> llm
    # after executing a tool need to update budget
    # then use verifier to check if over budget or other problems
    # then back to llm making decisions
    graph.add_edge("tools", "budget")
    graph.add_edge("budget", "verifier")
    graph.add_edge("verifier", "llm")
    return graph.compile()




def increase_budget(state: AgentState) -> dict:
    """increase steps_used after each tool execution."""
    budget = state["budget"][-1].increment()
    logger.debug(f"[budget] {budget.summary()}")
    return {"budget": [budget]}

# routing logic
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


async def run_agent_async(user_input: str, user_id: str = 'default', max_steps:int= 20, max_seconds: int = 300,) -> str:
    """
    Main async entry point.
    MCP client stays alive for the full duration of the agent run.
    """
    client = MultiServerMCPClient(get_mcp_config())
    mcp_tools = await client.get_tools()
    all_tools = [calculator, web_search] + mcp_tools

    memory = LongTermMemory(user_id=user_id)
    memory_context = memory.format_for_llm(user_input)

    initial_messages = [HumanMessage(content=user_input)]
    if memory_context:
        logger.debug(f"Memory context: {memory_context}")
        initial_messages.insert(0, HumanMessage(content = f"Relevant context from past sessions:\n{memory_context}"))

    budget = BudgetState(max_steps=max_steps, max_seconds=max_seconds)
    run_log = RunLogger(user_id=user_id, user_input=user_input)

    agent = build_agent(all_tools,run_log)

    result = await agent.ainvoke({
        "messages": initial_messages,
        "completed_steps": [],
        "budget": [budget],
        "retry_counts": {},
    })

    final_answer = result["messages"][-1].content
    completed_steps = result.get("completed_steps", [])
    final_budget: BudgetState = result["budget"][-1]
    logger.debug(f"Run complete - {final_budget.summary()}")
    steps_as_dicts = [s.model_dump() for s in completed_steps]

    evaluation = evaluate_execution_quality(
        user_input=user_input,
        plan=[],
        completed_steps=steps_as_dicts,
        final_answer=final_answer,
    )
    log_path = run_log.finish(
        final_answer=final_answer,
        budget=final_budget,
        evaluation=evaluation,
    )
    logger.debug(f"Run log: {log_path}")

    memory.store_episode(
        user_input=user_input,
        completed_steps=steps_as_dicts,
        final_answer=final_answer,
        evaluation=evaluation,
    )

    memory.update_profile({
        "last_task": user_input[:100],
        "tools_used": list({s.tool for s in completed_steps}),
        "task_quality": evaluation.get("quality"),
        "steps_used": final_budget.steps_used,
        "session_date": datetime.now().isoformat()
    })

    return final_answer

def record_usage(run_log, msg):

    usage = getattr(msg, "usage_metadata", None)
    input_tokens = usage.get("input_tokens", 0) if usage else 0
    output_tokens = usage.get("output_tokens", 0) if usage else 0
    run_log.record_llm_call(
        input_tokens=input_tokens,
        output_tokens= output_tokens,
    )
    logger.debug(
            f"[llm] tool_calls={[t['name'] for t in getattr(msg, 'tool_calls', [])]} | "
            f"tokens={input_tokens}in/"
            f"{output_tokens}out"
        )

# ── 5. Sync wrapper ────────────────────────────────────────────────────────────
def run_agent(user_input: str, user_id: str = "default") -> str:
    """safe sync wrapper that works inside running event loops
    (FastAPI, Jupyter) as well as plain scripts."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop (FastAPI, Jupyter) — schedule as a task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, run_agent_async(user_input, user_id))
            return future.result()
    else:
        return asyncio.run(run_agent_async(user_input, user_id))


if __name__ == "__main__":
    # response = run_agent('Write a file called mcp_test.txt with content: MCP is working! Then read the content back.')
    # response = run_agent('Search online "what is a MCP",  and then write the result to MCP.txt file')
    # print(response)
    response = run_agent('''
    Create an Excel file called sales.xlsx with this data:
    | product  | revenue | units |
    | Widget A | 5000    | 100   |
    | Widget B | 3000    | 60    |
    | Widget C | 8000    | 160   |
    | Widget D | 1500    | 30    |
    Then calculate the average revenue, find the max revenue, and filter products where revenue > 4000.
    ''')
    # response = run_agent("Search for 'what is asyncio' and calculate 999 * 888 at the same time")

    print(response)
