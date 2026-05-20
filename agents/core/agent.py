import asyncio
from typing import Annotated
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from agents.core.executor import execute_next_step
from agents.core.llm import get_llm
from agents.core.logger import get_logger
from agents.core.models import PlanStep, StepRecord
from agents.core.planner import create_plan
from agents.core.skill_enum import Skill
from agents.core.tool_node import make_tool_node
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
# Agent State
class AgentState(TypedDict):
    """
    Everything the agents remembers during one run.
    add_message is a LangGraph reducer - it appends instead of overwriting.
    """
    messages: Annotated[list, add_messages]
    completed_steps: Annotated[list[StepRecord], _merge_steps]
    plan: list[PlanStep]
    current_step: int

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
            if len(matched) < len(msg.tool_calls):
                msg = msg.model_copy(update={"tool_calls": matched[:1], "additional_kwargs": {}})
            elif len(matched) > 1:
                msg = msg.model_copy(update={"tool_calls": matched[:1], "additional_kwargs": {}})

        if isinstance(msg, ToolMessage):
            if msg.tool_call_id not in present_tool_call_ids:
                continue
        cleaned.append(msg)
    return cleaned


# routing logic
def should_use_tool(state: AgentState) -> str:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    logger.debug(f"last_message: {last_message}")
    logger.debug(f"tool_calls: {tool_calls}")
    if tool_calls:
        logger.debug(f"Thought → Action: {[t['name'] for t in tool_calls]}")
        return "tools"
    logger.debug("Thought → Final Answer")
    return END


def build_agent(tools):
    def call_llm(state: AgentState) -> AgentState:
        plan: list[PlanStep] = state.get("plan", [])
        completed: list[StepRecord] = state.get("completed_steps", [])
        completed_tool_names = {s.tool for s in completed if s.succeeded}
        logger.debug(f"completed tools: {completed}")
        if plan:
            remaining = [
                s for s in plan
                if s.tool not in completed_tool_names
            ]

            if not remaining:
                # All steps done , ask LLM for final answer only
                llm = get_llm(skill=Skill.REASONING)
                response = llm.invoke([SystemMessage(content="All tasks are complete. Summarize the results clearly.")]
                                    + state["messages"])
                return {"messages": [response]}

            next_step = remaining[0]
            logger.debug(f"Executor → next step: {next_step.tool} ({next_step.reason})")
            clean_history = clean_message_history(state["messages"])
            response = execute_next_step(
                next_tool=next_step.tool,
                next_reason=next_step.reason,
                tools=tools,
                messages=clean_history,
            )


        else:
            logger.debug("No plan — free ReAct mode")
            llm = get_llm(skill=Skill.REASONING)
            llm_with_tools = llm.bind_tools(tools)
            clean_history = clean_message_history(state["messages"])
            response = llm_with_tools.invoke(
                [SystemMessage(content=SYSTEM_PROMPT)] + clean_history
            )
            if hasattr(response, "tool_calls") and len(response.tool_calls) > 1:
                response.tool_calls = [response.tool_calls[0]]
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", make_tool_node(tools))
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_use_tool)
    graph.add_edge("tools", "llm")

    return graph.compile()


async def run_agent_async(user_input: str, user_id: str = 'default') -> str:
    """
    Main async entry point.
    MCP client stays alive for the full duration of the agent run.
    """
    client = MultiServerMCPClient(get_mcp_config())
    mcp_tools = await client.get_tools()
    all_tools = [calculator, web_search] + mcp_tools

    memory = LongTermMemory(user_id=user_id)
    memory_context = memory.format_for_llm(user_input)
    if memory_context:
        logger.debug(f"Memory context: {memory_context}")

    logger.debug("Planning...")
    plan = create_plan(user_input, all_tools, memory_context = memory_context)
    if plan:
        plan_text = "\n".join([f"  Step {s.step}: {s.tool} — {s.reason}" for s in plan])
        logger.debug(f"Plan: {plan_text}")
    else:
        logger.debug("No plan generated — executor will decide")

    agent = build_agent(all_tools)

    result = await agent.ainvoke({
        "messages": [HumanMessage(content=user_input)],
        "completed_steps": [],
        "plan": plan,
        "current_step": 0,
    })

    final_answer = result["messages"][-1].content
    completed_steps = result.get("completed_steps",[])

    steps_as_dicts = [s.model_dump() for s in completed_steps]
    plan_as_dicts = [s.model_dump() for s in plan] if plan else []

    evaluation = evaluate_execution_quality(
        user_input=user_input,
        plan = plan_as_dicts,
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
        "tools_used": list({s.tool for s in completed_steps}),
        "task_quality": evaluation.get("quality"),
        "session_date": datetime.now().isoformat()
    })

    return final_answer


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
    print(response)
