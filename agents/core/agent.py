import asyncio
from typing import Annotated
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from agents.core.executor import execute_next_step
from agents.core.llm import get_llm
from agents.core.logger import get_logger
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

You operate in a tool-calling system.

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

# Agent State
class AgentState(TypedDict):
    """
    Everything the agents remembers during one run.
    add_message is a LangGraph reducer - it appends instead of overwriting.
    """
    messages: Annotated[list, add_messages]
    completed_steps: Annotated[list, lambda old, new: old + new]
    plan: list
    current_step: int

def clean_message_history(messages: list) -> list:
    """
    Removes stale tool_calls from AIMessages that had multiple tool calls planned.
    This prevents the LLM from re-following its old batch plan.
    Also removes orphaned ToolMessages whose tool_call_id no longer matches.
    """
    cleaned = []
    valid_tool_call_ids = set()

    for msg in messages:
        if hasattr(msg, "tool_calls"):
            if len(msg.tool_calls) > 1:
                # Strip extra tool calls — keep content but clear the batch plan
                msg = msg.model_copy(update={"tool_calls": [], "additional_kwargs": {}})
            else:
                for tc in msg.tool_calls:
                    valid_tool_call_ids.add(tc["id"])

        # Skip orphaned ToolMessages
        if hasattr(msg, "tool_call_id"):
            if msg.tool_call_id not in valid_tool_call_ids:
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
        plan = state.get("plan", [])
        completed = state.get("completed_steps", [])
        completed_tool_names = {s["tool"] for s in completed if s and s["status"] == "success"}
        logger.debug(f"completed tools: {completed}")
        if plan:
            remaining = [
                s for s in plan
                if s["tool"] not in completed_tool_names
            ]

            if not remaining:
                # All steps done , ask LLM for final answer only
                llm = get_llm(skill=Skill.REASONING)
                results_summary = "\n".join(completed_tool_names)
                response = llm.invoke([
                                          SystemMessage(
                                              content="Summarize the results of the completed tasks clearly."),
                                          SystemMessage(content=f"Results:\n{results_summary}"),
                                      ] + state["messages"][:1])
                return {"messages": [response]}

            next_step = remaining[0]
            next_reason = next_step["reason"]
            next_tool = next_step['tool']
            logger.debug(f"Executor → next step: {next_tool} ({next_reason})")
            clean_history = clean_message_history(state["messages"])
            response = execute_next_step(
                next_tool=next_tool,
                next_reason=next_reason,
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
        plan_text = "\n".join([
            f"  Step {s['step']}: {s['tool']} — {s['reason']}"
            for s in plan
        ])
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

    evaluation = evaluate_execution_quality(
        user_input=user_input,
        plan = plan,
        completed_steps=completed_steps,
        final_answer=final_answer,
    )

    memory.store_episode(
        user_input=user_input,
        completed_steps=completed_steps,
        final_answer=final_answer,
        evaluation=evaluation,
    )

    memory.update_profile({
        "last_task": user_input[:100],
        "tools_used": list({s.get("tool") for s in completed_steps}),
        "task_quality": evaluation.get("quality"),
        "session_date": datetime.now().isoformat()
    })

    return result["messages"][-1].content


# ── 5. Sync wrapper ────────────────────────────────────────────────────────────
def run_agent(user_input: str) -> str:
    """Sync wrapper for convenience."""
    return asyncio.run(run_agent_async(user_input))


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
