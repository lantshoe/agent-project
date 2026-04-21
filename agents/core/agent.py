from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode

from agents.core.llm import get_llm
from agents.core.logger import get_logger
from agents.core.skill_enum import Skill
from langgraph.graph.message import add_messages
import asyncio

from agents.core.tool_node import make_tool_node
from agents.tools.calculator import calculator
from agents.tools.mcp_client import get_mcp_config, get_sandbox_dir
from agents.tools.search import web_search
from langchain_mcp_adapters.client import MultiServerMCPClient


logger = get_logger("agent")

# Agent State
class AgentState(TypedDict):
    """
    Everything the agents remembers during one run.
    add_message is a LangGraph reducer - it appends instead of overwriting.
    """
    messages: Annotated[list, add_messages]

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
All filesystem operations MUST use paths inside the sandbox directory. 
Sandbox root: agent_workspace

IMPORTANT:
Tool execution is handled externally.
You only decide WHICH tool to call and WITH WHAT arguments.
"""


# routing logic
def should_use_tool(state: AgentState) -> str:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    print(f"last_message: {last_message}")
    print("tool_calls: ", tool_calls)
    if tool_calls:
        logger.debug(f"Thought → Action: {[t['name'] for t in tool_calls]}")
        return "tools"
    logger.debug("Thought → Final Answer")
    return END

def build_agent(tools):
    def call_llm(state: AgentState) -> AgentState:
        llm = get_llm(skill=Skill.REASONING)
        llm_with_tools = llm.bind_tools(tools=tools)
        message = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        llm_response = llm_with_tools.invoke(message)
        return {"messages": [llm_response]}


    graph = StateGraph(AgentState)

    graph.add_node("llm", call_llm)
    graph.add_node("tools", make_tool_node(tools))

    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_use_tool)
    graph.add_edge("tools","llm")

    return graph.compile()

async def run_agent_async(user_input: str) -> str:
    """
    Main async entry point.
    MCP client stays alive for the full duration of the agent run.
    """
    client = MultiServerMCPClient(get_mcp_config())
    mcp_tools = await client.get_tools()
    all_tools = [calculator, web_search] + mcp_tools
    agent = build_agent(all_tools)

    result = await agent.ainvoke({
        "messages": [HumanMessage(content=user_input)]
    })

    return result["messages"][-1].content

# ── 5. Sync wrapper ────────────────────────────────────────────────────────────
def run_agent(user_input: str) -> str:
    """Sync wrapper for convenience."""
    return asyncio.run(run_agent_async(user_input))


if __name__ == "__main__":
    # response = run_agent('Write a file called mcp_test.txt with content: MCP is working! Then read the content back.')
    response = run_agent('Search online "what is a MCP",  and then write the result to MCP.txt file')
    print(response)