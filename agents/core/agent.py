from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode

from agents.core.llm import get_llm
from agents.core.skill_enum import Skill
from langgraph.graph.message import add_messages

from agents.tools.calculator import calculator
from agents.tools.filesystem import read_file, write_file, list_files
from agents.tools.search import web_search


# Agent State
class AgentState(TypedDict):
    """
    Everything the agents remembers during one run.
    add_message is a LangGraph reducer - it appends instead of overwriting.
    """
    messages: Annotated[list, add_messages]


# System Prompt
SYSTEM_PROMPT = """
You are a helpful, autonomous agents.
When given a task:
1. Think step by step about what needs to be done.
2. Use available tools when needed.
3. Give a clear, concise final answer.
"""

TOOLS = [calculator, read_file, write_file, list_files, web_search]

# Agent node
def call_llm(state: AgentState) -> AgentState:
    llm = get_llm(skill=Skill.REASONING)
    llm_with_tools = llm.bind_tools(tools=TOOLS)
    message = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    llm_response = llm_with_tools.invoke(message)
    return {"messages": [llm_response]}

# routing logic
def should_use_tool(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("llm", call_llm)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_use_tool)
    graph.add_edge("tools","llm")

    return graph.compile()


def run_agent(user_input: str) -> str:
    agent = build_agent()
    result = agent.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
    return result["messages"][-1].content
if __name__ == '__main__':
    response = run_agent('Search the web for: what is MCP model context protocol')
    print(response)