import json
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from agents.core.llm import get_llm
from agents.core.logger import get_logger
from agents.core.skill_enum import Skill

logger = get_logger("executor")

EXECUTOR_PROMPT = """You are a tool executor. You will be told exactly which tool to call next.
Your ONLY job is to call that tool with the correct arguments.
Do NOT call any other tool.
Do NOT explain anything.
Do NOT give a final answer yet unless told to.
Just call the specified tool with the right arguments based on the user's request and previous results.
"""
INSTRUCTION_PROMPT ="""
{EXECUTOR_PROMPT}

YOU MUST CALL THIS TOOL NOW: '{next_tool}'
Reason: {next_reason}

Look at the conversation history and previous results to determine the correct arguments.
"""

def execute_next_step(next_tool:str, next_reason:str, tools:list[BaseTool], messages: list) -> AIMessage:
    """
        Executor role — given the next tool to call, produces exactly one tool call.
        No free reasoning — just executes what the planner decided.
    """

    llm = get_llm(skill = Skill.REASONING)
    llm_with_tools = llm.bind_tools(tools)

    instruction = INSTRUCTION_PROMPT.format(EXECUTOR_PROMPT = EXECUTOR_PROMPT, next_tool = next_tool, next_reason = next_reason)

    response = llm_with_tools.invoke([instruction] + messages)

    if hasattr(response, "tool_calls") and response.tool_calls:
        actual_tool = response.tool_calls[0]["name"]
        if actual_tool != next_tool:
            logger.warning(f"Executor called wrong tool: {actual_tool} instead of {next_tool}")
            response.tool_calls[0]["name"] = next_tool

        if len(response.tool_calls) > 1:
            response.tool_calls = [response.tool_calls[0]]

    return response


