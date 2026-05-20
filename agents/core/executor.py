from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from agents.core.llm import get_llm
from agents.core.logger import get_logger
from agents.core.skill_enum import Skill

"""
!!!
This executor only works on decision layer and affects tool selection through
retry and prompt correction,
but cannot roll back side effects that have already occurred,
so the system is a forward only execution model.

"""


logger = get_logger("executor")

EXECUTOR_PROMPT = """You are a tool executor. You will be told exactly which tool to call next.
Your ONLY job is to call that tool with the correct arguments.
Do NOT call any other tool.
Do NOT explain anything.
Do NOT give a final answer yet unless told to.
Just call the specified tool with the right arguments based on the user's request and previous results.
"""
INSTRUCTION_PROMPT = """
{EXECUTOR_PROMPT}

YOU MUST CALL THIS TOOL NOW: '{next_tool}'
Reason: {next_reason}

Look at the conversation history and previous results to determine the correct arguments.
"""


def execute_next_step(next_tool: str, next_reason: str, tools: list[BaseTool], messages: list) -> AIMessage:
    """
        Executor role — given the next tool to call, produces exactly one tool call.
        No free reasoning — just executes what the planner decided.
    """

    llm = get_llm(skill=Skill.REASONING)
    llm_with_tools = llm.bind_tools(tools)

    instruction = INSTRUCTION_PROMPT.format(EXECUTOR_PROMPT=EXECUTOR_PROMPT, next_tool=next_tool,
                                            next_reason=next_reason)

    response = llm_with_tools.invoke([instruction] + messages)
    response = _enforce_single_tool(response)

    if not _called_correct_tool(response, next_tool):
        actual = response.tool_calls[0]["name"] if response.tool_calls else "none"
        logger.warning(f"Executor called wrong tool on first attempt: '{actual}' instead of '{next_tool}'. Retrying...")
        strict_instruction = (
            f"{instruction}\n\nCRITICAL: You MUST call '{next_tool}'. "
            f"You just called '{actual}' which is wrong. Call '{next_tool}' now."
        )
        response = llm_with_tools.invoke([strict_instruction] + messages)
        response = _enforce_single_tool(response)
        if not _called_correct_tool(response, next_tool):
            actual2 = response.tool_calls[0]["name"] if response.tool_calls else "none"
            logger.error(
                f"Executor called wrong tool twice: expected '{next_tool}', got '{actual2}'. "
                "Skipping step to avoid corrupting state."
            )
            # Return an AIMessage with no tool calls so the graph moves to final answer
            return AIMessage(content=f"[Executor failed to call '{next_tool}' after 2 attempts. Skipping.]")

    return response


def _called_correct_tool(response: AIMessage, expected: str) -> bool:
    tool_calls = getattr(response, "tool_calls", [])
    return bool(tool_calls) and tool_calls[0]["name"] == expected


def _enforce_single_tool(response) -> AIMessage:
    if hasattr(response, "tool_calls") and len(response.tool_calls) > 1:
        response = response.model_copy(update={"tool_calls": [response.tool_calls[0]], "additional_kwargs": {}})
    return response
