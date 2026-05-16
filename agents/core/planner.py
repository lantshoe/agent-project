import json
from langchain_core.messages import HumanMessage, SystemMessage
from agents.core.llm import get_llm
from agents.core.logger import get_logger
from agents.core.skill_enum import Skill

logger = get_logger("planner")

PLANNER_PROMPT = """You are a task planner. Given a user request and a list of available tools,
produce a step-by-step plan as a JSON array.

Each step must have:
- "step": step number
- "tool": exact tool name to call
- "reason": why this step is needed
- "depends_on": list of step numbers that must complete first ([] if none)

RULES:
- Only use tools from the provided list
- Order steps so dependencies are respected
- Be specific about what each tool needs to do

Respond ONLY with a valid JSON array, no other text.
"""
USER_PROMPT = """
{memory_section}

TOOL LIST (choose EXACT match only):
{tool_descriptions}

IMPORTANT: You must only use tool names from the list above. 
Do not invent or modify tool names.

User request: {user_input}
Produce a JSON plan array.
"""


MEMORY_PROMPT = """
Relevant memory from past sessions:
{memory_context}

Use this to avoid past mistakes and reuse good plans.
"""

RESPONSE_PROMPT = """
Your previous plan was invalid.

Original user request:
{user_input}

Allowed tools:
{available_tools}

Invalid tools used:
{invalid_tools}

You MUST ONLY use exact tool names from the allowed list.

Regenerate the ENTIRE plan as valid JSON.

Respond ONLY JSON.
"""
def create_plan(user_input: str, available_tools: list, memory_context = "") -> list[dict]:
    """
    Planner role — takes user input and produces a structured execution plan.
    Called ONCE before the executor starts.
    """
    max_retry = 3
    plans = generate_plan(user_input, available_tools,memory_context)
    invalid_tools = get_invalidate_tools(available_tools, plans)
    for _ in range(max_retry):
        if len(invalid_tools) > 0:
            logger.warning(f"Invalidating tools: {invalid_tools}")
            plans = retry_generate_plan(user_input, available_tools, invalid_tools)
            invalid_tools = get_invalidate_tools(available_tools,plans)
    logger.debug(f"Plan: {plans}")
    return plans


def generate_plan(user_input: str, available_tools: list, memory_context: str) -> list[dict]:
    tool_descriptions = "\n".join([
        f"-{t.name}: {t.description[:150]}"
        for t in available_tools
    ])

    llm = get_llm(skill=Skill.REASONING)
    memory_section = MEMORY_PROMPT.format(memory_context = memory_context) if memory_context else ""
    response = llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=USER_PROMPT.format(tool_descriptions=tool_descriptions, user_input=user_input, memory_section=memory_section)),
    ])
    return retrieve_plan(response)


def retrieve_plan(response) -> list[dict]:
    try:
        content = response.content.strip()
        # clean markdown format
        # sometime LLM return markdown format info
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        plan = json.loads(content.strip())
        logger.debug(f"Plan created: {json.dumps(plan, indent=2)}")
        return plan
    except Exception as e:
        logger.error(f"Planner failed to parse response: {e}")
        return []



def get_invalidate_tools(available_tools: list, plan_tools: list[dict]) -> list:
    allowed_tools = {t.name for t in available_tools}
    invalid_tools = []
    for step in plan_tools:
        if step["tool"] not in allowed_tools:
            invalid_tools.append(step["tool"])
    return invalid_tools

def retry_generate_plan(user_input: str, available_tools:list, invalid_tools:list) -> list[dict]:
    llm = get_llm(skill=Skill.REASONING)
    tool_descriptions = "\n".join([
        f"-{t.name}: {t.description[:150]}"
        for t in available_tools
    ])
    retry_prompt = RESPONSE_PROMPT.format(user_input=user_input, invalid_tools=invalid_tools, available_tools=tool_descriptions)
    response = llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=retry_prompt)
    ])
    return retrieve_plan(response)