import asyncio
import json

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from agents.core.arg_validator import validate_and_fix_args
from agents.core.logger import get_logger
from agents.core.models import ToolResult, StepRecord

logger = get_logger("tool_node")


def make_tool_node(tools: list[BaseTool]):
    """
    supports both single and parallel tool execution.

    When the LLM requests multiple independent tool calls in one response,
    they run concurrently with asyncio.gather() instead of sequentially.
    Each call still produces its own StepRecord and ToolMessage.

    Dependency chains should instead emerge through iterative ReAct loops:
    the LLM observes one tool result before deciding the next tool call.
    """
    tool_map = {tool.name: tool for tool in tools}

    async def tool_node(state) -> dict:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        if not tool_calls:
            return {"messages": [], "completed_steps": []}
        if len(tool_calls) == 1:
            result, step = await _execute_one(tool_calls[0], tool_map)
            return _build_return([(result, step, tool_calls[0]["id"])])
        logger.debug(f"Parallel execution: {[tc['name'] for tc in tool_calls]}")
        tasks = [_execute_one(tc, tool_map) for tc in tool_calls]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        pairs = []
        for i, outcome in enumerate(outcomes):
            tc = tool_calls[i]
            if isinstance(outcome, Exception):
                result = ToolResult(status="error", error=str(outcome))
                step = StepRecord(tool=tc['name'], args=tc["args"], status='error', error=str(outcome))
            else:
                result, step = outcome
            pairs.append((result, step, tc["id"]))
        return _build_return(pairs)

    return tool_node

async def _execute_one(tool_call: dict, tool_map:dict[str, BaseTool]) -> tuple[ToolResult, StepRecord]:
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    logger.debug(f"-> Action: {tool_name} | args: {tool_args}")
    tool = tool_map.get(tool_name)
    if not tool:
        result = ToolResult(status="error", error=f"Tool '{tool_name}' not found.")
        step = StepRecord(tool=tool_name, args=tool_args, status="error", error=result.error)
        return result, step
    fixed_args, warnings = validate_and_fix_args(tool, tool_args)
    if warnings:
        logger.debug(f"Arg validation warnings for {tool_name}: {warnings}")

    missing  = [w for w in warnings if w.startswith("Missing required")]
    if missing:
        result = ToolResult(status="error", error=f"Missing required arguments: {missing}")
        step = StepRecord(tool=tool_name, args=tool_args, status="error", error=result.error)
        return result, step

    try:
        raw_output = await tool.ainvoke(fixed_args)
        result = ToolResult.from_raw(raw_output)
        step = StepRecord(
            tool=tool_name,
            args=fixed_args,
            status=result.status,
            output=result.output,
            error = result.error,
        )
    except Exception as e:
        result = ToolResult(status="error", error=str(e))
        step = StepRecord(tool=tool_name, args=tool_args, status="error", error=str(e))
    logger.debug(f"<- {tool_name}: status={result.status} | {str(result.output)[:100]}")
    return result, step



def _build_return(pairs: list[tuple[ToolResult, StepRecord, str]]) -> dict:
    """
    packages results into the LangGraph state update format.
    """
    messages = []
    steps = []
    for result, step, tool_id in pairs:
        messages.append(ToolMessage(
            content=json.dumps({
                "status": result.status,
                "output": result.output,
                "error": result.error,
            }),
            tool_call_id=tool_id
        ))
        steps.append(step)
    return {
        "messages": messages,
        "completed_steps": steps,
    }
