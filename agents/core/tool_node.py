import json
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from agents.core.logger import get_logger

logger = get_logger("tool_node")


def unwrap_mcp_result(raw_output) -> str:
    """
    MCP tools return results wrapped in a list of content blocks.
    This unwraps them to a clean string for the LLM.

    Input:  [{'type': 'text', 'text': '{"status": "success", ...}', 'id': '...'}]
    Output: {"status": "success", ...}
    """
    if isinstance(raw_output, list):
        texts = []
        for block in raw_output:
            if isinstance(block, dict) and "text" in block:
                texts.append(block["text"])
            elif hasattr(block, "text"):
                texts.append(block.text)
        return "\n".join(texts) if texts else str(raw_output)
    return str(raw_output)

def make_tool_node(tools: list[BaseTool]):
    """
    ReAct strict mode tool node:
    - Executes exactly ONE tool per cycle
    - Returns structured JSON result so LLM can reason about success/failure
    - Next action is decided AFTER seeing this result
    """
    tool_map = {tool.name: tool for tool in tools}
    async def tool_node(state) -> dict:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        if not tool_calls:
            return {"messages": []}

        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        logger.debug(f"-> Action: {tool_name} | args: {tool_args} | ID: {tool_id}")
        tool = tool_map.get(tool_name)
        step_record = {}
        if not tool:
            result = {
                "status": "error",
                "error": f"Tool '{tool_name}' not found."
            }
        else:
            try:
                output = await tool.ainvoke(tool_args)
                clean_output = unwrap_mcp_result(output)
                result = json.loads(clean_output)
                step_record = {
                    "tool": tool_name,
                    "args": tool_args,
                    "status": result.get("status", None),
                    # "summary": _summarize_result(tool_name, result)
                }
            except Exception as e:
                result = {
                    "status": "error",
                    "error": str(e)
                }
        logger.debug(f"<- Observation result: {result}")

        return {
            "messages": [ToolMessage(
                content=json.dumps(result),
                tool_call_id=tool_id
            )],
            "completed_steps": [step_record]
        }

    return tool_node
