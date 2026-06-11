"""
Data processing specialist subagent.

Focused on file operations, Excel, and calculations.
Has access to filesystem tools, MCP Excel server, and calculator.
Does not search the web — it works with data it's given.
"""

from agents.models import AgentType
from agents.myagent.base_agent import BaseAgent
from agents.tools.calculator import calculator


DATA_SYSTEM_PROMPT = """
You are a data processing specialist. Your job is to work with files,
spreadsheets, and calculations.

EXCEL WORKFLOW — always follow this exact sequence:
  Step 1: create_workbook  → creates the empty file
  Step 2: write_sheet      → writes data into the workbook
  Step 3: read_workbook    → verify the data was written correctly

Never stop after create_workbook — always follow up with write_sheet.

RULES:
- Use calculator for all math
- Use filesystem tools to read, write, and organize files
- Do NOT search the web — that's another agent's job
- Be precise with numbers

FILESYSTEM RULES:
Sandbox root: agent_workspace
All file paths must be relative and inside agent_workspace/.
"""
def make_data_agent(mcp_tools: list=None, max_steps: int=15, max_seconds: int=600)->BaseAgent:
    tools = [calculator] + (mcp_tools or [])
    return BaseAgent(
        tools=tools,
        max_steps=max_steps,
        max_seconds=max_seconds,
        system_prompt=DATA_SYSTEM_PROMPT,
        agent_type=AgentType.DATA,
    )