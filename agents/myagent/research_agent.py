from agents.models import AgentType
from agents.myagent.base_agent import BaseAgent
from agents.tools.search import web_search

RESEARCH_SYSTEM_PROMPT = """
You are a research specialist. Your only job is to find accurate,
up-to-date information by searching the web.

RULES:
- Use web_search for every information request
- Search multiple times if one search isn't enough
- Summarize findings clearly and concisely
- Always cite your sources
- Do NOT write files or perform calculations — that's another agent's job
- When you have enough information, give a clear structured summary
"""

def make_research_agent(max_steps: int=10, max_seconds:int = 600) -> BaseAgent:
    return BaseAgent(
        tools=[web_search],
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        agent_type=AgentType.RESEARCH,
        max_steps=max_steps,
        max_seconds=max_seconds,
    )