from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

from agents.core.skill_enum import Skill

load_dotenv()

SKILL_PROVIDER_MAP = {
    Skill.REASONING: os.getenv("LLM_PROVIDER", "ollama"),
    Skill.CODING: os.getenv("LLM_PROVIDER", "ollama"),
    Skill.SIMPLE: os.getenv("LLM_PROVIDER", "ollama"),
}


class LLMFactory:
    """
    Factory Pattern — creates the right LLM based on provider and skill.
    To add a new provider: just add a new _create_<provider> method.
    To route skills to different models: update SKILL_PROVIDER_MAP.
    """

    @staticmethod
    def create(skill: str = Skill.REASONING, temperature: float = 0.0) -> BaseChatModel:
        provider = SKILL_PROVIDER_MAP.get(skill, os.getenv("LLM_PROVIDER", "ollama"))
        provider_creators = {
            "claude": LLMFactory._create_claude,
            "ollama": LLMFactory._create_ollama,
        }
        creator = provider_creators.get(provider)
        if not creator:
            raise ValueError(f"Unknown provider: '{provider}'. Choose from: claude, ollama")

        return creator(temperature)

    @staticmethod
    def _create_claude(temperature: float) -> BaseChatModel:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in .env")
        return ChatAnthropic(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            api_key=api_key,
            temperature=temperature,
        )

    @staticmethod
    def _create_ollama(temperature: float) -> BaseChatModel:
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )


def get_llm(skill: str = Skill.REASONING, temperature: float = 0.0) -> BaseChatModel:
    return LLMFactory.create(skill=skill, temperature=temperature)

if __name__ == '__main__':
    llm = get_llm(skill='reasoning')
    response = llm.invoke('Say hello in one sentence.')
    print(response.content)