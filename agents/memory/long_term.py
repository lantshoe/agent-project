import json
import os
import faiss
import numpy as np
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_core.messages import SystemMessage, HumanMessage
from agents.core.llm import get_llm
from agents.core.logger import get_logger
from dotenv import load_dotenv

from agents.core.skill_enum import Skill

"""
Session 1 (struggles with filter_rows):
  evaluate → quality: "partial", learning: "filter_rows needs operator+value"
  store → FAISS

Session 2 (same task type):
  retrieve → "filter_rows needs operator+value"
  planner sees memory → includes operator+value in plan
  execution → smooth, no mistakes
  evaluate → quality: "good", score: 9
  
"""

load_dotenv()
logger         = get_logger("long_term_memory")
MEMORY_DIR     = Path(os.getenv("MEMORY_DIR", "./agent_memory"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

PROFILE_MERGE_SYSTEM_PROMPT = """You are a user profile manager.
Given an existing user profile and observations from a new session,
produce an updated merged profile.

Rules:
- Update preferences if new session shows different behavior
- Increase skill_level if user handled complex tasks
- Keep all fields, update values that have changed
- Add new fields if new information is discovered

Respond ONLY with a valid JSON object."""

PROFILE_MERGE_USER_PROMPT = """Existing profile:
{old_profile}

New session observations:
{session_observations}

Produce an updated merged profile JSON.
"""


class LongTermMemory:
    """
    Long-term memory using FAISS for semantic search.
    Stores three types of memory:
    - user_profile: merged, always up to date (one record)
    - episodic: past conversation summaries
    - learnings: execution lessons the agent learned
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.memory_dir = MEMORY_DIR / user_id
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        self.index_path = self.memory_dir / "faiss.index"
        self.records_path = self.memory_dir / "records.json"
        self.profile_path = self.memory_dir / "profile.json"

        # Load embedding model
        logger.debug(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.dim = self.embedder.get_embedding_dimension()

        # Load or create FAISS index
        self._load_or_create_index()

    def _load_or_create_index(self):
        if self.index_path.exists() and self.records_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.records = json.loads(self.records_path.read_text())
            logger.debug(f"Loaded {len(self.records)} memories from disk")
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.records = []
            logger.debug("Created new FAISS index")

    def _save_index(self):
        faiss.write_index(self.index, str(self.index_path))
        self.records_path.write_text(json.dumps(self.records, indent=2))

    def _embed(self, text: str) -> np.ndarray:
        return self.embedder.encode([text], normalize_embeddings=True)

    def get_profile(self) -> dict:
        if self.profile_path.exists():
            return json.loads(self.profile_path.read_text())
        return {
            "user_id": self.user_id,
            "skill_level": "unknown",
            "prefers": [],
            "common_tasks": [],
            "created_at": datetime.now().isoformat()
        }

    def update_profile(self, session_observations: dict):
        """
        Uses LLM to merge old profile with new session observations.
        """
        old_profile = self.get_profile()

        llm = get_llm(skill=Skill.REASONING)

        try:
            format = PROFILE_MERGE_USER_PROMPT.format(old_profile=json.dumps(old_profile), session_observations=json.dumps(session_observations))
            response = llm.invoke([
                SystemMessage(content=PROFILE_MERGE_SYSTEM_PROMPT),
                HumanMessage(content=format)
            ])
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            new_profile = json.loads(content.strip())
            new_profile["updated_at"] = datetime.now().isoformat()
            self.profile_path.write_text(json.dumps(new_profile, indent=2))
            logger.debug(f"Profile updated: {new_profile}")

        except Exception as e:
            logger.error(f"Profile merge failed: {e} — keeping old profile")

    def store_episode(self,user_input: str, completed_steps: list,final_answer: str,evaluation: dict):
        """
        Stores a session as an episodic memory + learnings.
        """
        timestamp = datetime.now().isoformat()
        episode = {
            "type": "episode",
            "timestamp": timestamp,
            "user_input": user_input,
            "steps": [s.get("tool") for s in completed_steps],
            "outcome": evaluation.get("quality"),
            "score": evaluation.get("efficiency_score"),
            "summary": f"Task: {user_input[:100]}. "
                       f"Steps: {[s.get('tool') for s in completed_steps]}. "
                       f"Result: {evaluation.get('quality')}"
        }
        self._add_record(episode)

        # Store each learning separately for precise retrieval
        for learning in evaluation.get("learnings", []):
            record = {
                "type": "learning",
                "timestamp": timestamp,
                "learning": learning,
                "context": user_input[:100],
                "summary": learning
            }
            self._add_record(record)

        # Store suggested plan if execution was good
        if evaluation.get("quality") == "good" and evaluation.get("suggested_plan"):
            record = {
                "type": "plan_template",
                "timestamp": timestamp,
                "task_pattern": user_input[:100],
                "plan": evaluation.get("suggested_plan"),
                "summary": f"Good plan for: {user_input[:100]}"
            }
            self._add_record(record)

        self._save_index()
        logger.debug(f"Stored episode + {len(evaluation.get('learnings', []))} learnings")

    def _add_record(self, record: dict):
        vector = self._embed(record["summary"])
        self.index.add(vector)
        self.records.append(record)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieves most relevant memories for a given query.
        """
        if self.index.ntotal == 0:
            return []

        vector = self._embed(query)
        distances, indices = self.index.search(vector, min(top_k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.records):
                record = self.records[idx].copy()
                record["relevance_score"] = float(1 / (1 + dist))
                results.append(record)

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        logger.debug(f"Retrieved {len(results)} memories for query: '{query[:50]}'")
        return results

    def format_for_llm(self, query: str) -> str:
        """
        Retrieves memories and formats them as a context string for the LLM.
        """
        memories = self.retrieve(query)
        if not memories:
            return ""

        profile = self.get_profile()

        sections = [f"User profile: skill_level={profile.get('skill_level')}, "
                    f"prefers={profile.get('prefers')}"]

        episodes = [m for m in memories if m["type"] == "episode"]
        learnings = [m for m in memories if m["type"] == "learning"]
        templates = [m for m in memories if m["type"] == "plan_template"]

        if learnings:
            sections.append("Relevant learnings from past sessions:")
            for m in learnings[:3]:
                sections.append(f"  - {m['learning']}")

        if templates:
            sections.append("Known good plan for similar tasks:")
            for m in templates[:1]:
                sections.append(f"  Steps: {m['plan']}")

        if episodes:
            sections.append("Similar past tasks:")
            for m in episodes[:2]:
                sections.append(f"  - {m['summary']}")

        return "\n".join(sections)

