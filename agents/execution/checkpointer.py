"""
Persistent checkpointing for all agents using LangGraph's sqlitesaver

Every agent graph compiled with get_checkpointer() automatically saves state
after every node execution,
If the process crashes mid-run, the run can be resumed from the last checkpoint using the same thread_id

Thread Id design:
supervisor run -> thread_id = run_id
subagent run -> thread_id = "{run_id}_{agent_type}_{task_id}"

this hierarchy means you can checkpoint supervisor state and each subagent state independently
and resume either level after a crash.

"""
from __future__ import annotations

import os
from pathlib import Path

from agents.core.logger import get_logger
logger = get_logger("checkpointer")

CHECKPOINT_DIR = Path(os.getenv("AGENT_CHECKPOINT_DIR", "./checkpoints"))

async def get_checkpointer():
    """
    return a sqlitesaver checkpointer for persistent state storage

    create the checkpoint directory if it doesn't exist
    one SQLite database per process - langgraoh handles concurrency.

    return None if langgraph-checkpoint-sqlite is not installed,
    so the agent degrades gracefully to in-memory only.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    db_path = CHECKPOINT_DIR / "agent_state.db"

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        checkpointer = AsyncSqliteSaver.from_conn_string(str(db_path))
        logger.debug(f"[checkpointer] Async checkpointing enabled: {db_path}")
        return checkpointer

    except ImportError:
        logger.warning(
            "[checkpointer] aiosqlite not installed. "
            "Falling back to MemorySaver. "
            "Install with: pip install aiosqlite"
        )
        try:
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()
        except ImportError:
            logger.warning("[checkpointer] No checkpointer available — running stateless")
            return None

def make_run_id(user_id: str) -> str:
    """Generates a unique run ID for thread_id assignment."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{ts}_{user_id}"

def get_checkpointer_cm(db_path: str):
    """Returns the AsyncSqliteSaver context manager — use with `async with`."""
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        return AsyncSqliteSaver.from_conn_string(db_path)
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

def make_subagent_thread_id(run_id: str, agent_type: str, task_id: str, user_id:str) -> str:
    """Generates a thread_id for a subagent run within a supervisor run."""
    return  f"{run_id}_{agent_type}_{task_id}" if run_id else f"{user_id}_{agent_type}_{task_id}"

