"""
These models define the communication protocol between the supervisor
and its subagents, every piece of data that crosses an agent boundary
is typed here, no row dicts in coordination layer

Data flow:
use input
supervisor create delegation plan
each task dispatched to the right agent
each subagent returns a subagent result
supervisor aggregates subagent result to get final answer
"""
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """
    RESEARCH: web search, information gathering, summarization
    DATA: file operation, excel calculation, data processing
    GENERALIST: anything that doesnt fit a specialist
    """
    RESEARCH = "research"
    DATA = "data"
    GENERALIST = "generalist"

class Task(BaseModel):
    """
    a single unit of work the supervisor delegates to a subagent.

    the supervisor creates a list of Task from the user's request.
    each task is routed to the appropriate AgentType and run
    independently with its own context and budget,

    instruction: the specific instruction sent to the subagent
    context: any data from previous tasks this task  needs
    priority: higher number means higher priority.
    """
    task_id: str
    agent_type: AgentType
    instruction: str
    context: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    priority: int = 1

    @property
    def is_independent(self) -> bool:
        return len(self.depends_on) == 0

    def with_context(self, key:str, value:Any) -> "Task":
        """
        create a new Task with new information added, {key: value }
        """
        new_context = {**self.context, key: value}
        return self.model_copy(update={"context": new_context})

class SubagentResult(BaseModel):
    """
    the typed result a subagent returns to the supervisor.

    this is the only thing the supervisor sees from a subagent run,
    not the raw message history, not the individual tool calls.

    the subagent summerizes its work into this structured object.

    status: success|partial|failed
    """
    task_id: str
    agent_type: AgentType
    status: str
    answer: str
    steps_taken: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=datetime.now)

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"

    @property
    def usable(self) -> bool:
        """true if supervisor is can use this result even partially"""
        return self.status in ["success", "partial"]

    @property
    def short_summary(self) -> str:
        tool_str = ", ".join(self.steps_taken) if self.steps_taken else "no tools"
        return (
            f"{self.task_id} | {self.agent_type} | {self.status}"
            f"{self.answer}"
            f"(tools: {tool_str}, {self.duration_s}s)"
        )

class DelegationPlan(BaseModel):
    """
    the supervisor's full decomposition of a user request.

    create once at the start of a multi-agent run, contains all tasks plus the supervisors reasoning
    which gets logged for debugging.

    tasks: ordered list of tasks to execute
    reasoning: why the supervisor chose this decomposition, logged to run, not show to user
    strategy: the way how agents run, parallel, sequential, mixed
    """
    tasks: list[Task]
    reasoning: str = ""
    strategy: str = "parallel"

    @property
    def independent_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.is_independent]

    @property
    def dependent_tasks(self) -> list[Task]:
        return [t for t in self.tasks if not t.is_independent]

    def tasks_for_agent(self, agent_type: AgentType) -> list[Task]:
        """return all tasks assigned to a specific agent type"""
        return [t for t in self.tasks if t.agent_type == agent_type]

    def get_task(self, task_id: str) -> Task | None:
        return next((t for t in self.tasks if t.task_id == task_id), None)

    def summary(self) -> str:
        counts = {}
        for t in self.tasks:
            counts[t.agent_type] = counts.get(t.agent_type, 0) + 1
        parts = [f"{v}x {k}" for k, v in counts.items()]
        return f"{len(self.tasks)} tasks ({', '.join(parts)}) — strategy: {self.strategy}"


class SupervisorState(BaseModel):
    """
    traces the full state of  a supervisor run,

    Used for logging, checkpointing, and partial failure recovery.
    The supervisor updates this as tasks complete or fail.
    """
    run_id: str
    user_input: str
    plan: DelegationPlan|None = None
    results: list[SubagentResult] = Field(default_factory=list)
    failed_tasks: list[str] = Field(default_factory=list)
    start_at: datetime = Field(default_factory=datetime.now)

    @property
    def completed_task_ids(self) -> list[str]:
        return [r.task_id for r in self.results]

    @property
    def successful_results(self) -> list[SubagentResult]:
        return [r for r in self.results if r.usable]

    @property
    def is_complete(self) -> bool:
        if not self.plan:
            return False
        return len(self.completed_task_ids) == len(self.plan.tasks)

    @property
    def success_rate(self) -> float:
        if not self.is_complete:
            return 0.0
        return len(self.successful_results) / len(self.completed_task_ids)

    def add_result(self, result: SubagentResult) -> None:
        self.results.append(result)
        if result.is_failed:
            self.failed_tasks.append(result.task_id)

    def context_for_supervisor(self) -> str:
        if not self.results:
            return "No results yet."

        return "\n".join([r.short_summary for r in self.successful_results])






