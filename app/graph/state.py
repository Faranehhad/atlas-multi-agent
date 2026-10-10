"""State definitions for the Atlas LangGraph workflow."""

import operator
from typing import Annotated, TypedDict

from app.agents.schemas import QueryPlan, Task


class TaskResult(TypedDict):
    """Result produced by a specialized agent."""

    task_id: str
    task_type: str
    answer: str


class AgentState(TypedDict):
    """Shared state carried through the Atlas graph."""

    user_message: str
    conversation_history: list[dict[str, str]]
    query_plan: QueryPlan | None
    task_results: Annotated[list[TaskResult], operator.add]
    final_answer: str | None


class TaskState(TypedDict):
    """State passed to an individual task worker."""

    task: Task
    conversation_history: list[dict[str, str]]
