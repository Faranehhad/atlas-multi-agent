"""Schemas used by Atlas agents."""

from typing import Literal

from pydantic import BaseModel, Field


TaskType = Literal["weather", "city_info", "github"]


class Task(BaseModel):
    """A single task extracted from a user request."""

    id: str = Field(min_length=1)
    type: TaskType
    question: str = Field(min_length=1)
    dependencies: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    """Structured plan produced by the query analyzer."""

    tasks: list[Task] = Field(min_length=1)
