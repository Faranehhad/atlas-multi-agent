"""Structured output schemas for the Weather Agent."""

from pydantic import BaseModel, Field


class WeatherRequest(BaseModel):
    """Weather request interpreted by the LLM."""

    location: str | None = Field(
        default=None,
        description=(
            "The requested city, region, or place. "
            "Use null if it cannot be resolved from the question "
            "or conversation history."
        ),
    )

    start_date: str = Field(
        description="Start date of the requested forecast in YYYY-MM-DD format.",
    )

    end_date: str = Field(
        description="End date of the requested forecast in YYYY-MM-DD format.",
    )
