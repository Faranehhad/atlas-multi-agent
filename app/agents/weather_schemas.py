"""Structured output schemas for the Weather Agent."""

from typing import Literal

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

    request_type: Literal["current", "forecast"] = Field(
        description=(
            "Whether the user requests current weather conditions "
            "or a forecast for a date or date range."
        ),
    )

    start_date: str | None = Field(
        default=None,
        description=(
            "Start date of a forecast in YYYY-MM-DD format. "
            "Null for a request for current conditions."
        ),
    )

    end_date: str | None = Field(
        default=None,
        description=(
            "End date of a forecast in YYYY-MM-DD format. "
            "Null for a request for current conditions."
        ),
    )
