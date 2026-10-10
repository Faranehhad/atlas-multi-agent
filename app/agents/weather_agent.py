"""LLM-powered weather agent."""

from collections.abc import Callable
from datetime import date, datetime
from zoneinfo import ZoneInfo

from app.agents.weather_schemas import WeatherRequest
from app.llm.client import LLMClient
from app.tools.weather import (
    OpenMeteoClient,
    WeatherAPIError,
)


WEATHER_REQUEST_PROMPT = """
You interpret weather questions for Atlas.

Extract the requested location and inclusive start/end dates.

Rules:
- Use the reference date supplied in the user message to resolve relative
  dates, weekdays, weekends, and other natural-language date expressions.
- Use conversation history to resolve references and omitted locations.
- Do not guess a location. Return null if it cannot be resolved.
- If no period is specified, use the reference date for both dates.
- Return dates in YYYY-MM-DD format.
- Set start_date and end_date to the complete requested date range.
- Do not answer the weather question.
"""


WEATHER_ANSWER_PROMPT = """
You are Atlas's weather assistant.

Answer the user's original question using only the supplied forecast data.

Rules:
- Do not invent weather observations or forecast values.
- Use Celsius for temperatures and km/h for wind speeds.
- Explain precipitation probabilities clearly when available.
- Explain the weather conditions using the supplied weather codes only
  when you can do so confidently.
- State the relevant forecast dates.
- If a data field is missing, do not invent it.
- If the user asks for advice, such as whether to bring an umbrella,
  base it on the forecast data and explain the reason.
- Be clear and concise.
"""


def _prague_today() -> date:
    """Return today's date in the Europe/Prague time zone."""

    return datetime.now(ZoneInfo("Europe/Prague")).date()


class WeatherAgent:
    """Interpret weather questions and answer using Open-Meteo data."""

    def __init__(
        self,
        llm: LLMClient,
        weather_client: OpenMeteoClient,
        today_provider: Callable[[], date] | None = None,
    ) -> None:
        self._llm = llm
        self._weather_client = weather_client
        self._today_provider = today_provider or _prague_today

    def answer(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Answer a weather question using live forecast data."""

        today = self._today_provider()
        history = conversation_history or []

        history_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in history
        )

        interpretation_input = f"""
REFERENCE DATE:
{today.isoformat()}
Reference time zone: Europe/Prague

CONVERSATION HISTORY:
<conversation_history>
{history_text}
</conversation_history>

CURRENT USER QUESTION:
<user_question>
{user_message}
</user_question>
"""

        request = self._llm.structured(
            schema=WeatherRequest,
            system_prompt=WEATHER_REQUEST_PROMPT,
            user_message=interpretation_input,
        )

        if not request.location or not request.location.strip():
            return "Which city or location would you like the weather forecast for?"

        try:
            start_date = date.fromisoformat(request.start_date)
            end_date = date.fromisoformat(request.end_date)
        except ValueError:
            return (
                "I couldn't determine the requested forecast dates. "
                "Could you clarify which dates you mean?"
            )

        if end_date < start_date:
            return (
                "I couldn't determine a valid date range. "
                "Could you clarify the dates you want?"
            )

        if start_date < today:
            return (
                "I can retrieve forecasts for today and future dates, "
                "but the current weather source does not provide "
                "historical weather through this forecast endpoint."
            )

        forecast_days = (end_date - today).days + 1

        if forecast_days > 16:
            return (
                "The current weather source supports forecasts up to "
                "16 days ahead. Please choose a date range within that period."
            )

        try:
            location = self._weather_client.search_location(request.location)
            forecasts = self._weather_client.get_daily_forecast(
                location,
                forecast_days=forecast_days,
            )
        except WeatherAPIError:
            return (
                "I couldn't retrieve the weather forecast right now. "
                "Please try again shortly."
            )

        selected_forecasts = [
            forecast
            for forecast in forecasts
            if start_date.isoformat() <= forecast.date <= end_date.isoformat()
        ]

        if not selected_forecasts:
            return (
                "The weather service returned no forecast data "
                "for the requested dates."
            )

        forecast_lines = []

        for forecast in selected_forecasts:
            values = [f"Date: {forecast.date}"]

            if forecast.temperature_max_c is not None:
                values.append(
                    f"Maximum temperature: {forecast.temperature_max_c} °C"
                )

            if forecast.temperature_min_c is not None:
                values.append(
                    f"Minimum temperature: {forecast.temperature_min_c} °C"
                )

            if forecast.precipitation_probability_max_percent is not None:
                values.append(
                    "Maximum precipitation probability: "
                    f"{forecast.precipitation_probability_max_percent}%"
                )

            if forecast.weather_code is not None:
                values.append(f"WMO weather code: {forecast.weather_code}")

            if forecast.wind_speed_max_kmh is not None:
                values.append(
                    f"Maximum wind speed: {forecast.wind_speed_max_kmh} km/h"
                )

            forecast_lines.append("; ".join(values))

        country = f", {location.country}" if location.country else ""

        answer_input = f"""
ORIGINAL USER QUESTION:
{user_message}

RESOLVED LOCATION:
{location.name}{country}

REQUESTED DATES:
{start_date.isoformat()} through {end_date.isoformat()}

FORECAST DATA:
<forecast_data>
{"\n".join(forecast_lines)}
</forecast_data>
"""

        return self._llm.generate(
            system_prompt=WEATHER_ANSWER_PROMPT,
            user_message=answer_input,
        )
