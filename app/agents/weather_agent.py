"""LLM-powered weather agent."""

from collections.abc import Callable
from datetime import date, datetime
from zoneinfo import ZoneInfo

from app.agents.weather_schemas import WeatherRequest
from app.llm.client import LLMClient
from app.tools.weather import OpenMeteoClient, WeatherAPIError


WEATHER_REQUEST_PROMPT = """
You interpret weather requests for Atlas.

Use the user's complete message and conversation history to understand
the requested location, whether they want current conditions or a forecast,
and the dates they want.

Rules:
- Use request_type="current" for a request about conditions at the present
  time. Use request_type="forecast" for a future date or date range.
- Resolve references such as "there" using conversation history.
- Do not guess an unresolved location; return null instead.
- The reference date and Europe/Prague time zone are supplied in the input.
- Resolve relative dates and date ranges using that reference date.
- For a current-weather request, set start_date and end_date to null.
- For a forecast request, provide inclusive start_date and end_date.
- Return dates in YYYY-MM-DD format.
- Do not answer the weather question.
"""


WEATHER_ANSWER_PROMPT = """
You are Atlas's weather assistant.

Answer the original question using only the supplied weather data.

Rules:
- Do not invent observations or forecast values.
- Use Celsius for temperatures and km/h for wind speeds.
- Explain precipitation probabilities clearly when available.
- Describe weather codes only when their meaning is known confidently.
- State the relevant observation time or forecast dates.
- Do not invent values for missing fields.
- If the user asks for advice, base it on the available weather data.
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
        """Answer a weather question using current or forecast data."""

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
            return "Which city or location would you like the weather for?"

        try:
            location = self._weather_client.search_location(request.location)

            if request.request_type == "current":
                current = self._weather_client.get_current_weather(location)

                weather_lines = [
                    f"Observation time: {current.time}",
                ]

                if current.temperature_c is not None:
                    weather_lines.append(
                        f"Temperature: {current.temperature_c} °C"
                    )
                if current.apparent_temperature_c is not None:
                    weather_lines.append(
                        f"Feels like: {current.apparent_temperature_c} °C"
                    )
                if current.relative_humidity_percent is not None:
                    weather_lines.append(
                        f"Relative humidity: "
                        f"{current.relative_humidity_percent}%"
                    )
                if current.precipitation_mm is not None:
                    weather_lines.append(
                        f"Precipitation: {current.precipitation_mm} mm"
                    )
                if current.weather_code is not None:
                    weather_lines.append(
                        f"WMO weather code: {current.weather_code}"
                    )
                if current.wind_speed_kmh is not None:
                    weather_lines.append(
                        f"Wind speed: {current.wind_speed_kmh} km/h"
                    )
                if current.wind_direction_degrees is not None:
                    weather_lines.append(
                        "Wind direction: "
                        f"{current.wind_direction_degrees} degrees"
                    )

                answer_input = f"""
ORIGINAL USER QUESTION:
{user_message}

RESOLVED LOCATION:
{location.name}, {location.country or "country not specified"}

CURRENT CONDITIONS:
<weather_data>
{"\n".join(weather_lines)}
</weather_data>
"""

            else:
                if not request.start_date or not request.end_date:
                    return (
                        "I couldn't determine the forecast dates. "
                        "Could you clarify which dates you mean?"
                    )

                try:
                    start_date = date.fromisoformat(request.start_date)
                    end_date = date.fromisoformat(request.end_date)
                except ValueError:
                    return (
                        "I couldn't determine valid forecast dates. "
                        "Could you clarify which dates you mean?"
                    )

                if end_date < start_date:
                    return (
                        "I couldn't determine a valid date range. "
                        "Could you clarify the dates you want?"
                    )

                if start_date < today:
                    return (
                        "This forecast endpoint does not provide historical "
                        "weather for dates before today."
                    )

                forecast_days = (end_date - today).days + 1

                if forecast_days > 16:
                    return (
                        "The weather service supports forecasts up to "
                        "16 days ahead. Please choose a date range within "
                        "that period."
                    )

                forecasts = self._weather_client.get_daily_forecast(
                    location,
                    forecast_days=forecast_days,
                )

                selected_forecasts = [
                    forecast
                    for forecast in forecasts
                    if start_date.isoformat()
                    <= forecast.date
                    <= end_date.isoformat()
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
                            f"Maximum temperature: "
                            f"{forecast.temperature_max_c} °C"
                        )
                    if forecast.temperature_min_c is not None:
                        values.append(
                            f"Minimum temperature: "
                            f"{forecast.temperature_min_c} °C"
                        )
                    if (
                        forecast.precipitation_probability_max_percent
                        is not None
                    ):
                        values.append(
                            "Maximum precipitation probability: "
                            f"{forecast.precipitation_probability_max_percent}%"
                        )
                    if forecast.weather_code is not None:
                        values.append(
                            f"WMO weather code: {forecast.weather_code}"
                        )
                    if forecast.wind_speed_max_kmh is not None:
                        values.append(
                            f"Maximum wind speed: "
                            f"{forecast.wind_speed_max_kmh} km/h"
                        )

                    forecast_lines.append("; ".join(values))

                answer_input = f"""
ORIGINAL USER QUESTION:
{user_message}

RESOLVED LOCATION:
{location.name}, {location.country or "country not specified"}

REQUESTED DATES:
{start_date.isoformat()} through {end_date.isoformat()}

FORECAST DATA:
<weather_data>
{"\n".join(forecast_lines)}
</weather_data>
"""

        except WeatherAPIError:
            return (
                "I couldn't retrieve the weather data right now. "
                "Please try again shortly."
            )

        return self._llm.generate(
            system_prompt=WEATHER_ANSWER_PROMPT,
            user_message=answer_input,
        )
