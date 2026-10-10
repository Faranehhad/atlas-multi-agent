from datetime import date
from unittest.mock import Mock

from app.agents.weather_agent import WeatherAgent
from app.agents.weather_schemas import WeatherRequest
from app.tools.weather import (
    DailyForecast,
    WeatherAPIError,
    WeatherLocation,
)


TODAY = date(2026, 10, 10)


def make_agent(request, forecasts=None):
    """Create a WeatherAgent with mocked external dependencies."""

    llm = Mock()
    llm.structured.return_value = request
    llm.generate.return_value = "Here is your weather forecast."

    weather_client = Mock()

    weather_client.search_location.return_value = WeatherLocation(
        name="Prague",
        country="Czechia",
        latitude=50.0755,
        longitude=14.4378,
        timezone="Europe/Prague",
    )

    weather_client.get_daily_forecast.return_value = forecasts or []

    agent = WeatherAgent(
        llm=llm,
        weather_client=weather_client,
        today_provider=lambda: TODAY,
    )

    return agent, llm, weather_client


def test_weather_agent_returns_forecast_answer():
    request = WeatherRequest(
        location="Prague",
        start_date="2026-10-11",
        end_date="2026-10-12",
    )

    forecasts = [
        DailyForecast(
            date="2026-10-10",
            temperature_max_c=16.0,
            temperature_min_c=8.0,
        ),
        DailyForecast(
            date="2026-10-11",
            temperature_max_c=18.5,
            temperature_min_c=9.2,
            precipitation_probability_max_percent=20,
            weather_code=1,
            wind_speed_max_kmh=12.0,
        ),
        DailyForecast(
            date="2026-10-12",
            temperature_max_c=17.0,
            temperature_min_c=8.5,
            precipitation_probability_max_percent=60,
            weather_code=61,
            wind_speed_max_kmh=18.5,
        ),
    ]

    agent, llm, weather_client = make_agent(request, forecasts)

    answer = agent.answer(
        "What will the weather be like in Prague tomorrow and Sunday?"
    )

    assert answer == "Here is your weather forecast."

    llm.structured.assert_called_once()
    structured_kwargs = llm.structured.call_args.kwargs
    assert structured_kwargs["schema"] is WeatherRequest
    assert "REFERENCE DATE:\n2026-10-10" in structured_kwargs["user_message"]

    weather_client.search_location.assert_called_once_with("Prague")
    weather_client.get_daily_forecast.assert_called_once_with(
        weather_client.search_location.return_value,
        forecast_days=3,
    )

    llm.generate.assert_called_once()
    answer_input = llm.generate.call_args.kwargs["user_message"]

    # The requested date range is included, but the day before is excluded.
    assert "Date: 2026-10-11" in answer_input
    assert "Date: 2026-10-12" in answer_input
    assert "Date: 2026-10-10" not in answer_input


def test_weather_agent_asks_for_location_when_unknown():
    request = WeatherRequest(
        location=None,
        start_date="2026-10-10",
        end_date="2026-10-10",
    )

    agent, llm, weather_client = make_agent(request)

    answer = agent.answer("What will the weather be like?")

    assert "Which city or location" in answer
    weather_client.search_location.assert_not_called()
    llm.generate.assert_not_called()


def test_weather_agent_handles_invalid_date():
    request = WeatherRequest(
        location="Prague",
        start_date="not-a-date",
        end_date="2026-10-11",
    )

    agent, llm, weather_client = make_agent(request)

    answer = agent.answer("What's the weather in Prague?")

    assert "clarify" in answer
    weather_client.search_location.assert_not_called()
    llm.generate.assert_not_called()


def test_weather_agent_rejects_reversed_date_range():
    request = WeatherRequest(
        location="Prague",
        start_date="2026-10-12",
        end_date="2026-10-11",
    )

    agent, llm, weather_client = make_agent(request)

    answer = agent.answer("What's the weather in Prague?")

    assert "valid date range" in answer
    weather_client.search_location.assert_not_called()
    llm.generate.assert_not_called()


def test_weather_agent_rejects_forecast_beyond_supported_range():
    request = WeatherRequest(
        location="Prague",
        start_date="2026-10-27",
        end_date="2026-10-27",
    )

    agent, llm, weather_client = make_agent(request)

    answer = agent.answer("What's the weather in Prague on October 27?")

    assert "up to 16 days ahead" in answer
    weather_client.search_location.assert_not_called()
    llm.generate.assert_not_called()


def test_weather_agent_handles_api_failure():
    request = WeatherRequest(
        location="Prague",
        start_date="2026-10-11",
        end_date="2026-10-11",
    )

    agent, llm, weather_client = make_agent(request)
    weather_client.search_location.side_effect = WeatherAPIError(
        "Weather service unavailable"
    )

    answer = agent.answer("What's the weather in Prague tomorrow?")

    assert "couldn't retrieve the weather forecast" in answer
    llm.generate.assert_not_called()
