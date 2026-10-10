from datetime import date
from unittest.mock import Mock

from app.agents.weather_agent import WeatherAgent
from app.agents.weather_schemas import WeatherRequest
from app.tools.weather import (
    CurrentWeather,
    DailyForecast,
    WeatherAPIError,
    WeatherLocation,
)


TODAY = date(2026, 10, 10)


def make_agent(request, forecasts=None, location=None):
    """Create a WeatherAgent with mocked external dependencies."""

    llm = Mock()
    llm.structured.return_value = request
    llm.generate.return_value = "Here is your weather answer."

    weather_client = Mock()
    weather_client.search_location.return_value = location or WeatherLocation(
        name="Prague",
        country="Czechia",
        latitude=50.0755,
        longitude=14.4378,
        timezone="Europe/Prague",
    )
    weather_client.get_daily_forecast.return_value = (
        forecasts if forecasts is not None else []
    )

    agent = WeatherAgent(
        llm=llm,
        weather_client=weather_client,
        today_provider=lambda: TODAY,
    )

    return agent, llm, weather_client


def test_weather_agent_returns_forecast_answer():
    request = WeatherRequest(
        location="Prague",
        request_type="forecast",
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

    assert answer == "Here is your weather answer."
    assert llm.structured.call_args.kwargs["schema"] is WeatherRequest
    weather_client.search_location.assert_called_once_with("Prague")
    weather_client.get_daily_forecast.assert_called_once_with(
        weather_client.search_location.return_value,
        forecast_days=3,
    )

    answer_input = llm.generate.call_args.kwargs["user_message"]
    assert "Date: 2026-10-11" in answer_input
    assert "Date: 2026-10-12" in answer_input
    assert "Date: 2026-10-10" not in answer_input


def test_weather_agent_returns_current_conditions():
    request = WeatherRequest(
        location="Brno",
        request_type="current",
    )

    brno = WeatherLocation(
        name="Brno",
        country="Czechia",
        latitude=49.1951,
        longitude=16.6068,
        timezone="Europe/Prague",
    )

    agent, llm, weather_client = make_agent(
        request,
        location=brno,
    )

    weather_client.get_current_weather.return_value = CurrentWeather(
        time="2026-10-10T12:00",
        temperature_c=14.1,
        relative_humidity_percent=65,
        apparent_temperature_c=13.5,
        is_day=1,
        precipitation_mm=0.0,
        weather_code=2,
        wind_speed_kmh=12.5,
        wind_direction_degrees=240,
    )

    answer = agent.answer("How is the weather now in Brno?")

    assert answer == "Here is your weather answer."
    weather_client.get_current_weather.assert_called_once_with(brno)
    weather_client.get_daily_forecast.assert_not_called()

    answer_input = llm.generate.call_args.kwargs["user_message"]
    assert "CURRENT CONDITIONS" in answer_input
    assert "Temperature: 14.1 °C" in answer_input
    assert "Relative humidity: 65%" in answer_input
    assert "Observation time: 2026-10-10T12:00" in answer_input


def test_weather_agent_asks_for_location_when_unknown():
    request = WeatherRequest(
        location=None,
        request_type="current",
    )

    agent, llm, weather_client = make_agent(request)

    answer = agent.answer("What will the weather be like?")

    assert "Which city or location" in answer
    weather_client.search_location.assert_not_called()
    llm.generate.assert_not_called()


def test_weather_agent_handles_invalid_date():
    request = WeatherRequest(
        location="Prague",
        request_type="forecast",
        start_date="not-a-date",
        end_date="2026-10-11",
    )

    agent, llm, weather_client = make_agent(request)

    answer = agent.answer("What's the weather in Prague?")

    assert "clarify" in answer
    weather_client.get_daily_forecast.assert_not_called()
    llm.generate.assert_not_called()


def test_weather_agent_rejects_reversed_date_range():
    request = WeatherRequest(
        location="Prague",
        request_type="forecast",
        start_date="2026-10-12",
        end_date="2026-10-11",
    )

    agent, llm, weather_client = make_agent(request)

    answer = agent.answer("What's the weather in Prague?")

    assert "valid date range" in answer
    weather_client.get_daily_forecast.assert_not_called()
    llm.generate.assert_not_called()


def test_weather_agent_rejects_forecast_beyond_supported_range():
    request = WeatherRequest(
        location="Prague",
        request_type="forecast",
        start_date="2026-10-27",
        end_date="2026-10-27",
    )

    agent, llm, weather_client = make_agent(request)

    answer = agent.answer("What's the weather in Prague on October 27?")

    assert "up to" in answer
    weather_client.get_daily_forecast.assert_not_called()
    llm.generate.assert_not_called()


def test_weather_agent_handles_api_failure():
    request = WeatherRequest(
        location="Prague",
        request_type="forecast",
        start_date="2026-10-11",
        end_date="2026-10-11",
    )

    agent, llm, weather_client = make_agent(request)
    weather_client.search_location.side_effect = WeatherAPIError(
        "Weather service unavailable"
    )

    answer = agent.answer("What's the weather in Prague tomorrow?")

    assert "couldn't retrieve the weather data" in answer
    llm.generate.assert_not_called()


def test_weather_agent_asks_to_clarify_missing_forecast_dates():
    request = WeatherRequest(
        location="Prague",
        request_type="forecast",
    )

    agent, llm, weather_client = make_agent(request)

    answer = agent.answer("What's the weather in Prague?")

    assert "clarify" in answer
    weather_client.search_location.assert_called_once_with("Prague")
    weather_client.get_daily_forecast.assert_not_called()
    llm.generate.assert_not_called()
