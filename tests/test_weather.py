import httpx
import pytest
from unittest.mock import Mock

from app.tools.weather import (
    DailyForecast,
    LocationNotFoundError,
    OpenMeteoClient,
    WeatherAPIError,
    WeatherLocation,
)


def test_search_location_returns_location():
    http_client = Mock()
    response = Mock()
    response.json.return_value = {
        "results": [
            {
                "name": "Prague",
                "country": "Czechia",
                "latitude": 50.0755,
                "longitude": 14.4378,
                "timezone": "Europe/Prague",
            }
        ]
    }
    http_client.get.return_value = response

    weather_client = OpenMeteoClient(client=http_client)
    location = weather_client.search_location("Prague")

    assert location.name == "Prague"
    assert location.country == "Czechia"
    assert location.latitude == 50.0755
    assert location.longitude == 14.4378
    assert location.timezone == "Europe/Prague"
    http_client.get.assert_called_once()


def test_search_location_raises_when_not_found():
    http_client = Mock()
    response = Mock()
    response.json.return_value = {"results": []}
    http_client.get.return_value = response

    weather_client = OpenMeteoClient(client=http_client)

    with pytest.raises(LocationNotFoundError):
        weather_client.search_location("UnknownPlace")


def test_search_location_rejects_empty_query():
    weather_client = OpenMeteoClient(client=Mock())

    with pytest.raises(ValueError):
        weather_client.search_location("   ")


def test_get_daily_forecast_parses_response():
    http_client = Mock()
    response = Mock()
    response.json.return_value = {
        "daily": {
            "time": ["2026-10-11", "2026-10-12"],
            "temperature_2m_max": [18.5, 17.0],
            "temperature_2m_min": [9.2, 8.5],
            "precipitation_probability_max": [20, 60],
            "weather_code": [1, 61],
            "wind_speed_10m_max": [12.0, 18.5],
        }
    }
    http_client.get.return_value = response

    weather_client = OpenMeteoClient(client=http_client)
    location = WeatherLocation(
        name="Prague",
        country="Czechia",
        latitude=50.0755,
        longitude=14.4378,
        timezone="Europe/Prague",
    )

    forecasts = weather_client.get_daily_forecast(location, forecast_days=2)

    assert len(forecasts) == 2
    assert isinstance(forecasts[0], DailyForecast)
    assert forecasts[0].date == "2026-10-11"
    assert forecasts[0].temperature_max_c == 18.5
    assert forecasts[0].temperature_min_c == 9.2
    assert forecasts[0].precipitation_probability_max_percent == 20
    assert forecasts[1].weather_code == 61
    assert forecasts[1].wind_speed_max_kmh == 18.5
    http_client.get.assert_called_once()


def test_get_daily_forecast_rejects_invalid_day_count():
    http_client = Mock()
    weather_client = OpenMeteoClient(client=http_client)
    location = WeatherLocation(
        name="Prague",
        latitude=50.0755,
        longitude=14.4378,
    )

    with pytest.raises(ValueError):
        weather_client.get_daily_forecast(location, forecast_days=0)

    http_client.get.assert_not_called()


def test_api_connection_error_is_wrapped():
    http_client = Mock()
    http_client.get.side_effect = httpx.ConnectError("Connection failed")

    weather_client = OpenMeteoClient(client=http_client)

    with pytest.raises(WeatherAPIError):
        weather_client.search_location("Prague")


def test_get_current_weather_parses_response():
    http_client = Mock()
    response = Mock()
    response.json.return_value = {
        "current": {
            "time": "2026-10-10T12:00",
            "temperature_2m": 14.1,
            "relative_humidity_2m": 65,
            "apparent_temperature": 13.5,
            "is_day": 1,
            "precipitation": 0.0,
            "weather_code": 2,
            "wind_speed_10m": 12.5,
            "wind_direction_10m": 240,
        }
    }
    http_client.get.return_value = response

    weather_client = OpenMeteoClient(client=http_client)
    location = WeatherLocation(
        name="Brno",
        country="Czechia",
        latitude=49.1951,
        longitude=16.6068,
        timezone="Europe/Prague",
    )

    current = weather_client.get_current_weather(location)

    assert current.time == "2026-10-10T12:00"
    assert current.temperature_c == 14.1
    assert current.relative_humidity_percent == 65
    assert current.apparent_temperature_c == 13.5
    assert current.precipitation_mm == 0.0
    assert current.wind_speed_kmh == 12.5

    params = http_client.get.call_args.kwargs["params"]
    assert "current" in params
    assert params["timezone"] == "Europe/Prague"


def test_get_current_weather_rejects_missing_current_data():
    http_client = Mock()
    response = Mock()
    response.json.return_value = {"daily": {"time": []}}
    http_client.get.return_value = response

    weather_client = OpenMeteoClient(client=http_client)
    location = WeatherLocation(
        name="Brno",
        latitude=49.1951,
        longitude=16.6068,
    )

    with pytest.raises(WeatherAPIError):
        weather_client.get_current_weather(location)
