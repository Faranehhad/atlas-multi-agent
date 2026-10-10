"""Weather data client using the public Open-Meteo APIs."""

import httpx
from pydantic import BaseModel


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherAPIError(RuntimeError):
    """Raised when a weather API request fails or returns invalid data."""


class LocationNotFoundError(WeatherAPIError):
    """Raised when the geocoding API cannot find a location."""


class WeatherLocation(BaseModel):
    """A location resolved by the Open-Meteo geocoding API."""

    name: str
    country: str | None = None
    latitude: float
    longitude: float
    timezone: str = "auto"


class DailyForecast(BaseModel):
    """Daily weather forecast for one date."""

    date: str
    temperature_max_c: float | None = None
    temperature_min_c: float | None = None
    precipitation_probability_max_percent: float | None = None
    weather_code: int | None = None
    wind_speed_max_kmh: float | None = None


class CurrentWeather(BaseModel):
    """Current weather conditions returned by Open-Meteo."""

    time: str
    temperature_c: float | None = None
    relative_humidity_percent: int | None = None
    apparent_temperature_c: float | None = None
    is_day: int | None = None
    precipitation_mm: float | None = None
    weather_code: int | None = None
    wind_speed_kmh: float | None = None
    wind_direction_degrees: float | None = None


class OpenMeteoClient:
    """Client for Open-Meteo geocoding and weather forecast endpoints."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._client = client or httpx.Client(timeout=timeout)

    def _get_json(self, url: str, params: dict) -> dict:
        """Request a JSON response and handle API or network errors."""

        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WeatherAPIError(
                "The Open-Meteo request failed or returned invalid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise WeatherAPIError(
                "The Open-Meteo API returned an unexpected response."
            )

        return payload

    def search_location(self, query: str) -> WeatherLocation:
        """Resolve a city or place name to coordinates and a time zone."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Location query must not be empty.")

        payload = self._get_json(
            GEOCODING_URL,
            params={
                "name": normalized_query,
                "count": 1,
                "language": "en",
                "format": "json",
            },
        )

        results = payload.get("results") or []
        if not results:
            raise LocationNotFoundError(
                f"No location was found for '{normalized_query}'."
            )

        result = results[0]

        try:
            return WeatherLocation(
                name=result["name"],
                country=result.get("country"),
                latitude=result["latitude"],
                longitude=result["longitude"],
                timezone=result.get("timezone") or "auto",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WeatherAPIError(
                "The Open-Meteo geocoding response was incomplete."
            ) from exc

    def get_daily_forecast(
        self,
        location: WeatherLocation,
        forecast_days: int = 7,
    ) -> list[DailyForecast]:
        """Retrieve a daily forecast for a resolved location."""

        if not 1 <= forecast_days <= 16:
            raise ValueError("forecast_days must be between 1 and 16.")

        payload = self._get_json(
            FORECAST_URL,
            params={
                "latitude": location.latitude,
                "longitude": location.longitude,
                "daily": (
                    "temperature_2m_max,"
                    "temperature_2m_min,"
                    "precipitation_probability_max,"
                    "weather_code,"
                    "wind_speed_10m_max"
                ),
                "timezone": location.timezone,
                "forecast_days": forecast_days,
            },
        )

        daily = payload.get("daily")
        if not isinstance(daily, dict) or not isinstance(
            daily.get("time"), list
        ):
            raise WeatherAPIError(
                "The Open-Meteo response does not contain daily forecasts."
            )

        dates = daily["time"]

        def value_at(field: str, index: int):
            values = daily.get(field)
            if isinstance(values, list) and index < len(values):
                return values[index]
            return None

        forecasts = []

        for index, forecast_date in enumerate(dates):
            forecasts.append(
                DailyForecast(
                    date=forecast_date,
                    temperature_max_c=value_at(
                        "temperature_2m_max", index
                    ),
                    temperature_min_c=value_at(
                        "temperature_2m_min", index
                    ),
                    precipitation_probability_max_percent=value_at(
                        "precipitation_probability_max", index
                    ),
                    weather_code=value_at("weather_code", index),
                    wind_speed_max_kmh=value_at(
                        "wind_speed_10m_max", index
                    ),
                )
            )

        return forecasts

    def get_current_weather(
        self,
        location: WeatherLocation,
    ) -> CurrentWeather:
        """Retrieve current weather conditions for a resolved location."""

        payload = self._get_json(
            FORECAST_URL,
            params={
                "latitude": location.latitude,
                "longitude": location.longitude,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "apparent_temperature,"
                    "is_day,"
                    "precipitation,"
                    "weather_code,"
                    "wind_speed_10m,"
                    "wind_direction_10m"
                ),
                "timezone": location.timezone,
            },
        )

        current = payload.get("current")
        if not isinstance(current, dict):
            raise WeatherAPIError(
                "The Open-Meteo response does not contain current conditions."
            )

        try:
            return CurrentWeather(
                time=current["time"],
                temperature_c=current.get("temperature_2m"),
                relative_humidity_percent=current.get("relative_humidity_2m"),
                apparent_temperature_c=current.get("apparent_temperature"),
                is_day=current.get("is_day"),
                precipitation_mm=current.get("precipitation"),
                weather_code=current.get("weather_code"),
                wind_speed_kmh=current.get("wind_speed_10m"),
                wind_direction_degrees=current.get("wind_direction_10m"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WeatherAPIError(
                "The Open-Meteo current-weather response was invalid."
            ) from exc
