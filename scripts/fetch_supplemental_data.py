from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
PROCESSED = ROOT / "data" / "processed"
CONTEXT = PROCESSED / "context"
DASHBOARD = PROCESSED / "dashboard"

START_DATE = "2014-04-01"
END_DATE = date.today().isoformat()
NYC_LATITUDE = 40.7128
NYC_LONGITUDE = -74.0060

TLC_ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
TLC_ZONE_SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"


def ensure_dirs() -> None:
    (EXTERNAL / "tlc").mkdir(parents=True, exist_ok=True)
    (EXTERNAL / "weather").mkdir(parents=True, exist_ok=True)
    CONTEXT.mkdir(parents=True, exist_ok=True)
    DASHBOARD.mkdir(parents=True, exist_ok=True)


def download_file(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "uber-dashboard-data/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())


def fetch_open_meteo_daily_weather() -> pd.DataFrame:
    params = {
        "latitude": NYC_LATITUDE,
        "longitude": NYC_LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(
            [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "apparent_temperature_max",
                "apparent_temperature_min",
                "precipitation_sum",
                "rain_sum",
                "snowfall_sum",
                "precipitation_hours",
                "wind_speed_10m_max",
            ]
        ),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "America/New_York",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "uber-dashboard-data/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("error"):
        raise RuntimeError(payload.get("reason", "Open-Meteo returned an error"))

    raw_path = EXTERNAL / "weather" / "open_meteo_nyc_daily_2014_2015.json"
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    daily = pd.DataFrame(payload["daily"]).rename(columns={"time": "date"})
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    daily["temperature_avg_f"] = (
        daily["temperature_2m_max"] + daily["temperature_2m_min"]
    ) / 2
    daily["apparent_temperature_avg_f"] = (
        daily["apparent_temperature_max"] + daily["apparent_temperature_min"]
    ) / 2
    daily["has_rain"] = daily["rain_sum"] > 0
    daily["has_snow"] = daily["snowfall_sum"] > 0
    daily["has_precipitation"] = daily["precipitation_sum"] > 0
    daily = daily[
        [
            "date",
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_avg_f",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "apparent_temperature_avg_f",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "precipitation_hours",
            "wind_speed_10m_max",
            "has_rain",
            "has_snow",
            "has_precipitation",
        ]
    ]
    daily.to_csv(CONTEXT / "nyc_daily_weather.csv", index=False)
    return daily


def build_calendar_features() -> pd.DataFrame:
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    calendar = pd.DataFrame({"date": dates})
    # pandas exposes federal holiday dates directly, but not names in a compact frame.
    named_holidays = []
    for rule in USFederalHolidayCalendar().rules:
        for holiday_date in rule.dates(START_DATE, END_DATE):
            named_holidays.append(
                {"date": holiday_date.date(), "holiday_name": rule.name}
            )
    holiday_names = pd.DataFrame(named_holidays)

    calendar["date"] = calendar["date"].dt.date
    calendar["year"] = pd.to_datetime(calendar["date"]).dt.year
    calendar["month"] = pd.to_datetime(calendar["date"]).dt.month
    calendar["month_name"] = pd.to_datetime(calendar["date"]).dt.month_name()
    calendar["weekday"] = pd.to_datetime(calendar["date"]).dt.day_name()
    calendar["weekday_num"] = pd.to_datetime(calendar["date"]).dt.weekday
    calendar["week_start"] = (
        pd.to_datetime(calendar["date"])
        - pd.to_timedelta(pd.to_datetime(calendar["date"]).dt.weekday, unit="D")
    ).dt.date
    calendar["is_weekend"] = calendar["weekday_num"] >= 5
    calendar["is_month_start"] = pd.to_datetime(calendar["date"]).dt.is_month_start
    calendar["is_month_end"] = pd.to_datetime(calendar["date"]).dt.is_month_end

    if not holiday_names.empty:
        calendar = calendar.merge(holiday_names, on="date", how="left")
    else:
        calendar["holiday_name"] = pd.NA
    calendar["is_federal_holiday"] = calendar["holiday_name"].notna()

    # Major local events are maintained in data/processed/context/nyc_event_calendar.csv
    # so their categories, venues, and boroughs stay explicit.
    special_days = {}
    calendar["nyc_event"] = calendar["date"].astype(str).map(special_days)
    calendar["has_nyc_event"] = calendar["nyc_event"].notna()
    calendar.to_csv(CONTEXT / "calendar_features.csv", index=False)
    return calendar


def write_dashboard_context(weather: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    context = calendar.merge(weather, on="date", how="left")
    context.to_csv(DASHBOARD / "daily_context.csv", index=False)
    return context


def enrich_existing_dashboard_tables(context: pd.DataFrame) -> dict[str, int]:
    context_for_join = context.copy()
    context_for_join["pickup_date"] = context_for_join["date"].astype(str)
    context_for_join = context_for_join.drop(columns=["date"])

    table_names = [
        "uber_2014_daily_by_base",
        "uber_2015_daily_by_zone",
        "uber_2015_daily_by_borough",
        "uber_jan_feb_2015_daily_by_base",
        "fhv_2015_daily_by_base",
    ]
    row_counts: dict[str, int] = {}
    for table_name in table_names:
        input_path = DASHBOARD / f"{table_name}.csv"
        if not input_path.exists():
            continue
        df = pd.read_csv(input_path)
        if "pickup_date" not in df.columns:
            continue
        df["pickup_date"] = df["pickup_date"].astype(str)
        enriched = df.merge(context_for_join, on="pickup_date", how="left")
        output_name = f"{table_name}_with_context"
        enriched.to_csv(DASHBOARD / f"{output_name}.csv", index=False)
        row_counts[output_name] = len(enriched)
    return row_counts


def download_tlc_reference_files() -> None:
    lookup_path = EXTERNAL / "tlc" / "taxi_zone_lookup.csv"
    shapefile_path = EXTERNAL / "tlc" / "taxi_zones.zip"
    if not lookup_path.exists():
        download_file(TLC_ZONE_LOOKUP_URL, lookup_path)
    if not shapefile_path.exists():
        download_file(TLC_ZONE_SHAPEFILE_URL, shapefile_path)


def main() -> None:
    ensure_dirs()
    download_tlc_reference_files()
    weather = fetch_open_meteo_daily_weather()
    calendar = build_calendar_features()
    context = write_dashboard_context(weather, calendar)
    enriched_counts = enrich_existing_dashboard_tables(context)
    result = {
        "nyc_daily_weather": len(weather),
        "calendar_features": len(calendar),
        "daily_context": len(context),
        "enriched_dashboard_tables": enriched_counts,
        "tlc_reference_files": [
            str((EXTERNAL / "tlc" / "taxi_zone_lookup.csv").relative_to(ROOT)),
            str((EXTERNAL / "tlc" / "taxi_zones.zip").relative_to(ROOT)),
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
