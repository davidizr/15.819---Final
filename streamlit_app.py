from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st


ROOT = Path(__file__).resolve().parent
DASHBOARD = ROOT / "data" / "processed" / "dashboard"
PROCESSED = ROOT / "data" / "processed"

ACCENT_BLUE = "#ff8a3d"
ACCENT_BLUE_DARK = "#ff5a1f"
ACCENT_BLUE_SOFT = "#ffe2d0"
CYAN = "#ff9f5a"
AQUA = "#5dade2"
INDIGO = "#8b98a8"
INK = "#17202a"
BLUE = ACCENT_BLUE_DARK
GREEN = "#257a4a"
GOLD = "#b27600"
GRAY = "#6b7280"
NEGATIVE_RED = "#c43d3d"
POSITIVE_GREEN = "#2e7d32"
NEUTRAL_GRAY = "#9ca3af"
ANALYSIS_START_DATE = pd.Timestamp("2015-01-01")
SOURCE_CHANGE_DATE = pd.Timestamp("2019-02-01")
BOROUGH_ORDER = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR", "Unknown"]
HEATMAP_SCALE = [
    [0.0, "#fff0e6"],
    [0.2, "#ffd1ad"],
    [0.45, "#ff9f5a"],
    [0.7, "#f36b2b"],
    [1.0, "#9f320f"],
]
EVENT_CATEGORY_ORDER = [
    "Federal holiday",
    "Sports",
    "Concerts & music festivals",
    "Parades & civic festivals",
    "Convention / expo",
    "Civic / city event",
    "Other event",
]
EVENT_CATEGORY_ALIASES = {
    "Sporting event": "Sports",
    "Sports event": "Sports",
    "Concert/festival": "Concerts & music festivals",
    "Concert": "Concerts & music festivals",
    "Festival/parade": "Parades & civic festivals",
    "Parade": "Parades & civic festivals",
    "Entertainment event": "Civic / city event",
    "NYC event": "Civic / city event",
    "Custom event": "Other event",
}

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    BLUE,
    ACCENT_BLUE,
    "#ffb36f",
    "#c9471b",
    "#f4a261",
    "#ffd1ad",
    "#8b98a8",
]


st.set_page_config(
    page_title="Uber NYC Performance Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
      .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
      h1, h2, h3 { letter-spacing: 0 !important; color: #f4f7fb; }
      h1 { font-size: 2.05rem !important; }
      h2 { font-size: 1.35rem !important; margin-top: .25rem; }
      h3 { font-size: 1.05rem !important; }
      div[data-testid="stMetric"] {
        border: 1px solid #ff5a1f;
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        background: rgba(255, 138, 61, 0.08);
      }
      div[data-testid="stMetricLabel"] { color: inherit; opacity: .82; }
      div[data-testid="stMetricValue"] { color: inherit; }
      .muted { color: #aeb8c5; font-size: .92rem; }
      .section-note { color: #aeb8c5; margin-top: -0.4rem; margin-bottom: 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(name: str, parse_dates: tuple[str, ...] = ("pickup_date",)) -> pd.DataFrame:
    path = DASHBOARD / name
    if not path.exists():
        raise FileNotFoundError(f"Missing dashboard table: {path}")
    df = pd.read_csv(path, low_memory=False)
    for col in parse_dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"])
    return df


@st.cache_data(show_spinner=False)
def load_trip_sample(max_points: int = 80_000) -> pd.DataFrame:
    path = PROCESSED / "uber_2014_trips.parquet"
    trips = pd.read_parquet(
        path, columns=["pickup_datetime", "lat", "lon", "base_name"]
    )
    trips["pickup_datetime"] = pd.to_datetime(trips["pickup_datetime"])
    trips["pickup_month"] = trips["pickup_datetime"].dt.to_period("M").astype(str)
    trips["hour"] = trips["pickup_datetime"].dt.hour
    if len(trips) > max_points:
        trips = trips.sample(max_points, random_state=42)
    return trips


@st.cache_data(show_spinner=False)
def load_all_tables() -> dict[str, pd.DataFrame]:
    tables = {
        "uber_2014_daily": load_csv("uber_2014_daily_by_base_with_context.csv"),
        "uber_2014_hourly": load_csv("uber_2014_hourly_by_base.csv"),
        "uber_2014_weekday_hour": load_csv(
            "uber_2014_weekday_hour_by_base.csv", parse_dates=()
        ),
        "uber_2015_borough": load_csv("uber_2015_daily_by_borough_with_context.csv"),
        "uber_2015_zone": load_csv("uber_2015_daily_by_zone_with_context.csv"),
        "uber_2015_supply": load_csv("uber_jan_feb_2015_daily_by_base_with_context.csv"),
        "fhv_2015": load_csv("fhv_2015_daily_by_base_with_context.csv"),
        "context": load_csv("daily_context.csv", parse_dates=("date",)),
    }
    modern_daily = DASHBOARD / "modern_hvfhv_daily_by_borough.csv"
    if modern_daily.exists():
        tables.update(
            {
                "modern_borough": load_csv("modern_hvfhv_daily_by_borough.csv"),
                "modern_zone": load_csv("modern_hvfhv_daily_by_zone.csv"),
                "modern_hourly": load_csv("modern_hvfhv_hourly_by_company.csv"),
                "modern_weekday_hour": load_csv(
                    "modern_hvfhv_weekday_hour_by_company.csv", parse_dates=()
                ),
                "modern_company": load_csv("modern_hvfhv_daily_by_company.csv"),
            }
        )
    unified_daily = DASHBOARD / "unified" / "unified_uber_daily_by_borough.csv"
    if unified_daily.exists():
        unified_dir = DASHBOARD / "unified"
        tables.update(
            {
                "unified_borough": pd.read_csv(
                    unified_dir / "unified_uber_daily_by_borough.csv",
                    low_memory=False,
                    parse_dates=["pickup_date"],
                ),
                "unified_zone": pd.read_csv(
                    unified_dir / "unified_uber_daily_by_zone.csv",
                    low_memory=False,
                    parse_dates=["pickup_date"],
                ),
                "unified_hourly": pd.read_csv(
                    unified_dir / "unified_uber_hourly_by_company.csv",
                    low_memory=False,
                    parse_dates=["pickup_date"],
                ),
                "unified_annual": pd.read_csv(
                    unified_dir / "unified_uber_annual_summary.csv",
                    low_memory=False,
                ),
            }
        )
    return tables


@st.cache_data(show_spinner=False)
def load_zone_centroids() -> pd.DataFrame:
    path = PROCESSED / "dim_taxi_zone_centroids.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=["location_id", "borough", "zone", "latitude", "longitude"]
        )
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_custom_events(mtime: float = 0.0) -> pd.DataFrame:
    path = PROCESSED / "context" / "nyc_event_calendar.csv"
    columns = ["date", "event_name", "event_category", "venue", "borough"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    events = pd.read_csv(path)
    if "date" not in events.columns or "event_name" not in events.columns:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in events.columns:
            events[column] = pd.NA
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    events = events.dropna(subset=["date", "event_name"])
    events["event_category"] = events["event_category"].map(normalize_event_category)
    return events[columns]


def fmt_int(value: float | int) -> str:
    return f"{value:,.0f}"


def fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.1%}"


def normalize_event_category(category: object) -> str:
    if pd.isna(category):
        return "Other event"
    text = str(category).strip()
    return EVENT_CATEGORY_ALIASES.get(text, text or "Other event")


def infer_nyc_event_category(event_name: object) -> str:
    if pd.isna(event_name):
        return "Civic / city event"
    name = str(event_name).lower()
    if "open" in name or "marathon" in name or "half" in name:
        return "Sports"
    if "festival" in name or "parade" in name or "pride" in name:
        return "Parades & civic festivals"
    if "concert" in name or "ball" in name:
        return "Concerts & music festivals"
    return "Civic / city event"


def ordered_event_categories(categories: list[str] | pd.Series) -> list[str]:
    available = [normalize_event_category(category) for category in categories]
    ordered = [category for category in EVENT_CATEGORY_ORDER if category in available]
    extras = sorted(set(available).difference(EVENT_CATEGORY_ORDER))
    return ordered + extras


def weekly_delta(daily_totals: pd.DataFrame) -> float | None:
    if daily_totals.empty or daily_totals["pickup_date"].nunique() < 14:
        return None
    by_day = daily_totals.groupby("pickup_date", as_index=False)["trips"].sum()
    by_day = by_day.sort_values("pickup_date").tail(14)
    current = by_day.tail(7)["trips"].sum()
    prior = by_day.head(7)["trips"].sum()
    if prior == 0:
        return None
    return current / prior - 1


def period_filter(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    date_col: str = "pickup_date",
) -> pd.DataFrame:
    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    return df.loc[mask].copy()


def total_by_day(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("pickup_date", as_index=False)["trips"]
        .sum()
        .sort_values("pickup_date")
    )


def with_period_rolling(df: pd.DataFrame) -> pd.DataFrame:
    out = total_by_day(df)
    out["rolling_7d"] = out["trips"].rolling(7, min_periods=1).mean()
    return out


FORECAST_MODELS = ["SARIMA", "Holt-Winters", "Regression + seasonality", "Seasonal naive"]


def _seasonal_naive_forecast(train_series: pd.Series, steps: int) -> pd.Series:
    pattern = train_series.tail(min(7, len(train_series)))
    if pattern.empty:
        return pd.Series([0] * steps)
    return pd.Series([pattern.iloc[i % len(pattern)] for i in range(steps)])


def _fit_holt_winters(train_series: pd.Series, steps: int) -> pd.Series:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    model = ExponentialSmoothing(
        train_series,
        trend="add",
        seasonal="add",
        seasonal_periods=7,
        initialization_method="estimated",
    ).fit(optimized=True)
    return model.forecast(steps)


def _fit_sarima(train_series: pd.Series, steps: int) -> pd.Series:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        train_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    return model.forecast(steps=steps)


def _forecast_context_frame(
    context: pd.DataFrame | None,
    full_index: pd.DatetimeIndex,
    forecast_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    base = pd.DataFrame({"pickup_date": full_index.union(forecast_index)})
    if context is not None and not context.empty:
        context_frame = context.rename(columns={"date": "pickup_date"}).copy()
        context_frame["pickup_date"] = pd.to_datetime(context_frame["pickup_date"])
        base = base.merge(context_frame, on="pickup_date", how="left")

    base["weekday_num"] = base["pickup_date"].dt.weekday
    base["month"] = base["pickup_date"].dt.month
    base["day_of_year"] = base["pickup_date"].dt.dayofyear
    for column in ["is_weekend", "is_month_start", "is_month_end", "is_federal_holiday", "has_nyc_event", "has_precipitation"]:
        if column not in base.columns:
            base[column] = False
        base[column] = base[column].fillna(False).astype(bool)
    for column in ["temperature_avg_f", "precipitation_sum_in"]:
        if column not in base.columns:
            base[column] = np.nan
        climatology = base.groupby("day_of_year")[column].transform("mean")
        monthly = base.groupby("month")[column].transform("mean")
        base[column] = base[column].fillna(climatology).fillna(monthly).fillna(base[column].mean()).fillna(0)
    return base


def _regression_features(frame: pd.DataFrame, train_start: pd.Timestamp) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    days = (frame["pickup_date"] - train_start).dt.days.astype(float)
    out["trend"] = days
    out["trend_sq"] = days ** 2
    out["is_weekend"] = frame["is_weekend"].astype(int)
    out["is_month_start"] = frame["is_month_start"].astype(int)
    out["is_month_end"] = frame["is_month_end"].astype(int)
    out["is_federal_holiday"] = frame["is_federal_holiday"].astype(int)
    out["has_nyc_event"] = frame["has_nyc_event"].astype(int)
    out["has_precipitation"] = frame["has_precipitation"].astype(int)
    out["temperature_avg_f"] = frame["temperature_avg_f"].astype(float)
    out["precipitation_sum_in"] = frame["precipitation_sum_in"].astype(float)
    for period in [7, 365.25]:
        angle = 2 * np.pi * days / period
        suffix = "weekly" if period == 7 else "annual"
        out[f"sin_{suffix}"] = np.sin(angle)
        out[f"cos_{suffix}"] = np.cos(angle)
    weekday_dummies = pd.get_dummies(frame["weekday_num"], prefix="dow", dtype=float)
    month_dummies = pd.get_dummies(frame["month"], prefix="month", dtype=float)
    return pd.concat([out, weekday_dummies, month_dummies], axis=1).astype(float)


def _fit_regression_forecast(
    train_series: pd.Series,
    forecast_index: pd.DatetimeIndex,
    context: pd.DataFrame | None = None,
) -> pd.Series:
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_start = train_series.index.min()
    all_frame = _forecast_context_frame(context, train_series.index, forecast_index)
    features = _regression_features(all_frame, train_start)
    feature_frame = all_frame[["pickup_date"]].join(features)
    train_x = feature_frame[feature_frame["pickup_date"].isin(train_series.index)].drop(columns=["pickup_date"])
    future_x = feature_frame[feature_frame["pickup_date"].isin(forecast_index)].drop(columns=["pickup_date"])
    train_y = train_series.reindex(
        feature_frame[feature_frame["pickup_date"].isin(train_series.index)]["pickup_date"]
    ).values
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0]),
    )
    model.fit(train_x, train_y)
    return pd.Series(model.predict(future_x), index=forecast_index)


def _fit_forecast_model(
    train_series: pd.Series,
    steps: int,
    model_name: str = "SARIMA",
    forecast_index: pd.DatetimeIndex | None = None,
    context: pd.DataFrame | None = None,
) -> pd.Series:
    """Fit the selected daily forecasting model."""
    import warnings

    selected = model_name if model_name in FORECAST_MODELS else "SARIMA"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if selected == "Holt-Winters":
                forecast = _fit_holt_winters(train_series, steps)
            elif selected == "Regression + seasonality":
                if forecast_index is None:
                    raise ValueError("Regression forecast requires a forecast index.")
                forecast = _fit_regression_forecast(train_series, forecast_index, context)
            elif selected == "Seasonal naive":
                forecast = _seasonal_naive_forecast(train_series, steps)
            else:
                forecast = _fit_sarima(train_series, steps)
    except Exception:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                forecast = _fit_holt_winters(train_series, steps)
            except Exception:
                forecast = _seasonal_naive_forecast(train_series, steps)
    return forecast.clip(lower=0)


def build_forecast(
    df_full: pd.DataFrame,
    forecast_months: int = 1,
    model_name: str = "SARIMA",
    context: pd.DataFrame | None = None,
):
    """Return (daily_actuals, None, future_forecast) DataFrames."""

    daily = total_by_day(df_full)
    daily["pickup_date"] = pd.to_datetime(daily["pickup_date"])
    if daily.empty:
        return None, None, None

    series = (
        daily.set_index("pickup_date")["trips"]
        .sort_index()
        .asfreq("D", fill_value=0)
    )

    train_start = pd.Timestamp("2021-01-01")
    final_train_end = series.index.max()
    forecast_start = final_train_end + pd.Timedelta(days=1)
    forecast_months = max(1, min(int(forecast_months), 12))
    forecast_end = forecast_start + pd.DateOffset(months=forecast_months) - pd.Timedelta(days=1)

    train_final = series.loc[train_start:final_train_end]
    if len(train_final) < 365:
        return None, None, None

    forecast_index = pd.date_range(forecast_start, forecast_end, freq="D")
    forecast_values = _fit_forecast_model(
        train_final,
        len(forecast_index),
        model_name,
        forecast_index=forecast_index,
        context=context,
    )
    future = pd.DataFrame(
        {
            "pickup_date": forecast_index,
            "trips_forecast": forecast_values.values,
        }
    )

    daily_df = series.reset_index()
    daily_df.columns = ["pickup_date", "trips"]
    return daily_df, None, future


_GRANULARITY_FREQ = {"Daily": "D", "Weekly": "W-MON", "Monthly": "MS", "Yearly": "YS"}


def period_start_for_granularity(dates: pd.Series, granularity: str) -> pd.Series:
    dates = pd.to_datetime(dates)
    if granularity == "Weekly":
        return dates - pd.to_timedelta(dates.dt.weekday, unit="D")
    if granularity == "Monthly":
        return dates.dt.to_period("M").dt.to_timestamp()
    if granularity == "Yearly":
        return dates.dt.to_period("Y").dt.to_timestamp()
    return dates


def period_label_for_granularity(dates: pd.Series, granularity: str) -> pd.Series:
    dates = pd.to_datetime(dates)
    if granularity == "Weekly":
        return "Week of " + dates.dt.strftime("%b %d, %Y")
    if granularity == "Monthly":
        return dates.dt.strftime("%b %Y")
    if granularity == "Yearly":
        return dates.dt.strftime("%Y")
    return dates.dt.strftime("%b %d, %Y")


def resample_to_granularity(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    freq = _GRANULARITY_FREQ.get(granularity, "D")
    daily = total_by_day(df)
    if freq == "D":
        return daily
    daily["pickup_date"] = pd.to_datetime(daily["pickup_date"])
    daily["period_start"] = period_start_for_granularity(
        daily["pickup_date"], granularity
    )
    resampled = (
        daily.groupby("period_start", as_index=False)["trips"]
        .sum()
        .rename(columns={"period_start": "pickup_date"})
    )
    resampled["period_label"] = period_label_for_granularity(
        resampled["pickup_date"], granularity
    )
    return resampled


def resample_series_for_granularity(
    series_df: pd.DataFrame,
    granularity: str,
    value_col: str,
) -> pd.DataFrame:
    if series_df is None or series_df.empty:
        return pd.DataFrame(columns=["pickup_date", value_col])
    freq = _GRANULARITY_FREQ.get(granularity, "D")
    out = series_df.copy()
    out["pickup_date"] = pd.to_datetime(out["pickup_date"])
    out = out.sort_values("pickup_date")
    if freq == "D":
        return out[["pickup_date", value_col]]
    out["period_start"] = period_start_for_granularity(out["pickup_date"], granularity)
    aggregated = (
        out.groupby("period_start", as_index=False)[value_col]
        .sum()
        .rename(columns={"period_start": "pickup_date"})
    )
    aggregated["period_label"] = period_label_for_granularity(
        aggregated["pickup_date"], granularity
    )
    return aggregated


def growth_stats(df: pd.DataFrame) -> dict:
    daily = total_by_day(df)
    daily["pickup_date"] = pd.to_datetime(daily["pickup_date"])
    daily = daily.sort_values("pickup_date")
    if daily.empty:
        return {"dod": None, "wow": None, "mom": None, "yoy": None}

    last = daily["pickup_date"].max()

    def window_sum(start, end):
        mask = (daily["pickup_date"] >= start) & (daily["pickup_date"] <= end)
        return daily.loc[mask, "trips"].sum()

    def pct(cur, prev):
        return (cur - prev) / prev if prev else None

    dod = pct(
        window_sum(last, last),
        window_sum(last - pd.Timedelta(days=1), last - pd.Timedelta(days=1)),
    )
    wow = pct(
        window_sum(last - pd.Timedelta(days=6), last),
        window_sum(last - pd.Timedelta(days=13), last - pd.Timedelta(days=7)),
    )
    mom = pct(
        window_sum(last - pd.Timedelta(days=29), last),
        window_sum(last - pd.Timedelta(days=59), last - pd.Timedelta(days=30)),
    )
    yoy = pct(
        window_sum(last - pd.Timedelta(days=364), last),
        window_sum(last - pd.Timedelta(days=729), last - pd.Timedelta(days=365)),
    )
    return {"dod": dod, "wow": wow, "mom": mom, "yoy": yoy}


def metric_row(df: pd.DataFrame, label_prefix: str) -> None:
    daily = total_by_day(df)
    total_trips = df["trips"].sum()
    avg_daily = daily["trips"].mean() if not daily.empty else 0
    peak_day = daily.loc[daily["trips"].idxmax()] if not daily.empty else None
    rainy_lift = weather_lift(df)

    col1, col2, col3 = st.columns(3)
    col1.metric(f"{label_prefix} trips", fmt_int(total_trips))
    col2.metric("Avg daily trips", fmt_int(avg_daily))
    if peak_day is not None:
        col3.metric(
            "Peak day",
            fmt_int(peak_day["trips"]),
            peak_day["pickup_date"].strftime("%b %d, %Y"),
        )
    else:
        col3.metric("Rainy-day lift", fmt_pct(rainy_lift))


def weather_lift(df: pd.DataFrame) -> float | None:
    if "has_precipitation" not in df.columns:
        return None
    daily = (
        df.groupby(["pickup_date", "has_precipitation"], as_index=False)["trips"]
        .sum()
        .dropna()
    )
    if daily["has_precipitation"].nunique() < 2:
        return None
    means = daily.groupby("has_precipitation")["trips"].mean()
    dry = means.get(False)
    wet = means.get(True)
    if dry in (None, 0) or pd.isna(dry) or pd.isna(wet):
        return None
    return wet / dry - 1


def build_event_calendar(daily_context: pd.DataFrame) -> pd.DataFrame:
    built_in: list[pd.DataFrame] = []
    holiday_rows = daily_context[daily_context["holiday_name"].notna()].copy()
    if not holiday_rows.empty:
        built_in.append(
            pd.DataFrame(
                {
                    "date": holiday_rows["pickup_date"],
                    "event_name": holiday_rows["holiday_name"],
                    "event_category": "Federal holiday",
                    "venue": pd.NA,
                    "borough": pd.NA,
                }
            )
        )

    nyc_rows = daily_context[
        daily_context["nyc_event"].notna()
        & ~daily_context["is_federal_holiday"].fillna(False)
    ].copy()
    if not nyc_rows.empty:
        built_in.append(
            pd.DataFrame(
                {
                    "date": nyc_rows["pickup_date"],
                    "event_name": nyc_rows["nyc_event"],
                    "event_category": nyc_rows["nyc_event"].map(infer_nyc_event_category),
                    "venue": pd.NA,
                    "borough": pd.NA,
                }
            )
        )

    _cal_path = PROCESSED / "context" / "nyc_event_calendar.csv"
    _cal_mtime = _cal_path.stat().st_mtime if _cal_path.exists() else 0.0
    custom = load_custom_events(_cal_mtime).rename(columns={"date": "pickup_date"})
    if not custom.empty:
        custom = custom.rename(columns={"pickup_date": "date"})
        built_in.append(custom)

    if not built_in:
        return pd.DataFrame(
            columns=["date", "event_name", "event_category", "venue", "borough"]
        )
    events = pd.concat(built_in, ignore_index=True)
    events["date"] = pd.to_datetime(events["date"])
    events["event_category"] = events["event_category"].map(normalize_event_category)
    events["_event_name_key"] = events["event_name"].astype(str).str.lower().str.strip()
    events = events.drop_duplicates(
        subset=["date", "_event_name_key", "event_category"], keep="last"
    ).drop(columns="_event_name_key")
    events = events.drop_duplicates(
        subset=["date", "event_name", "event_category", "venue", "borough"]
    )
    return events.sort_values(["date", "event_category", "event_name"])


NYC_BOROUGHS = {"Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"}


def event_impact_table(
    daily_context: pd.DataFrame,
    events: pd.DataFrame,
    baseline_days: int,
    borough_daily: pd.DataFrame | None = None,
    exclusion_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if events.empty or daily_context.empty:
        return pd.DataFrame()

    daily_all = daily_context[["pickup_date", "weekday_num", "trips"]].copy()
    daily_all["pickup_date"] = pd.to_datetime(daily_all["pickup_date"])

    # Build a per-borough lookup so each event can use its own borough's trips
    boro_lookup: dict[str, pd.DataFrame] = {}
    if borough_daily is not None and "borough" in borough_daily.columns:
        for boro_name, grp in borough_daily.groupby("borough"):
            if str(boro_name) in NYC_BOROUGHS:
                boro_df = grp[["pickup_date", "weekday_num", "trips"]].copy()
                boro_df["pickup_date"] = pd.to_datetime(boro_df["pickup_date"])
                boro_lookup[str(boro_name)] = boro_df

    exclusion_calendar = exclusion_events if exclusion_events is not None else events
    event_dates = set(pd.to_datetime(exclusion_calendar["date"]).dt.normalize())
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        event_date = pd.Timestamp(event.date).normalize()
        event_borough = str(event.borough) if pd.notna(event.borough) else ""
        use_did = event_borough in boro_lookup
        # Use borough-specific series when available; fall back to all-NYC
        daily = boro_lookup.get(event_borough, daily_all)

        event_day = daily[daily["pickup_date"].dt.normalize().eq(event_date)]
        if event_day.empty:
            continue
        weekday_num = int(event_day["weekday_num"].iloc[0])
        start = event_date - pd.Timedelta(days=baseline_days)
        end = event_date + pd.Timedelta(days=baseline_days)
        baseline = daily[
            daily["pickup_date"].between(start, end)
            & daily["weekday_num"].eq(weekday_num)
            & ~daily["pickup_date"].dt.normalize().isin(event_dates)
        ]
        baseline_trips = baseline["trips"].mean() if not baseline.empty else pd.NA
        event_trips = event_day["trips"].sum()
        raw_lift = (
            event_trips / baseline_trips - 1
            if pd.notna(baseline_trips) and baseline_trips > 0
            else pd.NA
        )

        # Diff-in-diff: remove the citywide lift that happened on the same day
        # Control = all other boroughs (daily_all minus the event borough)
        ctrl_lift: object = pd.NA
        did_lift: object = pd.NA
        if use_did:
            baseline_dates_set = set(baseline["pickup_date"].dt.normalize())
            total_on_baseline = daily_all[
                daily_all["pickup_date"].dt.normalize().isin(baseline_dates_set)
            ]
            treat_on_baseline = daily[
                daily["pickup_date"].dt.normalize().isin(baseline_dates_set)
            ]
            merged_bl = total_on_baseline[["pickup_date", "trips"]].merge(
                treat_on_baseline[["pickup_date", "trips"]].rename(columns={"trips": "treat_trips"}),
                on="pickup_date",
                how="inner",
            )
            merged_bl["ctrl_trips"] = merged_bl["trips"] - merged_bl["treat_trips"]
            ctrl_baseline = merged_bl["ctrl_trips"].mean() if not merged_bl.empty else pd.NA

            all_event_day = daily_all[daily_all["pickup_date"].dt.normalize().eq(event_date)]
            total_event = all_event_day["trips"].sum() if not all_event_day.empty else pd.NA
            ctrl_event = (total_event - event_trips) if pd.notna(total_event) else pd.NA

            ctrl_lift = (
                ctrl_event / ctrl_baseline - 1
                if pd.notna(ctrl_baseline) and ctrl_baseline > 0 and pd.notna(ctrl_event)
                else pd.NA
            )
            did_lift = (
                raw_lift - ctrl_lift
                if pd.notna(raw_lift) and pd.notna(ctrl_lift)
                else pd.NA
            )

        rows.append(
            {
                "Date": event_date,
                "Event": event.event_name,
                "Category": event.event_category,
                "Venue": event.venue,
                "Borough": event.borough,
                "Trips": event_trips,
                "Baseline Trips": baseline_trips,
                "Lift": raw_lift,
                "Control Lift": ctrl_lift,
                "DiD Lift": did_lift,
                "Baseline Days": len(baseline),
            }
        )
    return pd.DataFrame(rows).sort_values(["Date", "Category", "Event"])


def line_trend(
    df: pd.DataFrame,
    title: str,
    granularity: str = "Daily",
    future_forecast=None,
) -> go.Figure:
    fig = go.Figure()
    trend = pd.DataFrame()
    category_order: list[str] = []
    if granularity == "Daily":
        trend = total_by_day(df)
        fig.add_trace(
            go.Scatter(
                x=trend["pickup_date"],
                y=trend["trips"],
                mode="lines",
                name="Daily trips",
                line=dict(color=ACCENT_BLUE, width=2.2),
            )
        )
    else:
        trend = resample_to_granularity(df, granularity)
        category_order.extend(trend["period_label"].astype(str).tolist())
        fig.add_trace(
            go.Bar(
                x=trend["period_label"],
                y=trend["trips"],
                name=f"{granularity} trips",
                marker_color=ACCENT_BLUE,
            )
        )

    if future_forecast is not None and not future_forecast.empty:
        forecast_plot = resample_series_for_granularity(
            future_forecast,
            granularity,
            "trips_forecast",
        )
        if granularity == "Daily":
            if not trend.empty:
                anchor = pd.DataFrame(
                    {
                        "pickup_date": [trend["pickup_date"].max()],
                        "trips_forecast": [trend.sort_values("pickup_date")["trips"].iloc[-1]],
                    }
                )
                forecast_plot = pd.concat([anchor, forecast_plot], ignore_index=True)
            fig.add_trace(
                go.Scatter(
                    x=forecast_plot["pickup_date"],
                    y=forecast_plot["trips_forecast"],
                    mode="lines",
                    name="Forecast",
                    line=dict(color=NEUTRAL_GRAY, width=2.5, dash="dash"),
                )
            )
        else:
            category_order.extend(
                forecast_plot["period_label"].astype(str).tolist()
            )
            fig.add_trace(
                go.Bar(
                    x=forecast_plot["period_label"],
                    y=forecast_plot["trips_forecast"],
                    name="Forecast",
                    marker_color=NEUTRAL_GRAY,
                    opacity=0.6,
                )
            )
    if granularity != "Daily":
        fig.update_layout(barmode="group")
        category_order = list(dict.fromkeys(category_order))
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=category_order,
        )

    fig.update_layout(
        title=title,
        height=400,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Trips",
        xaxis_title=None,
    )
    if (
        granularity == "Daily"
        and
        not trend.empty
        and trend["pickup_date"].min() <= SOURCE_CHANGE_DATE <= trend["pickup_date"].max()
    ):
        fig.add_vline(
            x=SOURCE_CHANGE_DATE,
            line_width=1.5,
            line_dash="dash",
            line_color=BLUE,
        )
        fig.add_annotation(
            x=SOURCE_CHANGE_DATE,
            y=1,
            xref="x",
            yref="paper",
            text="Dataset changes to TLC HVFHV Uber records",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color=BLUE, size=12),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=BLUE,
            borderwidth=1,
        )
    return fig


def bar_by_group(df: pd.DataFrame, group_col: str, title: str, top_n: int = 12) -> go.Figure:
    grouped = (
        df.groupby(group_col, as_index=False)["trips"]
        .sum()
        .sort_values("trips", ascending=False)
        .head(top_n)
    )
    fig = px.bar(grouped, x="trips", y=group_col, orientation="h", title=title)
    fig.update_traces(marker_color=ACCENT_BLUE)
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis_title="Trips",
        yaxis_title=None,
        yaxis=dict(categoryorder="total ascending"),
    )
    return fig


def comparison_start(latest_date: pd.Timestamp, period_label: str) -> pd.Timestamp:
    if period_label == "Week":
        return latest_date - pd.Timedelta(days=latest_date.weekday())
    if period_label == "Month":
        return latest_date.replace(day=1)
    return latest_date.replace(month=1, day=1)


def build_period_comparison(
    df: pd.DataFrame,
    period_label: str,
    comparison_mode: str,
) -> tuple[pd.DataFrame, dict]:
    daily = total_by_day(df)
    if daily.empty:
        return pd.DataFrame(), {}

    series = (
        daily.assign(pickup_date=pd.to_datetime(daily["pickup_date"]))
        .set_index("pickup_date")["trips"]
        .sort_index()
        .asfreq("D", fill_value=0)
    )
    current_end = series.index.max().normalize()
    current_start = comparison_start(current_end, period_label)
    days = (current_end - current_start).days + 1

    if comparison_mode == "Previous period":
        if period_label == "Month":
            comparison_start_date = current_start - pd.DateOffset(months=1)
            comparison_end = comparison_start_date + pd.Timedelta(days=days - 1)
        elif period_label == "Year":
            comparison_start_date = current_start - pd.DateOffset(years=1)
            comparison_end = comparison_start_date + pd.Timedelta(days=days - 1)
        else:
            comparison_start_date = current_start - pd.Timedelta(days=7)
            comparison_end = comparison_start_date + pd.Timedelta(days=days - 1)
    else:
        comparison_start_date = current_start - pd.DateOffset(years=1)
        comparison_end = current_end - pd.DateOffset(years=1)

    current_axis = pd.date_range(current_start, current_end, freq="D")

    def window_frame(start: pd.Timestamp, end: pd.Timestamp, label: str) -> pd.DataFrame:
        idx = pd.date_range(start, end, freq="D")
        values = series.reindex(idx, fill_value=0)
        return pd.DataFrame(
            {
                "day": range(1, len(values) + 1),
                "date": values.index,
                "plot_date": current_axis[: len(values)],
                "period": label,
                "trips": values.values,
            }
        )

    current_label = "Current"
    comparison_label = comparison_mode
    chart_df = pd.concat(
        [
            window_frame(current_start, current_end, current_label),
            window_frame(comparison_start_date, comparison_end, comparison_label),
        ],
        ignore_index=True,
    )
    current_total = series.reindex(pd.date_range(current_start, current_end, freq="D"), fill_value=0).sum()
    comparison_total = series.reindex(
        pd.date_range(comparison_start_date, comparison_end, freq="D"),
        fill_value=0,
    ).sum()
    delta = current_total / comparison_total - 1 if comparison_total else None
    meta = {
        "current_label": date_window_label(current_start, current_end),
        "comparison_label": date_window_label(comparison_start_date, comparison_end),
        "current_total": current_total,
        "comparison_total": comparison_total,
        "delta": delta,
    }
    return chart_df, meta


def period_comparison_chart(chart_df: pd.DataFrame, period_label: str) -> go.Figure:
    fig = go.Figure()
    for label, color, width in [
        ("Current", ACCENT_BLUE, 3),
        ("Previous period", NEUTRAL_GRAY, 2.5),
        ("Same period last year", INDIGO, 2.5),
    ]:
        period_df = chart_df[chart_df["period"].eq(label)]
        if period_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=period_df["plot_date"],
                y=period_df["trips"],
                mode="lines",
                name=label,
                line=dict(color=color, width=width, dash="solid"),
                customdata=period_df["date"].dt.strftime("%b %d, %Y"),
                hovertemplate="%{x|%b %d}<br>Actual date: %{customdata}<br>Trips: %{y:,.0f}<extra>%{fullData.name}</extra>",
            )
        )
    tick_format = "%b %d" if period_label != "Year" else "%b"
    fig.update_layout(
        title=f"{period_label} Comparison",
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date in current period",
        yaxis_title="Trips",
        xaxis=dict(tickformat=tick_format),
    )
    return fig


def month_growth_table(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    group_keys = ["pickup_month"] + ([group_col] if group_col else [])
    out = df.groupby(group_keys, as_index=False)["trips"].sum()
    sort_keys = [group_col, "pickup_month"] if group_col else ["pickup_month"]
    out = out.sort_values(sort_keys)
    if group_col:
        out["mom_growth"] = out.groupby(group_col)["trips"].pct_change()
    else:
        out["mom_growth"] = out["trips"].pct_change()
    return out


GROWTH_PERIODS = {
    "Last month": 1,
    "Last 6 months": 6,
    "Last 1 year": 12,
    "Last 3 years": 36,
    "Last 5 years": 60,
}


def date_window_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start.strftime('%b %d, %Y')} - {end.strftime('%b %d, %Y')}"


def missing_dates(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    if df.empty:
        return pd.date_range(start, end, freq="D")
    expected = pd.date_range(start, end, freq="D")
    present = pd.to_datetime(df["pickup_date"]).dropna().dt.normalize().unique()
    return expected.difference(pd.DatetimeIndex(present))


def shade_color(
    value: float,
    max_abs_value: float,
    positive_rgb: tuple[int, int, int] = (46, 125, 50),
    negative_rgb: tuple[int, int, int] = (196, 61, 61),
) -> str:
    if max_abs_value <= 0 or pd.isna(value):
        return NEUTRAL_GRAY
    target = positive_rgb if value > 0 else negative_rgb if value < 0 else (156, 163, 175)
    intensity = min(abs(value) / max_abs_value, 1)
    mix = 0.35 + 0.65 * intensity
    base = (229, 231, 235)
    rgb = [round(base[i] * (1 - mix) + target[i] * mix) for i in range(3)]
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def shade_rgba(value: float, max_abs_value: float, alpha: int = 190) -> list[int]:
    if max_abs_value <= 0 or pd.isna(value):
        return [156, 163, 175, alpha]
    target = (46, 125, 50) if value > 0 else (196, 61, 61) if value < 0 else (156, 163, 175)
    intensity = min(abs(value) / max_abs_value, 1)
    mix = 0.35 + 0.65 * intensity
    base = (229, 231, 235)
    rgb = [round(base[i] * (1 - mix) + target[i] * mix) for i in range(3)]
    return [rgb[0], rgb[1], rgb[2], alpha]


def growth_opportunities(
    df: pd.DataFrame,
    group_col: str,
    group_label: str,
    lookback_months: int,
    top_n: int | None = 15,
    min_volume_quantile: float | None = 0.5,
    rank_by: str = "largest_gains",
    require_complete_daily_coverage: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str | None]:
    if df.empty:
        empty = pd.DataFrame(columns=[group_label, "Current Trips", "Prior Trips", "Trip Delta", "Growth"])
        return empty, empty, "No matching data", "No rows match the selected filters."

    latest_date = pd.to_datetime(df["pickup_date"]).max().normalize()
    current_start = latest_date - pd.DateOffset(months=lookback_months) + pd.Timedelta(days=1)
    prior_end = current_start - pd.Timedelta(days=1)
    prior_start = current_start - pd.DateOffset(months=lookback_months)

    meta = (
        f"Comparing {date_window_label(current_start, latest_date)} "
        f"vs {date_window_label(prior_start, prior_end)}"
    )
    coverage_warning = None
    if require_complete_daily_coverage:
        missing_comparison_dates = missing_dates(df, prior_start, latest_date)
        if len(missing_comparison_dates):
            empty = pd.DataFrame(
                columns=[group_label, "Current Trips", "Prior Trips", "Trip Delta", "Growth", "Direction"]
            )
            coverage_warning = (
                f"Zone detail is missing for {fmt_int(len(missing_comparison_dates))} days "
                "inside this comparison window, so zone gains/losses are hidden to avoid treating missing data as zero trips."
            )
            return empty, empty.drop(columns=["Direction"]), meta, coverage_warning

    current = period_filter(df, current_start, latest_date)
    previous = period_filter(df, prior_start, prior_end)
    group_cols = [group_col]
    if group_col == "zone" and "borough" in df.columns:
        group_cols = ["zone", "borough"]

    current_sum = (
        current.groupby(group_cols, dropna=False, as_index=False)["trips"]
        .sum()
        .rename(columns={"trips": "current_trips"})
    )
    prior_sum = (
        previous.groupby(group_cols, dropna=False, as_index=False)["trips"]
        .sum()
        .rename(columns={"trips": "prior_trips"})
    )
    out = current_sum.merge(prior_sum, on=group_cols, how="left")
    out["prior_trips"] = out["prior_trips"].fillna(0)
    out["trip_delta"] = out["current_trips"] - out["prior_trips"]
    out["growth"] = pd.NA
    comparable = out["prior_trips"] > 0
    out.loc[comparable, "growth"] = (
        out.loc[comparable, "current_trips"] / out.loc[comparable, "prior_trips"] - 1
    )

    if out.empty:
        display = pd.DataFrame(columns=[group_label, "Current Trips", "Prior Trips", "Trip Delta", "Growth"])
        return display, display, "No matching data", "No rows match the selected filters."

    minimum_volume = 0
    if min_volume_quantile is not None:
        minimum_volume = out["current_trips"].quantile(min_volume_quantile)
    ranked = out[out["current_trips"] >= minimum_volume].copy()
    if rank_by == "growth":
        ranked = ranked[ranked["prior_trips"] > 0].copy()
    if ranked.empty:
        ranked = out.copy()
    if rank_by == "largest_losses":
        ranked = ranked.sort_values(["trip_delta", "growth"], ascending=True)
    elif rank_by == "largest_gains":
        ranked = ranked.sort_values(["trip_delta", "growth"], ascending=False)
    else:
        ranked = ranked.sort_values(["growth", "trip_delta"], ascending=False)
    if top_n is not None:
        ranked = ranked.head(top_n)

    if pd.to_datetime(df["pickup_date"]).min().normalize() > prior_start:
        coverage_warning = "The selected date range does not include the full prior comparison period."

    rename_map = {
        group_col: group_label,
        "current_trips": "Current Trips",
        "prior_trips": "Prior Trips",
        "trip_delta": "Trip Delta",
        "growth": "Growth",
    }
    if "borough" in ranked.columns:
        rename_map["borough"] = "Borough"
    raw = ranked.rename(columns=rename_map)
    raw["Direction"] = "Flat"
    raw.loc[pd.to_numeric(raw["Trip Delta"], errors="coerce") > 0, "Direction"] = "Positive"
    raw.loc[pd.to_numeric(raw["Trip Delta"], errors="coerce") < 0, "Direction"] = "Negative"

    display = raw.copy()
    for column in ["Current Trips", "Prior Trips", "Trip Delta"]:
        display[column] = display[column].map(fmt_int)
    display["Growth"] = display["Growth"].map(fmt_pct)

    columns = [group_label]
    if "Borough" in display.columns and group_label != "Borough":
        columns.append("Borough")
    columns.extend(["Current Trips", "Prior Trips", "Trip Delta", "Growth"])
    return raw[columns + ["Direction"]], display[columns], meta, coverage_warning


def selected_group_values(
    label: str,
    options: list[str],
    dataset_name: str,
) -> list[str]:
    all_label = f"All {label.lower()}s"
    presets = [all_label]

    if label == "Borough":
        preset_values = {
            all_label: options,
            "Core NYC boroughs": [
                value
                for value in ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
                if value in options
            ],
            "Manhattan only": ["Manhattan"] if "Manhattan" in options else [],
            "Outer boroughs": [
                value
                for value in ["Bronx", "Brooklyn", "Queens", "Staten Island"]
                if value in options
            ],
        }
        presets.extend(["Core NYC boroughs", "Manhattan only", "Outer boroughs"])
    else:
        preset_values = {
            all_label: options,
            "Top 3 by volume": options[:3],
            "Top 5 by volume": options[:5],
        }
        presets.extend(["Top 3 by volume", "Top 5 by volume"])

    presets.append("Custom")
    choice = st.sidebar.selectbox(label, presets, index=0, key=f"{dataset_name}_{label}_preset")
    if choice == "Custom":
        selected = st.sidebar.multiselect(
            f"Choose {label.lower()}s",
            options,
            default=options,
            key=f"{dataset_name}_{label}_custom",
            placeholder=f"Select {label.lower()}s",
        )
    else:
        selected = preset_values.get(choice, options)

    st.sidebar.caption(f"{len(selected)} of {len(options)} selected")
    return selected


tables = load_all_tables()

st.title("Uber NYC Performance Dashboard")
st.markdown(
    "<div class='section-note'>Historical demand cockpit for the NYC GM team, using FiveThirtyEight FOIL Uber data plus weather, calendar, and TLC zone references.</div>",
    unsafe_allow_html=True,
)

st.sidebar.header("Filters")

source_options = []
if "unified_borough" in tables:
    source_options.append("All available years (2015+)")
source_options.append("2015 Uber zone pickups")
if "modern_borough" in tables:
    source_options.append("Modern Uber HVFHV pickups")

source = st.sidebar.selectbox(
    "Demand dataset",
    source_options,
    index=0,
)

if source.startswith("All"):
    active = tables["unified_borough"]
    active = active[active["pickup_date"] >= ANALYSIS_START_DATE].copy()
    group_label = "borough"
    group_name = "Borough"
elif source.startswith("2014"):
    active = tables["uber_2014_daily"]
    group_label = "base_name"
    group_name = "Base"
elif source.startswith("2015"):
    active = tables["uber_2015_borough"]
    group_label = "borough"
    group_name = "Borough"
else:
    active = tables["modern_borough"]
    group_label = "borough"
    group_name = "Borough"

min_date = active["pickup_date"].min().date()
max_date = active["pickup_date"].max().date()

# Clear saved preset whenever the source changes
_PRESET_KEY = "date_range_preset"
if st.session_state.get("_active_source") != source:
    st.session_state.pop(_PRESET_KEY, None)
    st.session_state["_active_source"] = source

import datetime as _dt

_today = pd.Timestamp.today().date()
_range_end = min(max_date, _today)
_presets = {
    "1D": _range_end,
    "7D": _range_end - _dt.timedelta(days=6),
    "30D": _range_end - _dt.timedelta(days=29),
    "90D": _range_end - _dt.timedelta(days=89),
    "MTD": _dt.date(_range_end.year, _range_end.month, 1),
    "YTD": _dt.date(_range_end.year, 1, 1),
    "1Y": _range_end - _dt.timedelta(days=364),
    "5Y": _range_end - _dt.timedelta(days=(365 * 5) - 1),
    "All": min_date,
}
_preset_descriptions = {
    "1D": "latest loaded day only",
    "7D": "rolling 7 days",
    "30D": "rolling 30 days",
    "90D": "rolling 90 days",
    "MTD": "calendar month to date",
    "YTD": "calendar year to date",
    "1Y": "rolling 365 days",
    "5Y": "rolling 5 years",
    "All": "full loaded dataset",
}

_source_change = SOURCE_CHANGE_DATE.date()
if min_date < _source_change <= max_date:
    _presets["Post-2019"] = _source_change
    _preset_descriptions["Post-2019"] = "consistent HVFHV source period"

st.sidebar.markdown("**Date range**")
_preset_labels = list(_presets.keys())
_default_preset = "MTD"
_saved_preset = st.session_state.get(_PRESET_KEY, _default_preset)
if _saved_preset not in _preset_labels:
    _saved_preset = _default_preset
_selected_preset = st.sidebar.radio(
    "Date range",
    _preset_labels,
    index=_preset_labels.index(_saved_preset),
    key=_PRESET_KEY,
    horizontal=True,
    label_visibility="collapsed",
)

start_date = pd.Timestamp(max(min_date, _presets[_selected_preset]))
end_date = pd.Timestamp(_range_end)
selected_range_label = date_window_label(start_date, end_date)
selected_preset_detail = _preset_descriptions.get(_selected_preset, "selected range")
st.sidebar.caption(f"{selected_range_label} | {_selected_preset}: {selected_preset_detail}")

group_volume = (
    active.groupby(group_label, as_index=False)["trips"]
    .sum()
    .sort_values("trips", ascending=False)
)
groups = group_volume[group_label].dropna().astype(str).tolist()
selected_groups = selected_group_values(
    group_name,
    groups,
    source.lower().replace(" ", "_"),
)

filtered = period_filter(active, start_date, end_date)
filtered = filtered[filtered[group_label].astype(str).isin(selected_groups)]
forecast_input = active[active[group_label].astype(str).isin(selected_groups)].copy()

if filtered.empty:
    st.warning("No rows match the selected filters.")
    st.stop()

st.markdown(
    (
        "<div class='section-note'>"
        f"<strong>Data cutoff:</strong> {pd.Timestamp(max_date).strftime('%b %d, %Y')} "
        f"| <strong>Selected range:</strong> {selected_range_label} "
        f"| <strong>Preset:</strong> {_selected_preset} ({selected_preset_detail})"
        "</div>"
    ),
    unsafe_allow_html=True,
)

tab_overview, tab_timing, tab_geo, tab_context = st.tabs(
    ["Overview", "Timing", "Geography", "Context"]
)

with tab_overview:
    metric_row(filtered, "Selected")

    # --- Static growth stats row ---
    gs = growth_stats(active)

    def _fmt_growth(v):
        if v is None:
            return "N/A", None
        color = POSITIVE_GREEN if v >= 0 else NEGATIVE_RED
        arrow = "+" if v >= 0 else "-"
        return f"{arrow} {abs(v):.1%}", color

    g_dod, c_dod = _fmt_growth(gs["dod"])
    g_wow, c_wow = _fmt_growth(gs["wow"])
    g_mom, c_mom = _fmt_growth(gs["mom"])
    g_yoy, c_yoy = _fmt_growth(gs["yoy"])

    gcol1, gcol2, gcol3, gcol4 = st.columns(4)
    gcol1.metric("DoD", g_dod)
    gcol2.metric("WoW", g_wow)
    gcol3.metric("MoM", g_mom)
    gcol4.metric("YoY", g_yoy)

    st.divider()

    granularity = st.segmented_control(
        "Granularity",
        options=["Daily", "Weekly", "Monthly", "Yearly"],
        default="Daily",
        key="overview_granularity",
        label_visibility="collapsed",
    )

    horizon_col, model_col = st.columns(2)
    with horizon_col:
        forecast_months = st.select_slider(
            "Forecast horizon",
            options=list(range(1, 13)),
            value=1,
            format_func=lambda months: "1 month" if months == 1 else f"{months} months",
        )
    with model_col:
        forecast_model = st.selectbox(
            "Forecast model",
            FORECAST_MODELS,
            index=0,
        )

    # --- Daily forecast (trained on selected data from 2021 onward) ---
    _, _, _future_forecast = build_forecast(
        forecast_input,
        forecast_months,
        forecast_model,
        context=tables["context"],
    )

    left, right = st.columns((2, 1))
    with left:
        st.plotly_chart(
            line_trend(
                filtered,
                f"{granularity} Demand",
                granularity,
                future_forecast=_future_forecast,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        if source.startswith("All") and start_date <= SOURCE_CHANGE_DATE <= end_date:
            with st.expander("What does the Feb 2019 dataset change mean?"):
                st.markdown(
                    """
**Before Feb 2019 — FHV base dispatch records (FOIL data)**

NYC's Taxi & Limousine Commission required each licensed dispatch *base* to report trip dispatches monthly. Uber operated under several bases (e.g. Unter LLC, Hinter LLC). FiveThirtyEight obtained these records via a Freedom of Information Law (FOIL) request in 2015, and similar filings continued through early 2019. The data captures *dispatches* — one row per base per trip — and has limited geographic detail (no pickup/dropoff zones in the early years).

**After Feb 2019 — TLC High Volume FHV (HVFHV) records**

In 2018, NYC passed Local Law 149, requiring companies with ≥10,000 trips/day to register as High Volume FHV operators and submit detailed trip-level records directly to the TLC. Uber registered under license **HV0003**. Starting with February 2019 data, the TLC publishes monthly Parquet files with one row per *completed trip*, including pickup/dropoff taxi zone, trip distance, duration, fare, tips, and driver pay.

**Why this matters for the dashboard**

| | Pre-2019 FHV | Post-2019 HVFHV |
|---|---|---|
| Unit | Dispatch event | Completed trip |
| Geography | Base location only | Pickup + dropoff zone |
| Financials | None | Fare, tips, driver pay |
| Coverage | Uber-associated bases | All Uber trips (HV0003) |

The two sources count differently, so there is a **level discontinuity** at the dashed line — a jump or drop in the trend chart may reflect the source change, not a real demand shift. Use the **"Post-2019"** preset in the sidebar to restrict to the consistent source.
                    """
                )
    with right:
        st.plotly_chart(
            bar_by_group(filtered, group_label, f"Trips by {group_name}"),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.divider()
    st.subheader("Period Comparison")
    period_col, mode_col = st.columns((1, 1.4))
    with period_col:
        comparison_period = st.segmented_control(
            "Period",
            options=["Week", "Month", "Year"],
            default="Month",
            key="overview_comparison_period",
        )
    with mode_col:
        comparison_mode = st.radio(
            "Compare against",
            ["Previous period", "Same period last year"],
            horizontal=True,
            key="overview_comparison_mode",
        )
    st.caption(
        "Calendar periods are partial-to-date: Month compares month-to-date "
        "against the same day count in the prior month, e.g. Nov 1-16 vs Oct 1-16."
    )

    comparison_df, comparison_meta = build_period_comparison(
        forecast_input,
        comparison_period,
        comparison_mode,
    )
    if comparison_df.empty:
        st.info("No data is available for this comparison.")
    else:
        total_col, comparison_col, delta_col = st.columns(3)
        total_col.metric(
            f"Current {comparison_period.lower()}",
            fmt_int(comparison_meta["current_total"]),
        )
        comparison_col.metric(
            comparison_mode,
            fmt_int(comparison_meta["comparison_total"]),
        )
        delta_col.metric("Change", fmt_pct(comparison_meta["delta"]))
        st.caption(
            f"Current: {comparison_meta['current_label']} | "
            f"{comparison_mode}: {comparison_meta['comparison_label']}"
        )
        st.plotly_chart(
            period_comparison_chart(comparison_df, comparison_period),
            use_container_width=True,
            config={"displayModeBar": False},
        )

with tab_timing:
    st.subheader("When Demand Moves")
    if source.startswith("All"):
        hourly = period_filter(tables["unified_hourly"], start_date, end_date)
        hourly["weekday"] = hourly["pickup_date"].dt.day_name()
        hourly["weekday_num"] = hourly["pickup_date"].dt.weekday
        hourly_totals = hourly.groupby("hour", as_index=False)["trips"].sum()
        fig_hour = px.bar(
            hourly_totals,
            x="hour",
            y="trips",
            title="Trips by Hour of Day",
        )
        fig_hour.update_traces(marker_color=ACCENT_BLUE)
        fig_hour.update_layout(
            height=330,
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis_title="Hour",
            yaxis_title="Trips",
        )

        heat = hourly.groupby(["weekday_num", "weekday", "hour"], as_index=False)[
            "trips"
        ].sum()
        weekday_order = (
            heat[["weekday_num", "weekday"]]
            .drop_duplicates()
            .sort_values("weekday_num")["weekday"]
            .tolist()
        )
        pivot = heat.pivot_table(
            index="weekday",
            columns="hour",
            values="trips",
            aggfunc="sum",
            fill_value=0,
        ).reindex(weekday_order)
        fig_heat = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale=HEATMAP_SCALE,
            title="Weekday x Hour Demand Heatmap",
        )
        fig_heat.update_layout(
            height=390,
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis_title="Hour",
            yaxis_title=None,
        )

        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_hour, use_container_width=True, config={"displayModeBar": False})
        col2.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
    elif source.startswith("2014"):
        hourly = period_filter(tables["uber_2014_hourly"], start_date, end_date)
        hourly = hourly[hourly["base_name"].isin(selected_groups)]
    elif source.startswith("Modern"):
        hourly = period_filter(tables["modern_hourly"], start_date, end_date)
        hourly_totals = hourly.groupby("hour", as_index=False)["trips"].sum()
        fig_hour = px.bar(
            hourly_totals,
            x="hour",
            y="trips",
            title="Trips by Hour of Day",
        )
        fig_hour.update_traces(marker_color=ACCENT_BLUE)
        fig_hour.update_layout(
            height=330,
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis_title="Hour",
            yaxis_title="Trips",
        )

        heat = tables["modern_weekday_hour"].copy()
        heat = heat.groupby(["weekday_num", "weekday", "hour"], as_index=False)[
            "trips"
        ].sum()
        weekday_order = (
            heat[["weekday_num", "weekday"]]
            .drop_duplicates()
            .sort_values("weekday_num")["weekday"]
            .tolist()
        )
        pivot = heat.pivot_table(
            index="weekday",
            columns="hour",
            values="trips",
            aggfunc="sum",
            fill_value=0,
        ).reindex(weekday_order)
        fig_heat = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale=HEATMAP_SCALE,
            title="Weekday x Hour Demand Heatmap",
        )
        fig_heat.update_layout(
            height=390,
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis_title="Hour",
            yaxis_title=None,
        )

        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_hour, use_container_width=True, config={"displayModeBar": False})
        col2.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
    else:
        daily = filtered.groupby(["weekday_num", "weekday"], as_index=False)["trips"].sum()
        daily = daily.sort_values("weekday_num")
        fig_weekday = px.bar(
            daily,
            x="weekday",
            y="trips",
            title="Trips by Day of Week",
        )
        fig_weekday.update_traces(marker_color=ACCENT_BLUE)
        fig_weekday.update_layout(
            height=330,
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis_title=None,
            yaxis_title="Trips",
        )
        st.plotly_chart(fig_weekday, use_container_width=True, config={"displayModeBar": False})
        st.info("The 2015 FOIL table has pickup zones but not pickup hour, so the hourly view is available for 2014 and modern TLC HVFHV data.")
        hourly = None

    if source.startswith("2014"):
        hourly_totals = hourly.groupby("hour", as_index=False)["trips"].sum()
        fig_hour = px.bar(
            hourly_totals,
            x="hour",
            y="trips",
            title="Trips by Hour of Day",
        )
        fig_hour.update_traces(marker_color=ACCENT_BLUE)
        fig_hour.update_layout(
            height=330,
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis_title="Hour",
            yaxis_title="Trips",
        )

        heat = tables["uber_2014_weekday_hour"]
        heat = heat[heat["base_name"].isin(selected_groups)]
        heat = heat.groupby(["weekday_num", "weekday", "hour"], as_index=False)[
            "trips"
        ].sum()
        weekday_order = (
            heat[["weekday_num", "weekday"]]
            .drop_duplicates()
            .sort_values("weekday_num")["weekday"]
            .tolist()
        )
        pivot = heat.pivot_table(
            index="weekday",
            columns="hour",
            values="trips",
            aggfunc="sum",
            fill_value=0,
        ).reindex(weekday_order)
        fig_heat = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale=HEATMAP_SCALE,
            title="Weekday x Hour Demand Heatmap",
        )
        fig_heat.update_layout(
            height=390,
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis_title="Hour",
            yaxis_title=None,
        )

        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_hour, use_container_width=True, config={"displayModeBar": False})
        col2.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

with tab_geo:
    st.subheader("Where Demand Comes From")
    if source.startswith("2014"):
        trips = load_trip_sample()
        trips = trips[
            (trips["pickup_datetime"] >= start_date)
            & (trips["pickup_datetime"] <= end_date + pd.Timedelta(days=1))
            & (trips["base_name"].isin(selected_groups))
        ]
        st.markdown(
            f"<div class='muted'>Showing a deterministic sample of up to {fmt_int(len(trips))} pickup points from the 2014 GPS data.</div>",
            unsafe_allow_html=True,
        )
        layer = pdk.Layer(
            "HexagonLayer",
            data=trips,
            get_position="[lon, lat]",
            radius=180,
            elevation_scale=25,
            elevation_range=[0, 900],
            pickable=True,
            extruded=True,
            coverage=0.85,
        )
        view_state = pdk.ViewState(
            latitude=40.735,
            longitude=-73.96,
            zoom=10.2,
            pitch=42,
            bearing=-12,
        )
        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                initial_view_state=view_state,
                layers=[layer],
                tooltip={"text": "Pickup density"},
            ),
            use_container_width=True,
        )
    else:
        if source.startswith("All"):
            zone_table = "unified_zone"
        else:
            zone_table = "modern_zone" if source.startswith("Modern") else "uber_2015_zone"
        zone_df = period_filter(tables[zone_table], start_date, end_date)
        if selected_groups:
            zone_df = zone_df[zone_df["borough"].astype(str).isin(selected_groups)]
        col1, col2 = st.columns((1, 1))
        with col1:
            st.plotly_chart(
                bar_by_group(zone_df, "zone", "Top Pickup Zones", top_n=15),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with col2:
            borough = zone_df.groupby("borough", as_index=False)["trips"].sum()
            fig_tree = px.treemap(
                borough,
                path=["borough"],
                values="trips",
                title="Borough Share",
                color="trips",
                color_continuous_scale=HEATMAP_SCALE,
            )
            fig_tree.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig_tree, use_container_width=True, config={"displayModeBar": False})

        st.subheader("Growth Opportunities")
        growth_col, level_col, rank_col = st.columns((1.4, 0.9, 0.9))
        with growth_col:
            growth_period_label = st.radio(
                "Growth horizon",
                list(GROWTH_PERIODS.keys()),
                horizontal=True,
                key=f"{source}_growth_horizon",
            )
        with level_col:
            opportunity_level = st.selectbox(
                "Opportunity level",
                ["Borough", "Zone"],
                index=1,
                key=f"{source}_opportunity_level",
            )
        with rank_col:
            growth_rank_label = st.selectbox(
                "Show",
                ["Largest gains", "Largest losses"],
                index=0,
                key=f"{source}_growth_rank",
            )
        rank_by = "largest_gains" if growth_rank_label == "Largest gains" else "largest_losses"

        if opportunity_level == "Borough":
            opportunity_df = period_filter(active, start_date, end_date)
            opportunity_df = opportunity_df[
                opportunity_df[group_label].astype(str).isin(selected_groups)
            ]
            opportunity_group_col = group_label
            opportunity_group_label = group_name
            top_n = None
            min_volume_quantile = 0
            require_complete_daily_coverage = False
        else:
            opportunity_df = zone_df
            opportunity_group_col = "zone"
            opportunity_group_label = "Zone"
            top_n = 20
            min_volume_quantile = 0
            require_complete_daily_coverage = True
            missing_zone_days = missing_dates(opportunity_df, start_date, end_date)
            if len(missing_zone_days):
                st.info(
                    f"Zone-level detail is missing for {fmt_int(len(missing_zone_days))} "
                    "selected days because several large months were processed without zone aggregation."
                )

        raw_opportunities, display_opportunities, comparison_label, warning = growth_opportunities(
            opportunity_df,
            opportunity_group_col,
            opportunity_group_label,
            GROWTH_PERIODS[growth_period_label],
            top_n=top_n,
            min_volume_quantile=min_volume_quantile,
            rank_by=rank_by,
            require_complete_daily_coverage=require_complete_daily_coverage,
        )
        st.caption(comparison_label)
        if warning:
            st.warning(warning)

        chart_df = raw_opportunities.copy()
        chart_df["Growth"] = pd.to_numeric(chart_df["Growth"], errors="coerce")
        chart_df["Trip Delta"] = pd.to_numeric(chart_df["Trip Delta"], errors="coerce")
        chart_df = chart_df.dropna(subset=["Trip Delta"])
        if opportunity_level == "Borough":
            visible_boroughs = [
                borough for borough in BOROUGH_ORDER if borough in chart_df[opportunity_group_label].tolist()
            ]
            chart_df[opportunity_group_label] = pd.Categorical(
                chart_df[opportunity_group_label],
                categories=visible_boroughs,
                ordered=True,
            )
            chart_df = chart_df.sort_values(opportunity_group_label)
            chart_title = f"Borough Growth: {growth_period_label}"
        else:
            chart_df = chart_df.sort_values("Trip Delta", ascending=rank_by == "largest_gains")
            chart_title = f"Pickup Zone {'Gains' if rank_by == 'largest_gains' else 'Losses'}: {growth_period_label}"
        if not chart_df.empty:
            max_abs_delta = chart_df["Trip Delta"].abs().max()
            colors = [
                shade_color(value, max_abs_delta)
                for value in chart_df["Trip Delta"].tolist()
            ]
            hover_parts = [
                f"{label}<br>Trip delta: {fmt_int(delta)}<br>Growth: {fmt_pct(growth)}"
                for label, delta, growth in zip(
                    chart_df[opportunity_group_label].astype(str),
                    chart_df["Trip Delta"],
                    chart_df["Growth"],
                )
            ]
            if "Borough" in chart_df.columns:
                hover_parts = [
                    f"{label}<br>Borough: {borough}<br>Trip delta: {fmt_int(delta)}<br>Growth: {fmt_pct(growth)}"
                    for label, borough, delta, growth in zip(
                        chart_df[opportunity_group_label].astype(str),
                        chart_df["Borough"].astype(str),
                        chart_df["Trip Delta"],
                        chart_df["Growth"],
                    )
                ]
            fig_growth = go.Figure(
                go.Bar(
                    x=chart_df["Trip Delta"],
                    y=chart_df[opportunity_group_label].astype(str),
                    orientation="h",
                    marker=dict(color=colors),
                    hovertext=hover_parts,
                    hoverinfo="text",
                )
            )
            fig_growth.update_layout(title=chart_title)
            if opportunity_level == "Borough":
                fig_growth.update_yaxes(
                    categoryorder="array",
                    categoryarray=list(reversed(visible_boroughs)),
                    tickmode="array",
                    tickvals=visible_boroughs,
                    ticktext=visible_boroughs,
                )
            else:
                zone_labels = chart_df[opportunity_group_label].astype(str).tolist()
                fig_growth.update_yaxes(
                    categoryorder="array",
                    categoryarray=zone_labels,
                    tickmode="array",
                    tickvals=zone_labels,
                    ticktext=zone_labels,
                )
            fig_growth.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=45, b=10),
                xaxis_title="Trip delta",
                yaxis_title=None,
                showlegend=False,
            )
            st.plotly_chart(fig_growth, use_container_width=True, config={"displayModeBar": False})
            if opportunity_level == "Zone":
                centroids = load_zone_centroids()
                map_df = chart_df.merge(
                    centroids,
                    left_on=[opportunity_group_label, "Borough"],
                    right_on=["zone", "borough"],
                    how="left",
                ).dropna(subset=["latitude", "longitude"])
                if not map_df.empty:
                    map_df["radius_meters"] = (
                        220
                        + (map_df["Trip Delta"].abs() / max_abs_delta).clip(0, 1)
                        * 1450
                    )
                    map_df["fill_color"] = [
                        shade_rgba(value, max_abs_delta)
                        for value in map_df["Trip Delta"].tolist()
                    ]
                    map_df["tooltip"] = (
                        map_df[opportunity_group_label].astype(str)
                        + " | "
                        + map_df["Borough"].astype(str)
                        + "<br>Trip delta: "
                        + map_df["Trip Delta"].map(fmt_int)
                        + "<br>Growth: "
                        + map_df["Growth"].map(fmt_pct)
                    )
                    st.pydeck_chart(
                        pdk.Deck(
                            map_style=None,
                            initial_view_state=pdk.ViewState(
                                latitude=40.72,
                                longitude=-73.94,
                                zoom=9.7,
                                pitch=0,
                            ),
                            layers=[
                                pdk.Layer(
                                    "ScatterplotLayer",
                                    data=map_df,
                                    get_position="[longitude, latitude]",
                                    get_radius="radius_meters",
                                    get_fill_color="fill_color",
                                    get_line_color=[23, 32, 42, 170],
                                    line_width_min_pixels=1,
                                    stroked=True,
                                    filled=True,
                                    pickable=True,
                                    radius_min_pixels=4,
                                    radius_max_pixels=42,
                                )
                            ],
                            tooltip={"html": "{tooltip}"},
                        ),
                        use_container_width=True,
                    )
        elif opportunity_level == "Zone":
            st.info("No zone growth chart is shown for this selection because the comparison window does not have complete zone-level coverage.")
        st.dataframe(display_opportunities, use_container_width=True, hide_index=True)

with tab_context:
    st.subheader("Weather, Holidays, and Demand")
    daily_context = total_by_day(filtered).merge(
        tables["context"].rename(columns={"date": "pickup_date"}),
        on="pickup_date",
        how="left",
    )

    col1, col2 = st.columns((1, 1))
    with col1:
        fig_temp = px.scatter(
            daily_context,
            x="temperature_avg_f",
            y="trips",
            color="has_precipitation",
            color_discrete_map={False: ACCENT_BLUE, True: AQUA},
            trendline="ols",
            title="Trips vs Average Temperature",
            labels={
                "temperature_avg_f": "Average temperature (F)",
                "trips": "Trips",
                "has_precipitation": "Precipitation",
            },
        )
        fig_temp.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_temp, use_container_width=True, config={"displayModeBar": False})
    with col2:
        rainy = (
            daily_context.groupby("has_precipitation", as_index=False)["trips"]
            .mean()
            .replace({"has_precipitation": {False: "Dry", True: "Rain/Snow"}})
        )
        fig_rain = px.bar(
            rainy,
            x="has_precipitation",
            y="trips",
            title="Average Daily Trips by Weather",
            text_auto=".2s",
        )
        fig_rain.update_traces(marker_color=[ACCENT_BLUE, AQUA])
        fig_rain.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis_title=None,
            yaxis_title="Avg daily trips",
        )
        st.plotly_chart(fig_rain, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Event Impact")
    with st.expander("What events are included? How is the baseline calculated?"):
        st.markdown(
            """
            **Events included**
            - **Federal holiday:** official U.S. federal holidays from the calendar.
            - **Sports:** curated major home dates for NYC teams and major events, including Knicks, Nets, Rangers, Mets, Yankees, NYC Marathon, NYC Half Marathon, and US Open dates. This is a high-signal event calendar, not every regular-season home game.
            - **Concerts & music festivals:** Governors Ball, Global Citizen Festival, and Panorama Music Festival.
            - **Parades & civic festivals:** NYC Pride March, St. Patrick's Day Parade, Puerto Rican Day Parade, and Village Halloween Parade.
            - **Convention / expo:** New York Comic Con at the Javits Center.
            - **Civic / city event:** Times Square New Year's Eve and other local city demand moments that are not already federal holidays.

            **How the baseline is calculated**

            The selected date range controls which event dates appear. For each event date, the baseline is the **average daily trips on same-weekday days within +/- N days** (the "Baseline window" slider, default +/- 42 days), excluding other event dates in the calendar. So a Saturday Knicks game is compared with nearby Saturdays that have no events, not with the average across all weekdays.

            For single-borough events, the raw baseline uses that event borough's trips when that borough is selected. With all boroughs selected, a Knicks game uses Manhattan demand, a Mets game uses Queens demand, and a multi-borough event like the Marathon uses the selected NYC total. The DiD option then subtracts the same-day lift in the other boroughs to reduce citywide noise.
            """
        )
    event_scope = active[active[group_label].astype(str).isin(selected_groups)].copy()
    event_context = total_by_day(event_scope).merge(
        tables["context"].rename(columns={"date": "pickup_date"}),
        on="pickup_date",
        how="left",
    )
    events = build_event_calendar(event_context)
    if events.empty:
        st.info(
            "No events are available for this date range. Add rows to "
            "`data/processed/context/nyc_event_calendar.csv` with columns "
            "`date,event_name,event_category,venue,borough` to analyze sports, concerts, festivals, and other events."
        )
    else:
        available_categories = ordered_event_categories(events["event_category"].dropna().unique())
        event_col, baseline_col, mode_col = st.columns((1.4, 1, 0.8))
        with event_col:
            selected_event_categories = st.multiselect(
                "Event categories",
                available_categories,
                default=available_categories,
                placeholder="Choose event categories",
            )
        with baseline_col:
            baseline_days = st.slider(
                "Baseline window",
                min_value=14,
                max_value=84,
                value=42,
                step=7,
                help="Half-width of the same-weekday search window in days.",
            )
        with mode_col:
            lift_mode = st.radio(
                "Lift method",
                ["DiD", "Raw"],
                index=0,
                horizontal=True,
                help="DiD subtracts the same-day lift in other boroughs to remove citywide demand shocks. Raw uses only the event borough vs its same-weekday baseline.",
            )

        selected_events = events[
            events["event_category"].isin(selected_event_categories)
            & events["date"].between(start_date, end_date)
        ]
        if group_label == "borough":
            selected_borough_set = set(selected_groups)
            selected_events = selected_events[
                selected_events["borough"].isin(selected_borough_set)
                | selected_events["borough"].isin(["Multiple boroughs", ""])
                | selected_events["borough"].isna()
            ]

        # Build per-borough daily trips so each event can use its own borough's demand.
        if "borough" in event_scope.columns:
            _boro_daily = (
                event_scope.groupby(["pickup_date", "borough"], as_index=False)["trips"].sum()
            )
            _boro_daily["weekday_num"] = pd.to_datetime(_boro_daily["pickup_date"]).dt.weekday
        else:
            _boro_daily = None
        impacts = event_impact_table(
            event_context,
            selected_events,
            baseline_days,
            _boro_daily,
            exclusion_events=events,
        )
        if impacts.empty:
            st.info("No event days match the selected filters.")
        else:
            category_summary = (
                impacts.groupby("Category", as_index=False)
                .agg(
                    events=("Event", "count"),
                    avg_trips=("Trips", "mean"),
                    avg_baseline=("Baseline Trips", "mean"),
                    avg_lift=("Lift", "mean"),
                    avg_did_lift=("DiD Lift", "mean"),
                )
            )
            category_summary["did_available"] = category_summary["avg_did_lift"].notna()
            if lift_mode == "DiD":
                # DiD where we have borough data; fall back to raw for multi-borough events
                category_summary["plot_lift"] = category_summary["avg_did_lift"].where(
                    category_summary["avg_did_lift"].notna(), category_summary["avg_lift"]
                )
                chart_title = "Average Event-Day Lift - DiD (raw fallback for multi-borough events)"
                chart_caption = (
                    "DiD lift = event borough lift minus same-day lift in other boroughs, "
                    "removing citywide demand shocks. "
                    "Holidays and multi-borough events (Marathon, Half Marathon) show raw same-weekday lift."
                )
            else:
                category_summary["plot_lift"] = category_summary["avg_lift"]
                chart_title = "Average Event-Day Lift - Raw same-weekday baseline"
                chart_caption = (
                    "Raw lift = event borough trips on event day vs average on nearby same-weekday non-event days. "
                    "Does not control for citywide demand variation."
                )
            category_summary = category_summary.sort_values("plot_lift", ascending=False)
            category_summary["direction"] = category_summary["plot_lift"].apply(
                lambda value: "Higher than baseline" if value >= 0 else "Lower than baseline"
            )
            fig_events = px.bar(
                category_summary,
                x="plot_lift",
                y="Category",
                orientation="h",
                color="direction",
                color_discrete_map={
                    "Higher than baseline": POSITIVE_GREEN,
                    "Lower than baseline": NEGATIVE_RED,
                },
                title=chart_title,
                text=category_summary["plot_lift"].map(fmt_pct),
                hover_data={
                    "events": True,
                    "avg_trips": ":,.0f",
                    "avg_baseline": ":,.0f",
                    "avg_lift": ":.1%",
                    "avg_did_lift": ":.1%",
                    "direction": False,
                    "did_available": False,
                    "plot_lift": False,
                },
            )
            fig_events.update_layout(
                height=330,
                margin=dict(l=10, r=10, t=45, b=10),
                xaxis_title="Lift vs baseline",
                yaxis_title=None,
                legend_title_text=None,
            )
            st.plotly_chart(fig_events, use_container_width=True, config={"displayModeBar": False})
            st.caption(chart_caption)

            display_impacts = impacts.copy()
            display_impacts["Date"] = display_impacts["Date"].dt.strftime("%b %d, %Y")
            display_impacts["Trips"] = display_impacts["Trips"].map(fmt_int)
            display_impacts["Baseline Trips"] = display_impacts["Baseline Trips"].map(fmt_int)
            display_impacts["Lift"] = display_impacts["Lift"].map(fmt_pct)
            display_impacts["Control Lift"] = display_impacts["Control Lift"].map(fmt_pct)
            display_impacts["DiD Lift"] = display_impacts["DiD Lift"].map(fmt_pct)
            detail_columns = [
                "Date",
                "Event",
                "Venue",
                "Borough",
                "Trips",
                "Baseline Trips",
                "Lift",
                "Control Lift",
                "DiD Lift",
                "Baseline Days",
            ]
            st.markdown("**Event Details by Type**")
            detail_categories = ordered_event_categories(display_impacts["Category"].dropna().unique())
            if len(detail_categories) == 1:
                st.dataframe(
                    display_impacts[detail_columns],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                detail_tabs = st.tabs(detail_categories)
                for detail_tab, category in zip(detail_tabs, detail_categories):
                    with detail_tab:
                        category_rows = display_impacts[
                            display_impacts["Category"].eq(category)
                        ]
                        st.dataframe(
                            category_rows[detail_columns],
                            use_container_width=True,
                            hide_index=True,
                        )

st.caption(
    "Sources: FiveThirtyEight Uber TLC FOIL response, NYC TLC reference files, Open-Meteo historical weather. The dashboard tracks pickup demand, not revenue, completed trip duration, cancellations, or wait times."
)
