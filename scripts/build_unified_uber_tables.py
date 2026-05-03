from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DASHBOARD = PROCESSED / "dashboard"
UNIFIED = DASHBOARD / "unified"


SOURCE_PRIORITY = {
    "five_thirty_eight_2014_gps": 10,
    "five_thirty_eight_2015_foil": 20,
    "tlc_fhv_uber_bases": 30,
    "tlc_hvfhv_uber": 40,
}


def read_dashboard(name: str, parse_dates: tuple[str, ...] = ("pickup_date",)) -> pd.DataFrame:
    path = DASHBOARD / name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    for column in parse_dates:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])
    return df


def add_standard_columns(df: pd.DataFrame, source: str, granularity: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "company" not in out.columns:
        out["company"] = "Uber"
    out["source"] = source
    out["source_priority"] = SOURCE_PRIORITY[source]
    out["granularity"] = granularity
    out["pickup_date"] = pd.to_datetime(out["pickup_date"])
    out["pickup_month"] = out["pickup_date"].dt.to_period("M").astype(str)
    out["year"] = out["pickup_date"].dt.year
    return out


def prefer_best_source(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_values(keys + ["source_priority"])
    return out.drop_duplicates(keys, keep="last").sort_values(keys)


def write_csv(df: pd.DataFrame, name: str) -> int:
    UNIFIED.mkdir(parents=True, exist_ok=True)
    path = UNIFIED / f"{name}.csv"
    df.to_csv(path, index=False)
    return len(df)


def build_daily_by_company() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    foil_2014 = read_dashboard("uber_2014_daily_by_base.csv")
    if not foil_2014.empty:
        df = (
            foil_2014.groupby(["pickup_date", "pickup_month"], as_index=False)["trips"]
            .sum()
        )
        frames.append(add_standard_columns(df, "five_thirty_eight_2014_gps", "daily"))

    foil_2015 = read_dashboard("uber_2015_daily_by_borough.csv")
    if not foil_2015.empty:
        df = (
            foil_2015.groupby(["pickup_date", "pickup_month"], as_index=False)["trips"]
            .sum()
        )
        frames.append(add_standard_columns(df, "five_thirty_eight_2015_foil", "daily"))

    gap = read_dashboard("gap_fhv_uber_daily_by_company.csv")
    if not gap.empty:
        frames.append(add_standard_columns(gap, "tlc_fhv_uber_bases", "daily"))

    modern = read_dashboard("modern_hvfhv_daily_by_company.csv")
    if not modern.empty:
        frames.append(add_standard_columns(modern, "tlc_hvfhv_uber", "daily"))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    keys = ["pickup_date", "company"]
    return prefer_best_source(combined, keys)


def build_daily_by_borough() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    foil_2014 = read_dashboard("uber_2014_daily_by_base.csv")
    if not foil_2014.empty:
        df = (
            foil_2014.groupby(["pickup_date", "pickup_month"], as_index=False)["trips"]
            .sum()
        )
        df["borough"] = "Unknown"
        frames.append(add_standard_columns(df, "five_thirty_eight_2014_gps", "daily_borough"))

    foil_2015 = read_dashboard("uber_2015_daily_by_borough.csv")
    if not foil_2015.empty:
        frames.append(
            add_standard_columns(foil_2015, "five_thirty_eight_2015_foil", "daily_borough")
        )

    gap = read_dashboard("gap_fhv_uber_daily_by_borough.csv")
    if not gap.empty:
        frames.append(add_standard_columns(gap, "tlc_fhv_uber_bases", "daily_borough"))

    modern = read_dashboard("modern_hvfhv_daily_by_borough.csv")
    if not modern.empty:
        frames.append(add_standard_columns(modern, "tlc_hvfhv_uber", "daily_borough"))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    keys = ["pickup_date", "company", "borough"]
    return prefer_best_source(combined, keys)


def build_daily_by_zone() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    foil_2015 = read_dashboard("uber_2015_daily_by_zone.csv")
    if not foil_2015.empty:
        foil_2015["company"] = "Uber"
        frames.append(
            add_standard_columns(foil_2015, "five_thirty_eight_2015_foil", "daily_zone")
        )

    gap = read_dashboard("gap_fhv_uber_daily_by_zone.csv")
    if not gap.empty:
        frames.append(add_standard_columns(gap, "tlc_fhv_uber_bases", "daily_zone"))

    modern = read_dashboard("modern_hvfhv_daily_by_zone.csv")
    if not modern.empty:
        frames.append(add_standard_columns(modern, "tlc_hvfhv_uber", "daily_zone"))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    keys = ["pickup_date", "company", "location_id"]
    return prefer_best_source(combined, keys)


def build_hourly_by_company() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    foil_2014 = read_dashboard("uber_2014_hourly_by_base.csv")
    if not foil_2014.empty:
        df = (
            foil_2014.groupby(["pickup_date", "pickup_month", "hour"], as_index=False)[
                "trips"
            ].sum()
        )
        frames.append(add_standard_columns(df, "five_thirty_eight_2014_gps", "hourly"))

    gap = read_dashboard("gap_fhv_uber_hourly_by_company.csv")
    if not gap.empty:
        frames.append(add_standard_columns(gap, "tlc_fhv_uber_bases", "hourly"))

    modern = read_dashboard("modern_hvfhv_hourly_by_company.csv")
    if not modern.empty:
        frames.append(add_standard_columns(modern, "tlc_hvfhv_uber", "hourly"))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    keys = ["pickup_date", "company", "hour"]
    return prefer_best_source(combined, keys)


def build_annual_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    return (
        daily.groupby(["year", "company"], as_index=False)
        .agg(
            trips=("trips", "sum"),
            first_date=("pickup_date", "min"),
            last_date=("pickup_date", "max"),
            days=("pickup_date", "nunique"),
        )
        .sort_values("year")
    )


def main() -> None:
    daily_company = build_daily_by_company()
    daily_borough = build_daily_by_borough()
    daily_zone = build_daily_by_zone()
    hourly_company = build_hourly_by_company()
    annual = build_annual_summary(daily_company)

    counts = {
        "unified_uber_daily_by_company": write_csv(
            daily_company, "unified_uber_daily_by_company"
        ),
        "unified_uber_daily_by_borough": write_csv(
            daily_borough, "unified_uber_daily_by_borough"
        ),
        "unified_uber_daily_by_zone": write_csv(daily_zone, "unified_uber_daily_by_zone"),
        "unified_uber_hourly_by_company": write_csv(
            hourly_company, "unified_uber_hourly_by_company"
        ),
        "unified_uber_annual_summary": write_csv(annual, "unified_uber_annual_summary"),
    }
    manifest = {
        "sources": SOURCE_PRIORITY,
        "notes": [
            "2014 FiveThirtyEight data has pickup latitude/longitude but no TLC zone, so borough is Unknown in unified borough tables.",
            "2015 overlap is de-duplicated by preferring public TLC FHV Uber-base records over FiveThirtyEight FOIL records when both exist for the same date/location.",
            "2019+ uses TLC High Volume FHV records filtered to HV0003, Uber.",
        ],
        "tables": counts,
    }
    (UNIFIED / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
