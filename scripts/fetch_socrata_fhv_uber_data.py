from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DASHBOARD = PROCESSED / "dashboard"
SHARDS = PROCESSED / "monthly" / "gap_fhv_uber"
ZONE_LOOKUP = PROCESSED / "dim_taxi_zones.csv"

DATASETS = {
    2015: {"id": "7dfh-3irt", "datetime": "pickup_date", "location": "locationid"},
    2016: {"id": "yini-w76t", "datetime": "pickup_date", "location": "locationid"},
    2017: {"id": "avz8-mqzz", "datetime": "pickup_datetime", "location": "pulocationid"},
    2018: {"id": "am94-epxh", "datetime": "pickup_datetime", "location": "pulocationid"},
}

UBER_BASES = [
    "B02512",
    "B02598",
    "B02617",
    "B02682",
    "B02764",
    "B02765",
    "B02835",
    "B02836",
]


@dataclass(frozen=True)
class MonthRef:
    year: int
    month: int

    @property
    def label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def parse_month(value: str) -> MonthRef:
    year_text, month_text = value.split("-", maxsplit=1)
    return MonthRef(int(year_text), int(month_text))


def next_month(month: MonthRef) -> MonthRef:
    if month.month == 12:
        return MonthRef(month.year + 1, 1)
    return MonthRef(month.year, month.month + 1)


def month_range(start: MonthRef, end: MonthRef) -> list[MonthRef]:
    months: list[MonthRef] = []
    current = start
    while (current.year, current.month) <= (end.year, end.month):
        months.append(current)
        current = next_month(current)
    return months


def socrata_url(dataset_id: str, params: dict[str, str]) -> str:
    return f"https://data.cityofnewyork.us/resource/{dataset_id}.json?{urlencode(params)}"


def query_socrata(dataset_id: str, params: dict[str, str]) -> pd.DataFrame:
    url = socrata_url(dataset_id, params)
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=180)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(5 * (attempt + 1))
            continue
        raise RuntimeError(f"Socrata query failed {response.status_code}: {response.text[:500]}")
    raise RuntimeError(f"Socrata query failed after retries: {url}")


def base_where(datetime_col: str, month: MonthRef) -> str:
    start = f"{month.label}-01T00:00:00"
    end = f"{next_month(month).label}-01T00:00:00"
    bases = ",".join(f"'{base}'" for base in UBER_BASES)
    return (
        f"dispatching_base_num in({bases}) "
        f"AND {datetime_col} >= '{start}' "
        f"AND {datetime_col} < '{end}'"
    )


def load_zones() -> pd.DataFrame:
    zones = pd.read_csv(ZONE_LOOKUP)
    zones["location_id"] = pd.to_numeric(zones["location_id"], errors="coerce")
    return zones[["location_id", "borough", "zone"]]


def shard_specs() -> dict[str, list[str]]:
    return {
        "gap_fhv_uber_daily_by_company": ["pickup_date", "pickup_month", "company"],
        "gap_fhv_uber_daily_by_borough": ["pickup_date", "pickup_month", "company", "borough"],
        "gap_fhv_uber_daily_by_zone": [
            "pickup_date",
            "pickup_month",
            "company",
            "location_id",
            "borough",
            "zone",
        ],
        "gap_fhv_uber_hourly_by_company": ["pickup_date", "pickup_month", "hour", "company"],
        "gap_fhv_uber_weekday_hour_by_company": ["weekday_num", "weekday", "hour", "company"],
    }


def shard_exists(month: MonthRef) -> bool:
    return all((SHARDS / table / f"{month.label}.csv").exists() for table in shard_specs())


def write_shard(table: str, month: MonthRef, df: pd.DataFrame) -> None:
    out_dir = SHARDS / table
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{month.label}.csv", index=False)


def normalize_zone_counts(raw: pd.DataFrame, month: MonthRef, zones: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    df = raw.rename(columns={"pickup_day": "pickup_date", "trips": "trips"}).copy()
    df["pickup_date"] = pd.to_datetime(df["pickup_date"])
    df["pickup_month"] = month.label
    df["company"] = "Uber"
    df["location_id"] = pd.to_numeric(df["location_id"], errors="coerce")
    df["trips"] = pd.to_numeric(df["trips"], errors="coerce").fillna(0).astype("int64")
    df = df.merge(zones, on="location_id", how="left")
    df["borough"] = df["borough"].fillna("Unknown")
    df["zone"] = df["zone"].fillna("Unknown")
    return df[["pickup_date", "pickup_month", "company", "location_id", "borough", "zone", "trips"]]


def normalize_hour_counts(raw: pd.DataFrame, month: MonthRef) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    df = raw.rename(columns={"pickup_day": "pickup_date"}).copy()
    df["pickup_date"] = pd.to_datetime(df["pickup_date"])
    df["pickup_month"] = month.label
    df["company"] = "Uber"
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype("int64")
    df["trips"] = pd.to_numeric(df["trips"], errors="coerce").fillna(0).astype("int64")
    return df[["pickup_date", "pickup_month", "hour", "company", "trips"]]


def final_group(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=keys + ["trips"])
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby(keys, as_index=False, dropna=False)["trips"].sum().sort_values(keys)


def normalize_daily_counts(raw: pd.DataFrame, month: MonthRef) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    df = raw.rename(columns={"pickup_day": "pickup_date"}).copy()
    df["pickup_date"] = pd.to_datetime(df["pickup_date"])
    df["pickup_month"] = month.label
    df["company"] = "Uber"
    df["trips"] = pd.to_numeric(df["trips"], errors="coerce").fillna(0).astype("int64")
    return df[["pickup_date", "pickup_month", "company", "trips"]]


def process_month(month: MonthRef, zones: pd.DataFrame, daily_only: bool) -> None:
    config = DATASETS[month.year]
    dataset_id = config["id"]
    datetime_col = config["datetime"]
    location_col = config["location"]
    where = base_where(datetime_col, month)

    if daily_only:
        daily_raw = query_socrata(
            dataset_id,
            {
                "$select": f"date_trunc_ymd({datetime_col}) as pickup_day, count(*) as trips",
                "$where": where,
                "$group": "pickup_day",
                "$limit": "5000",
            },
        )
        daily = normalize_daily_counts(daily_raw, month)
        borough = daily.copy()
        borough["borough"] = "Unknown"
        zone = pd.DataFrame(
            columns=[
                "pickup_date",
                "pickup_month",
                "company",
                "location_id",
                "borough",
                "zone",
                "trips",
            ]
        )
    else:
        zone_raw = query_socrata(
            dataset_id,
            {
                "$select": (
                    f"date_trunc_ymd({datetime_col}) as pickup_day, "
                    f"{location_col} as location_id, count(*) as trips"
                ),
                "$where": where,
                "$group": f"pickup_day, {location_col}",
                "$limit": "50000",
            },
        )
        zone = normalize_zone_counts(zone_raw, month, zones)
        daily = final_group([zone], ["pickup_date", "pickup_month", "company"])
        borough = final_group([zone], ["pickup_date", "pickup_month", "company", "borough"])

    hour_raw = query_socrata(
        dataset_id,
        {
            "$select": (
                f"date_trunc_ymd({datetime_col}) as pickup_day, "
                f"date_extract_hh({datetime_col}) as hour, count(*) as trips"
            ),
            "$where": where,
            "$group": f"pickup_day, hour",
            "$limit": "50000",
        },
    )
    hourly = normalize_hour_counts(hour_raw, month)

    weekday_hour = hourly.copy()
    if not weekday_hour.empty:
        weekday_hour["weekday_num"] = weekday_hour["pickup_date"].dt.weekday
        weekday_hour["weekday"] = weekday_hour["pickup_date"].dt.day_name()
        weekday_hour = final_group(
            [weekday_hour], ["weekday_num", "weekday", "hour", "company"]
        )
    else:
        weekday_hour = pd.DataFrame(
            columns=["weekday_num", "weekday", "hour", "company", "trips"]
        )

    write_shard("gap_fhv_uber_daily_by_company", month, daily)
    write_shard("gap_fhv_uber_daily_by_borough", month, borough)
    write_shard("gap_fhv_uber_daily_by_zone", month, zone)
    write_shard("gap_fhv_uber_hourly_by_company", month, hourly)
    write_shard("gap_fhv_uber_weekday_hour_by_company", month, weekday_hour)


def combine_outputs() -> dict[str, int]:
    DASHBOARD.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for table, keys in shard_specs().items():
        shard_dir = SHARDS / table
        frames = [pd.read_csv(path, low_memory=False) for path in sorted(shard_dir.glob("*.csv"))]
        df = final_group(frames, keys)
        df.to_csv(DASHBOARD / f"{table}.csv", index=False)
        counts[table] = len(df)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Uber FHV aggregates from NYC Open Data.")
    parser.add_argument("--start", default="2015-01", help="First month, YYYY-MM.")
    parser.add_argument("--end", default="2018-12", help="Last month, YYYY-MM.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate cached month shards.")
    parser.add_argument(
        "--daily-only",
        action="store_true",
        help="Use lightweight daily/hourly aggregates and mark borough as Unknown.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_month(args.start)
    end = parse_month(args.end)
    zones = load_zones()
    months = [month for month in month_range(start, end) if month.year in DATASETS]
    for index, month in enumerate(months, start=1):
        if shard_exists(month) and not args.overwrite:
            print(f"using cached aggregates for {month.label} ({index}/{len(months)})")
            continue
        print(f"processing {month.label} ({index}/{len(months)})")
        process_month(month, zones, args.daily_only)
    counts = combine_outputs()
    manifest = {
        "source": "NYC Open Data Socrata year-specific FHV datasets",
        "datasets": {year: config["id"] for year, config in DATASETS.items()},
        "requested_start": start.label,
        "requested_end": end.label,
        "company": "Uber",
        "base_codes": UBER_BASES,
        "tables": counts,
    }
    (PROCESSED / "gap_fhv_uber_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
