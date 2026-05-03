from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq
import requests


warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external" / "tlc" / "hvfhv"
PROCESSED = ROOT / "data" / "processed"
DASHBOARD = PROCESSED / "dashboard"
ZONE_LOOKUP = PROCESSED / "dim_taxi_zones.csv"
SHARDS = PROCESSED / "monthly" / "modern_hvfhv"
LINK_MANIFEST = PROCESSED / "tlc_trip_record_links.csv"

TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_{year:04d}-{month:02d}.parquet"
HEADERS = {"User-Agent": "Mozilla/5.0 uber-nyc-dashboard/1.0"}
LINK_CACHE: dict[str, str] | None = None

HVFHV_COMPANIES = {
    "HV0002": "Juno",
    "HV0003": "Uber",
    "HV0004": "Via",
    "HV0005": "Lyft",
}

SUM_COLUMNS = {
    "trip_miles": "trip_miles",
    "trip_time": "trip_time_seconds",
    "base_passenger_fare": "passenger_fare",
    "tips": "tips",
    "driver_pay": "driver_pay",
    "airport_fee": "airport_fees",
    "congestion_surcharge": "congestion_surcharges",
    "cbd_congestion_fee": "cbd_congestion_fees",
}


@dataclass(frozen=True)
class MonthRef:
    year: int
    month: int

    @property
    def label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def url(self) -> str:
        return tlc_url_for_month(self)

    @property
    def path(self) -> Path:
        return EXTERNAL / f"fhvhv_tripdata_{self.label}.parquet"


def load_link_cache() -> dict[str, str]:
    global LINK_CACHE
    if LINK_CACHE is not None:
        return LINK_CACHE
    if not LINK_MANIFEST.exists():
        LINK_CACHE = {}
        return LINK_CACHE
    links = pd.read_csv(LINK_MANIFEST)
    links = links[links["record_type"].eq("hvfhv")]
    LINK_CACHE = dict(zip(links["month"].astype(str), links["url"].astype(str)))
    return LINK_CACHE


def tlc_url_for_month(month: MonthRef) -> str:
    return load_link_cache().get(
        month.label, TLC_URL.format(year=month.year, month=month.month)
    )


def parse_month(value: str) -> MonthRef:
    year_text, month_text = value.split("-", maxsplit=1)
    return MonthRef(int(year_text), int(month_text))


def month_range(start: MonthRef, end: MonthRef) -> Iterable[MonthRef]:
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield MonthRef(year, month)
        month += 1
        if month == 13:
            year += 1
            month = 1


def previous_month(today: date) -> MonthRef:
    year, month = today.year, today.month - 1
    if month == 0:
        year -= 1
        month = 12
    return MonthRef(year, month)


def request_head(month: MonthRef) -> tuple[bool, int | None]:
    if month.path.exists():
        return True, month.path.stat().st_size

    try:
        response = requests.head(month.url, timeout=60, allow_redirects=True, headers=HEADERS)
        if response.status_code == 200:
            size = response.headers.get("Content-Length")
            return True, int(size) if size and size.isdigit() else None
        response = requests.get(
            month.url,
            timeout=60,
            allow_redirects=True,
            headers={**HEADERS, "Range": "bytes=0-0"},
        )
        if response.status_code in (200, 206):
            size = response.headers.get("Content-Length")
            return True, int(size) if size and size.isdigit() else None
    except requests.RequestException:
        pass

    curl = subprocess.run(
        ["curl.exe", "-s", "-L", "-I", month.url],
        capture_output=True,
        text=True,
        check=False,
    )
    if " 200 " not in curl.stdout and " 200\r" not in curl.stdout and " 200\n" not in curl.stdout:
        return False, None
    size = None
    for line in curl.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            text = line.split(":", maxsplit=1)[1].strip()
            if text.isdigit():
                size = int(text)
    return True, size


def discover_available_months(start: MonthRef, end: MonthRef) -> list[MonthRef]:
    available: list[MonthRef] = []
    missed_after_seen = 0
    for month in month_range(start, end):
        exists, _ = request_head(month)
        if exists:
            available.append(month)
            missed_after_seen = 0
            print(f"found {month.label}")
        elif available:
            missed_after_seen += 1
            print(f"missing {month.label}")
            if missed_after_seen >= 3:
                break
        else:
            print(f"missing {month.label}")
    return available


def download_month(month: MonthRef, overwrite: bool = False) -> Path:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    if month.path.exists() and not overwrite:
        return month.path

    temp_path = month.path.with_suffix(".parquet.tmp")
    try:
        with requests.get(month.url, stream=True, timeout=180, headers=HEADERS) as response:
            response.raise_for_status()
            with temp_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
    except requests.RequestException:
        if temp_path.exists():
            temp_path.unlink()
        subprocess.run(
            [
                "curl.exe",
                "-L",
                "--fail",
                "--retry",
                "5",
                "--retry-delay",
                "5",
                "-o",
                str(temp_path),
                month.url,
            ],
            check=True,
        )
    temp_path.replace(month.path)
    return month.path


def remove_file_when_possible(path: Path) -> None:
    for _ in range(5):
        try:
            gc.collect()
            path.unlink()
            return
        except PermissionError:
            time.sleep(1)
    print(f"warning: could not delete locked raw file {path}")


def load_zones() -> pd.DataFrame:
    zones = pd.read_csv(ZONE_LOOKUP)
    zones["location_id"] = pd.to_numeric(zones["location_id"], errors="coerce")
    return zones[["location_id", "borough", "zone"]]


def normalize_flags(df: pd.DataFrame) -> pd.DataFrame:
    for column in ["shared_request_flag", "shared_match_flag"]:
        if column in df.columns:
            df[column] = df[column].eq("Y")
        else:
            df[column] = False
    return df


def prep_batch(
    df: pd.DataFrame,
    company_codes: set[str],
    zones: pd.DataFrame,
    pickup_month_label: str,
) -> pd.DataFrame:
    df = df[df["hvfhs_license_num"].isin(company_codes)].copy()
    if df.empty:
        return df

    df = df.rename(columns={"PULocationID": "location_id"})
    df["company"] = df["hvfhs_license_num"].map(HVFHV_COMPANIES).fillna("Unknown")
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime", "location_id"])
    df["pickup_date"] = df["pickup_datetime"].dt.date
    df["pickup_month"] = pickup_month_label
    df["weekday"] = df["pickup_datetime"].dt.day_name()
    df["weekday_num"] = df["pickup_datetime"].dt.weekday
    df["hour"] = df["pickup_datetime"].dt.hour
    df["location_id"] = pd.to_numeric(df["location_id"], errors="coerce")
    df = normalize_flags(df)
    df = df.merge(zones, on="location_id", how="left")
    return df


def aggregate_sum(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, str]] = {"trips": ("pickup_datetime", "size")}
    for source, output in SUM_COLUMNS.items():
        if source in df.columns:
            agg_spec[output] = (source, "sum")
    if "shared_request_flag" in df.columns:
        agg_spec["shared_requests"] = ("shared_request_flag", "sum")
    if "shared_match_flag" in df.columns:
        agg_spec["shared_matches"] = ("shared_match_flag", "sum")
    return df.groupby(keys, as_index=False, dropna=False).agg(**agg_spec)


def final_group(parts: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=keys + ["trips"])
    combined = pd.concat(parts, ignore_index=True)
    sum_cols = [col for col in combined.columns if col not in keys]
    return (
        combined.groupby(keys, as_index=False, dropna=False)[sum_cols]
        .sum()
        .sort_values(keys)
    )


def process_month(
    month: MonthRef,
    company_codes: set[str],
    zones: pd.DataFrame,
    keep_raw: bool,
    skip_zone: bool,
) -> dict[str, pd.DataFrame]:
    path = download_month(month)
    parquet = pq.ParquetFile(path)
    available_columns = set(parquet.schema_arrow.names)
    wanted_columns = [
        "hvfhs_license_num",
        "pickup_datetime",
        "PULocationID",
        "trip_miles",
        "trip_time",
        "base_passenger_fare",
        "tips",
        "driver_pay",
        "airport_fee",
        "congestion_surcharge",
        "cbd_congestion_fee",
        "shared_request_flag",
        "shared_match_flag",
    ]
    columns = [col for col in wanted_columns if col in available_columns]

    daily_parts: list[pd.DataFrame] = []
    borough_parts: list[pd.DataFrame] = []
    zone_parts: list[pd.DataFrame] = []
    hourly_parts: list[pd.DataFrame] = []
    weekday_hour_parts: list[pd.DataFrame] = []

    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        df = prep_batch(batch.to_pandas(), company_codes, zones, month.label)
        if df.empty:
            continue
        daily_parts.append(
            aggregate_sum(df, ["pickup_date", "pickup_month", "hvfhs_license_num", "company"])
        )
        borough_parts.append(
            aggregate_sum(
                df,
                ["pickup_date", "pickup_month", "hvfhs_license_num", "company", "borough"],
            )
        )
        if not skip_zone:
            zone_parts.append(
                aggregate_sum(
                    df,
                    [
                        "pickup_date",
                        "pickup_month",
                        "hvfhs_license_num",
                        "company",
                        "location_id",
                        "borough",
                        "zone",
                    ],
                )
            )
        hourly_parts.append(
            aggregate_sum(
                df,
                ["pickup_date", "pickup_month", "hour", "hvfhs_license_num", "company"],
            )
        )
        weekday_hour_parts.append(
            aggregate_sum(
                df,
                ["weekday_num", "weekday", "hour", "hvfhs_license_num", "company"],
            )
        )

    del parquet
    if not keep_raw and path.exists():
        remove_file_when_possible(path)

    return {
        "daily": final_group(daily_parts, ["pickup_date", "pickup_month", "hvfhs_license_num", "company"]),
        "borough": final_group(
            borough_parts,
            ["pickup_date", "pickup_month", "hvfhs_license_num", "company", "borough"],
        ),
        "zone": final_group(
            zone_parts,
            [
                "pickup_date",
                "pickup_month",
                "hvfhs_license_num",
                "company",
                "location_id",
                "borough",
                "zone",
            ],
        ),
        "hourly": final_group(
            hourly_parts,
            ["pickup_date", "pickup_month", "hour", "hvfhs_license_num", "company"],
        ),
        "weekday_hour": final_group(
            weekday_hour_parts,
            ["weekday_num", "weekday", "hour", "hvfhs_license_num", "company"],
        ),
    }


def shard_specs() -> dict[str, tuple[str, list[str]]]:
    return {
        "modern_hvfhv_daily_by_company": (
            "daily",
            ["pickup_date", "pickup_month", "hvfhs_license_num", "company"],
        ),
        "modern_hvfhv_daily_by_borough": (
            "borough",
            ["pickup_date", "pickup_month", "hvfhs_license_num", "company", "borough"],
        ),
        "modern_hvfhv_daily_by_zone": (
            "zone",
            [
                "pickup_date",
                "pickup_month",
                "hvfhs_license_num",
                "company",
                "location_id",
                "borough",
                "zone",
            ],
        ),
        "modern_hvfhv_hourly_by_company": (
            "hourly",
            ["pickup_date", "pickup_month", "hour", "hvfhs_license_num", "company"],
        ),
        "modern_hvfhv_weekday_hour_by_company": (
            "weekday_hour",
            ["weekday_num", "weekday", "hour", "hvfhs_license_num", "company"],
        ),
    }


def month_shards_exist(month: MonthRef) -> bool:
    return all(
        (SHARDS / table_name / f"{month.label}.csv").exists()
        for table_name in shard_specs()
    )


def write_month_shards(month: MonthRef, result: dict[str, pd.DataFrame]) -> None:
    for table_name, (result_key, _) in shard_specs().items():
        out_dir = SHARDS / table_name
        out_dir.mkdir(parents=True, exist_ok=True)
        result[result_key].to_csv(out_dir / f"{month.label}.csv", index=False)


def read_shards(table_name: str) -> list[pd.DataFrame]:
    shard_dir = SHARDS / table_name
    if not shard_dir.exists():
        return []
    frames: list[pd.DataFrame] = []
    for path in sorted(shard_dir.glob("*.csv")):
        try:
            frames.append(pd.read_csv(path, low_memory=False))
        except pd.errors.EmptyDataError:
            continue
    return frames


def write_outputs(month_results: list[dict[str, pd.DataFrame]]) -> dict[str, int]:
    DASHBOARD.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, (_, keys) in shard_specs().items():
        frames = read_shards(name)
        if not frames and month_results:
            result_key = shard_specs()[name][0]
            frames = [result[result_key] for result in month_results]
        frames = [frame for frame in frames if frame is not None and not frame.empty]
        df = final_group(frames, keys)
        path = DASHBOARD / f"{name}.csv"
        if df.empty and path.exists():
            counts[name] = len(pd.read_csv(path, low_memory=False))
            continue
        df.to_csv(path, index=False)
        counts[name] = len(df)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and aggregate modern TLC High Volume FHV data."
    )
    parser.add_argument("--start", default="2019-02", help="First month, YYYY-MM.")
    parser.add_argument(
        "--end",
        default=previous_month(date.today()).label,
        help="Last month to check, YYYY-MM. Future/unpublished months are skipped.",
    )
    parser.add_argument(
        "--companies",
        default="HV0003",
        help="Comma-separated HVFHV license codes. HV0003 is Uber; HV0005 is Lyft.",
    )
    parser.add_argument("--keep-raw", action="store_true", help="Keep downloaded Parquet files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSVs.")
    parser.add_argument(
        "--skip-zone",
        action="store_true",
        help="Skip high-cardinality zone aggregates for large backfills.",
    )
    parser.add_argument(
        "--limit-months",
        type=int,
        default=None,
        help="Process only the latest N discovered months. Useful for quick tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    company_codes = {
        value.strip().upper() for value in args.companies.split(",") if value.strip()
    }
    unknown = company_codes - set(HVFHV_COMPANIES)
    if unknown:
        raise ValueError(f"Unknown HVFHV codes: {sorted(unknown)}")

    start = parse_month(args.start)
    end = parse_month(args.end)
    zones = load_zones()
    months = discover_available_months(start, end)
    if args.limit_months:
        months = months[-args.limit_months :]
    if not months:
        raise FileNotFoundError("No available TLC HVFHV months were found.")

    results = []
    for index, month in enumerate(months, start=1):
        if month_shards_exist(month) and not args.overwrite:
            print(f"using cached aggregates for {month.label} ({index}/{len(months)})")
            continue
        print(f"processing {month.label} ({index}/{len(months)})")
        result = process_month(month, company_codes, zones, args.keep_raw, args.skip_zone)
        write_month_shards(month, result)
        results.append(result)

    counts = write_outputs(results)
    manifest = {
        "source": "NYC TLC High Volume For-Hire Vehicle Trip Records",
        "url_template": TLC_URL,
        "requested_start": start.label,
        "requested_end": end.label,
        "processed_start": months[0].label,
        "processed_end": months[-1].label,
        "companies": {code: HVFHV_COMPANIES[code] for code in sorted(company_codes)},
        "raw_files_kept": args.keep_raw,
        "tables": counts,
    }
    manifest_path = PROCESSED / "modern_hvfhv_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
