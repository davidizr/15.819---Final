from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq
import requests


warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external" / "tlc" / "fhv"
PROCESSED = ROOT / "data" / "processed"
DASHBOARD = PROCESSED / "dashboard"
ZONE_LOOKUP = PROCESSED / "dim_taxi_zones.csv"
SHARDS = PROCESSED / "monthly" / "gap_fhv_uber"
LINK_MANIFEST = PROCESSED / "tlc_trip_record_links.csv"

TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/fhv_tripdata_{year:04d}-{month:02d}.parquet"
HEADERS = {"User-Agent": "Mozilla/5.0 uber-nyc-dashboard/1.0"}
LINK_CACHE: dict[str, str] | None = None

UBER_BASE_NAMES = {
    "B02512": "Unter",
    "B02598": "Hinter",
    "B02617": "Weiter",
    "B02682": "Schmecken",
    "B02764": "Danach-NY",
    "B02765": "Grun",
    "B02835": "Dreist",
    "B02836": "Drinnen",
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
        return EXTERNAL / f"fhv_tripdata_{self.label}.parquet"


def load_link_cache() -> dict[str, str]:
    global LINK_CACHE
    if LINK_CACHE is not None:
        return LINK_CACHE
    if not LINK_MANIFEST.exists():
        LINK_CACHE = {}
        return LINK_CACHE
    links = pd.read_csv(LINK_MANIFEST)
    links = links[links["record_type"].eq("fhv")]
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


def request_exists(month: MonthRef) -> bool:
    if month.path.exists():
        return True
    try:
        response = requests.head(month.url, timeout=60, allow_redirects=True, headers=HEADERS)
        if response.status_code == 200:
            return True
        response = requests.get(
            month.url,
            timeout=60,
            allow_redirects=True,
            headers={**HEADERS, "Range": "bytes=0-0"},
        )
        if response.status_code in (200, 206):
            return True
    except requests.RequestException:
        pass

    curl = subprocess.run(
        ["curl.exe", "-s", "-L", "-I", month.url],
        capture_output=True,
        text=True,
        check=False,
    )
    if " 200 " in curl.stdout or " 200\r" in curl.stdout or " 200\n" in curl.stdout:
        return True
    return False


def discover_available_months(start: MonthRef, end: MonthRef) -> list[MonthRef]:
    available: list[MonthRef] = []
    for month in month_range(start, end):
        if request_exists(month):
            print(f"found {month.label}")
            available.append(month)
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


def normalize_fhv_batch(
    df: pd.DataFrame,
    zones: pd.DataFrame,
    pickup_month_label: str,
) -> pd.DataFrame:
    rename_map = {
        "PUlocationID": "location_id",
        "PULocationID": "location_id",
        "dispatching_base_num": "base",
    }
    df = df.rename(columns={old: new for old, new in rename_map.items() if old in df.columns})
    df["base"] = df["base"].fillna("")
    df = df[df["base"].isin(UBER_BASE_NAMES)].copy()
    if df.empty:
        return df

    df["company"] = "Uber"
    df["base_name"] = df["base"].map(UBER_BASE_NAMES).fillna("Unknown")
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime"])
    df["pickup_date"] = df["pickup_datetime"].dt.date
    df["pickup_month"] = pickup_month_label
    df["weekday"] = df["pickup_datetime"].dt.day_name()
    df["weekday_num"] = df["pickup_datetime"].dt.weekday
    df["hour"] = df["pickup_datetime"].dt.hour
    if "location_id" in df.columns:
        df["location_id"] = pd.to_numeric(df["location_id"], errors="coerce")
    else:
        df["location_id"] = pd.NA
    df = df.merge(zones, on="location_id", how="left")
    df["borough"] = df["borough"].fillna("Unknown")
    df["zone"] = df["zone"].fillna("Unknown")
    return df


def aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.groupby(keys, as_index=False, dropna=False).agg(
        trips=("pickup_datetime", "size")
    )


def process_month(
    month: MonthRef,
    zones: pd.DataFrame,
    keep_raw: bool,
    skip_zone: bool,
) -> dict[str, pd.DataFrame]:
    path = download_month(month)
    parquet = pq.ParquetFile(path)
    available_columns = set(parquet.schema_arrow.names)
    wanted = [
        "dispatching_base_num",
        "pickup_datetime",
        "PUlocationID",
        "PULocationID",
    ]
    columns = [column for column in wanted if column in available_columns]

    daily_parts: list[pd.DataFrame] = []
    borough_parts: list[pd.DataFrame] = []
    zone_parts: list[pd.DataFrame] = []
    hourly_parts: list[pd.DataFrame] = []
    weekday_hour_parts: list[pd.DataFrame] = []

    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        df = normalize_fhv_batch(batch.to_pandas(), zones, month.label)
        if df.empty:
            continue
        daily_parts.append(aggregate(df, ["pickup_date", "pickup_month", "company"]))
        borough_parts.append(
            aggregate(df, ["pickup_date", "pickup_month", "company", "borough"])
        )
        if not skip_zone:
            zone_parts.append(
                aggregate(
                    df,
                    ["pickup_date", "pickup_month", "company", "location_id", "borough", "zone"],
                )
            )
        hourly_parts.append(
            aggregate(df, ["pickup_date", "pickup_month", "hour", "company"])
        )
        weekday_hour_parts.append(
            aggregate(df, ["weekday_num", "weekday", "hour", "company"])
        )

    del parquet
    if not keep_raw and path.exists():
        remove_file_when_possible(path)

    return {
        "daily": final_group(daily_parts, ["pickup_date", "pickup_month", "company"]),
        "borough": final_group(
            borough_parts, ["pickup_date", "pickup_month", "company", "borough"]
        ),
        "zone": final_group(
            zone_parts,
            ["pickup_date", "pickup_month", "company", "location_id", "borough", "zone"],
        ),
        "hourly": final_group(
            hourly_parts, ["pickup_date", "pickup_month", "hour", "company"]
        ),
        "weekday_hour": final_group(
            weekday_hour_parts, ["weekday_num", "weekday", "hour", "company"]
        ),
    }


def shard_specs() -> dict[str, tuple[str, list[str]]]:
    return {
        "gap_fhv_uber_daily_by_company": (
            "daily",
            ["pickup_date", "pickup_month", "company"],
        ),
        "gap_fhv_uber_daily_by_borough": (
            "borough",
            ["pickup_date", "pickup_month", "company", "borough"],
        ),
        "gap_fhv_uber_daily_by_zone": (
            "zone",
            ["pickup_date", "pickup_month", "company", "location_id", "borough", "zone"],
        ),
        "gap_fhv_uber_hourly_by_company": (
            "hourly",
            ["pickup_date", "pickup_month", "hour", "company"],
        ),
        "gap_fhv_uber_weekday_hour_by_company": (
            "weekday_hour",
            ["weekday_num", "weekday", "hour", "company"],
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


def write_outputs() -> dict[str, int]:
    DASHBOARD.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for table_name, (_, keys) in shard_specs().items():
        df = final_group(read_shards(table_name), keys)
        df.to_csv(DASHBOARD / f"{table_name}.csv", index=False)
        counts[table_name] = len(df)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and aggregate public TLC FHV records for known Uber bases."
    )
    parser.add_argument("--start", default="2015-01", help="First month, YYYY-MM.")
    parser.add_argument("--end", default="2018-12", help="Last month, YYYY-MM.")
    parser.add_argument("--keep-raw", action="store_true", help="Keep downloaded Parquet files.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate month aggregate shards even when cached shards exist.",
    )
    parser.add_argument(
        "--skip-zone",
        action="store_true",
        help="Skip high-cardinality zone aggregates for large months.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_month(args.start)
    end = parse_month(args.end)
    zones = load_zones()
    months = discover_available_months(start, end)
    if not months:
        raise FileNotFoundError("No available TLC FHV months were found.")

    for index, month in enumerate(months, start=1):
        if month_shards_exist(month) and not args.overwrite:
            print(f"using cached aggregates for {month.label} ({index}/{len(months)})")
            continue
        print(f"processing {month.label} ({index}/{len(months)})")
        result = process_month(month, zones, args.keep_raw, args.skip_zone)
        write_month_shards(month, result)

    counts = write_outputs()
    manifest = {
        "source": "NYC TLC For-Hire Vehicle Trip Records",
        "url_template": TLC_URL,
        "requested_start": start.label,
        "requested_end": end.label,
        "processed_start": months[0].label,
        "processed_end": months[-1].label,
        "company": "Uber",
        "base_codes": UBER_BASE_NAMES,
        "raw_files_kept": args.keep_raw,
        "tables": counts,
    }
    (PROCESSED / "gap_fhv_uber_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
