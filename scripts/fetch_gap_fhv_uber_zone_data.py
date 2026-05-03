from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from fetch_gap_fhv_uber_data import (
    DASHBOARD,
    SHARDS,
    UBER_BASE_NAMES,
    discover_available_months,
    download_month,
    load_zones,
    month_range,
    parse_month,
    remove_file_when_possible,
)


ZONE_TABLE = "gap_fhv_uber_daily_by_zone"
ZONE_KEYS = ["pickup_date", "pickup_month", "company", "location_id", "borough", "zone"]


def zone_shard_path(month_label: str) -> Path:
    return SHARDS / ZONE_TABLE / f"{month_label}.csv"


def shard_has_rows(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return sum(1 for _ in path.open(encoding="utf-8")) > 1
    except OSError:
        return False


def final_group(parts: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=keys + ["trips"])
    combined = pd.concat(parts, ignore_index=True)
    return (
        combined.groupby(keys, as_index=False, dropna=False)["trips"]
        .sum()
        .sort_values(keys)
    )


def process_zone_month(month, zones: pd.DataFrame, keep_raw: bool) -> pd.DataFrame:
    path = download_month(month)
    parquet = pq.ParquetFile(path)
    available_columns = set(parquet.schema_arrow.names)
    wanted_columns = ["dispatching_base_num", "pickup_datetime", "PUlocationID", "PULocationID"]
    columns = [column for column in wanted_columns if column in available_columns]
    if "dispatching_base_num" not in columns or "pickup_datetime" not in columns:
        raise ValueError(f"{month.label} does not have expected FHV base/date columns")

    parts: list[pd.DataFrame] = []
    for batch in parquet.iter_batches(batch_size=250_000, columns=columns):
        df = batch.to_pandas()
        rename_map = {
            "PUlocationID": "location_id",
            "PULocationID": "location_id",
            "dispatching_base_num": "base",
        }
        df = df.rename(columns={old: new for old, new in rename_map.items() if old in df.columns})
        if "location_id" not in df.columns:
            continue
        df["base"] = df["base"].fillna("")
        df = df[df["base"].isin(UBER_BASE_NAMES)]
        if df.empty:
            continue
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        df["location_id"] = pd.to_numeric(df["location_id"], errors="coerce")
        df = df.dropna(subset=["pickup_datetime", "location_id"])
        if df.empty:
            continue
        df["pickup_date"] = df["pickup_datetime"].dt.date
        df["pickup_month"] = month.label
        df["company"] = "Uber"
        parts.append(
            df.groupby(
                ["pickup_date", "pickup_month", "company", "location_id"],
                as_index=False,
                dropna=False,
            )
            .size()
            .rename(columns={"size": "trips"})
        )

    del parquet
    if not keep_raw and path.exists():
        remove_file_when_possible(path)

    grouped = final_group(parts, ["pickup_date", "pickup_month", "company", "location_id"])
    if grouped.empty:
        return pd.DataFrame(columns=ZONE_KEYS + ["trips"])
    grouped["location_id"] = pd.to_numeric(grouped["location_id"], errors="coerce")
    grouped = grouped.merge(zones, on="location_id", how="left")
    grouped["borough"] = grouped["borough"].fillna("Unknown")
    grouped["zone"] = grouped["zone"].fillna("Unknown")
    return grouped[ZONE_KEYS + ["trips"]]


def read_zone_shards() -> list[pd.DataFrame]:
    shard_dir = SHARDS / ZONE_TABLE
    if not shard_dir.exists():
        return []
    frames: list[pd.DataFrame] = []
    for path in sorted(shard_dir.glob("*.csv")):
        try:
            frame = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    return frames


def write_combined_zone_output() -> int:
    DASHBOARD.mkdir(parents=True, exist_ok=True)
    combined = final_group(read_zone_shards(), ZONE_KEYS)
    combined.to_csv(DASHBOARD / f"{ZONE_TABLE}.csv", index=False)
    return len(combined)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill legacy FHV Uber-base zone aggregates without recomputing all tables."
    )
    parser.add_argument("--start", default="2015-01", help="First month, YYYY-MM.")
    parser.add_argument("--end", default="2019-01", help="Last month, YYYY-MM.")
    parser.add_argument("--keep-raw", action="store_true", help="Keep downloaded Parquet files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite nonempty zone shards.")
    parser.add_argument(
        "--trust-urls",
        action="store_true",
        help="Use the requested month range directly instead of probing URLs first.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_month(args.start)
    end = parse_month(args.end)
    months = list(month_range(start, end)) if args.trust_urls else discover_available_months(start, end)
    zones = load_zones()
    processed: list[str] = []
    skipped: list[str] = []
    failed: dict[str, str] = {}

    for index, month in enumerate(months, start=1):
        path = zone_shard_path(month.label)
        if shard_has_rows(path) and not args.overwrite:
            print(f"using cached zone aggregate for {month.label} ({index}/{len(months)})")
            skipped.append(month.label)
            continue
        print(f"processing zone aggregate {month.label} ({index}/{len(months)})")
        try:
            result = process_zone_month(month, zones, args.keep_raw)
        except Exception as exc:
            print(f"warning: failed {month.label}: {exc}")
            failed[month.label] = str(exc)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False)
        processed.append(month.label)

    row_count = write_combined_zone_output()
    manifest = {
        "source": "NYC TLC For-Hire Vehicle Trip Records",
        "table": ZONE_TABLE,
        "requested_start": start.label,
        "requested_end": end.label,
        "processed_months": processed,
        "skipped_months": skipped,
        "failed_months": failed,
        "company": "Uber",
        "base_codes": UBER_BASE_NAMES,
        "raw_files_kept": args.keep_raw,
        "rows": row_count,
    }
    manifest_path = DASHBOARD.parent / "gap_fhv_uber_zone_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
