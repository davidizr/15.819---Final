from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data_source" / "uber-tlc-foil-response"
UBER_TRIP_DATA = SOURCE / "uber-trip-data"
OTHER_FHV_DATA = SOURCE / "other-FHV-data"
OUT = ROOT / "data" / "processed"
DASHBOARD = OUT / "dashboard"

CHUNK_SIZE = 500_000

BASE_NAMES = {
    "B02512": "Unter",
    "B02598": "Hinter",
    "B02617": "Weiter",
    "B02682": "Schmecken",
    "B02764": "Danach-NY",
    "B02765": "Grun",
    "B02835": "Dreist",
    "B02836": "Drinnen",
}

MONTH_FILES_2014 = [
    "uber-raw-data-apr14.csv",
    "uber-raw-data-may14.csv",
    "uber-raw-data-jun14.csv",
    "uber-raw-data-jul14.csv",
    "uber-raw-data-aug14.csv",
    "uber-raw-data-sep14.csv",
]


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DASHBOARD.mkdir(parents=True, exist_ok=True)


def clean_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    ).fillna(0).astype("int64")


def add_time_parts(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    dt = df[datetime_col]
    df["pickup_date"] = dt.dt.date
    df["pickup_month"] = dt.dt.to_period("M").astype(str)
    df["weekday"] = dt.dt.day_name()
    df["weekday_num"] = dt.dt.weekday
    df["hour"] = dt.dt.hour
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def final_group(parts: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=keys + ["trips"])
    return (
        pd.concat(parts, ignore_index=True)
        .groupby(keys, as_index=False, dropna=False)["trips"]
        .sum()
        .sort_values(keys)
    )


def load_taxi_zones() -> pd.DataFrame:
    zones = pd.read_csv(UBER_TRIP_DATA / "taxi-zone-lookup.csv")
    zones = zones.rename(
        columns={"LocationID": "location_id", "Borough": "borough", "Zone": "zone"}
    )
    write_csv(zones, OUT / "dim_taxi_zones.csv")
    return zones


def write_uber_bases() -> pd.DataFrame:
    bases = pd.DataFrame(
        [{"base": base, "base_name": name} for base, name in BASE_NAMES.items()]
    ).sort_values("base")
    write_csv(bases, OUT / "dim_uber_bases.csv")
    return bases


def process_uber_2014() -> dict[str, int]:
    daily_parts: list[pd.DataFrame] = []
    hourly_parts: list[pd.DataFrame] = []
    weekday_hour_parts: list[pd.DataFrame] = []
    writer: pq.ParquetWriter | None = None
    total_rows = 0

    trip_path = OUT / "uber_2014_trips.parquet"
    if trip_path.exists():
        trip_path.unlink()

    for file_name in MONTH_FILES_2014:
        path = UBER_TRIP_DATA / file_name
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
            chunk = chunk.rename(
                columns={
                    "Date/Time": "pickup_datetime",
                    "Lat": "lat",
                    "Lon": "lon",
                    "Base": "base",
                }
            )
            chunk["pickup_datetime"] = pd.to_datetime(
                chunk["pickup_datetime"], errors="coerce"
            )
            chunk = chunk.dropna(subset=["pickup_datetime", "lat", "lon", "base"])
            chunk["base"] = chunk["base"].str.upper()
            chunk["base_name"] = chunk["base"].map(BASE_NAMES).fillna("Unknown")
            chunk = add_time_parts(chunk, "pickup_datetime")

            total_rows += len(chunk)
            daily_parts.append(
                chunk.groupby(
                    ["pickup_date", "pickup_month", "base", "base_name"],
                    as_index=False,
                )
                .size()
                .rename(columns={"size": "trips"})
            )
            hourly_parts.append(
                chunk.groupby(
                    ["pickup_date", "pickup_month", "hour", "base", "base_name"],
                    as_index=False,
                )
                .size()
                .rename(columns={"size": "trips"})
            )
            weekday_hour_parts.append(
                chunk.groupby(
                    ["weekday_num", "weekday", "hour", "base", "base_name"],
                    as_index=False,
                )
                .size()
                .rename(columns={"size": "trips"})
            )

            trip_cols = ["pickup_datetime", "lat", "lon", "base", "base_name"]
            table = pa.Table.from_pandas(chunk[trip_cols], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(trip_path, table.schema, compression="snappy")
            writer.write_table(table)

    if writer is not None:
        writer.close()

    daily = final_group(
        daily_parts, ["pickup_date", "pickup_month", "base", "base_name"]
    )
    hourly = final_group(
        hourly_parts, ["pickup_date", "pickup_month", "hour", "base", "base_name"]
    )
    weekday_hour = final_group(
        weekday_hour_parts, ["weekday_num", "weekday", "hour", "base", "base_name"]
    )

    write_csv(daily, DASHBOARD / "uber_2014_daily_by_base.csv")
    write_csv(hourly, DASHBOARD / "uber_2014_hourly_by_base.csv")
    write_csv(weekday_hour, DASHBOARD / "uber_2014_weekday_hour_by_base.csv")
    return {
        "uber_2014_trips": total_rows,
        "uber_2014_daily_by_base": len(daily),
        "uber_2014_hourly_by_base": len(hourly),
        "uber_2014_weekday_hour_by_base": len(weekday_hour),
    }


def csv_member_from_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if name.lower().endswith(".csv") and not name.startswith("__MACOSX/"):
                return name
    raise FileNotFoundError(f"No CSV member found in {zip_path}")


def process_uber_2015(zones: pd.DataFrame) -> dict[str, int]:
    zip_path = UBER_TRIP_DATA / "uber-raw-data-janjune-15.csv.zip"
    csv_member = csv_member_from_zip(zip_path)
    zone_daily_parts: list[pd.DataFrame] = []
    dispatch_daily_parts: list[pd.DataFrame] = []
    affiliate_daily_parts: list[pd.DataFrame] = []
    total_rows = 0

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(csv_member) as csv_file:
            chunks = pd.read_csv(
                csv_file,
                chunksize=CHUNK_SIZE,
                dtype={
                    "Dispatching_base_num": "string",
                    "Pickup_date": "string",
                    "Affiliated_base_num": "string",
                    "locationID": "Int64",
                },
            )
            for chunk in chunks:
                chunk = chunk.rename(
                    columns={
                        "Dispatching_base_num": "dispatching_base",
                        "Pickup_date": "pickup_datetime",
                        "Affiliated_base_num": "affiliated_base",
                        "locationID": "location_id",
                    }
                )
                chunk["pickup_datetime"] = pd.to_datetime(
                    chunk["pickup_datetime"], errors="coerce"
                )
                chunk = chunk.dropna(subset=["pickup_datetime", "location_id"])
                chunk["pickup_date"] = chunk["pickup_datetime"].dt.date
                chunk["pickup_month"] = (
                    chunk["pickup_datetime"].dt.to_period("M").astype(str)
                )
                chunk["dispatching_base"] = chunk["dispatching_base"].str.upper()
                chunk["affiliated_base"] = chunk["affiliated_base"].str.upper()

                total_rows += len(chunk)
                zone_daily_parts.append(
                    chunk.groupby(
                        ["pickup_date", "pickup_month", "location_id"],
                        as_index=False,
                    )
                    .size()
                    .rename(columns={"size": "trips"})
                )
                dispatch_daily_parts.append(
                    chunk.groupby(
                        ["pickup_date", "pickup_month", "dispatching_base"],
                        as_index=False,
                        dropna=False,
                    )
                    .size()
                    .rename(columns={"size": "trips"})
                )
                affiliate_daily_parts.append(
                    chunk.groupby(
                        ["pickup_date", "pickup_month", "affiliated_base"],
                        as_index=False,
                        dropna=False,
                    )
                    .size()
                    .rename(columns={"size": "trips"})
                )

    zone_daily = final_group(zone_daily_parts, ["pickup_date", "pickup_month", "location_id"])
    zone_daily = zone_daily.merge(zones, on="location_id", how="left")
    borough_daily = final_group(
        [
            zone_daily.groupby(
                ["pickup_date", "pickup_month", "borough"],
                as_index=False,
                dropna=False,
            )["trips"].sum()
        ],
        ["pickup_date", "pickup_month", "borough"],
    )
    dispatch_daily = final_group(
        dispatch_daily_parts, ["pickup_date", "pickup_month", "dispatching_base"]
    )
    affiliate_daily = final_group(
        affiliate_daily_parts, ["pickup_date", "pickup_month", "affiliated_base"]
    )

    write_csv(zone_daily, DASHBOARD / "uber_2015_daily_by_zone.csv")
    write_csv(borough_daily, DASHBOARD / "uber_2015_daily_by_borough.csv")
    write_csv(dispatch_daily, DASHBOARD / "uber_2015_daily_by_dispatch_base.csv")
    write_csv(affiliate_daily, DASHBOARD / "uber_2015_daily_by_affiliated_base.csv")
    return {
        "uber_2015_trips": total_rows,
        "uber_2015_daily_by_zone": len(zone_daily),
        "uber_2015_daily_by_borough": len(borough_daily),
        "uber_2015_daily_by_dispatch_base": len(dispatch_daily),
        "uber_2015_daily_by_affiliated_base": len(affiliate_daily),
    }


def process_uber_jan_feb_2015() -> dict[str, int]:
    df = pd.read_csv(SOURCE / "Uber-Jan-Feb-FOIL.csv")
    df = df.rename(
        columns={
            "dispatching_base_number": "dispatching_base",
            "date": "pickup_date",
        }
    )
    df["pickup_date"] = pd.to_datetime(df["pickup_date"], errors="coerce").dt.date
    df["pickup_month"] = pd.to_datetime(df["pickup_date"]).dt.to_period("M").astype(str)
    df["dispatching_base"] = df["dispatching_base"].str.upper()
    df["base_name"] = df["dispatching_base"].map(BASE_NAMES).fillna("Unknown")
    df["active_vehicles"] = clean_number(df["active_vehicles"])
    df["trips"] = clean_number(df["trips"])
    df = df[
        [
            "pickup_date",
            "pickup_month",
            "dispatching_base",
            "base_name",
            "active_vehicles",
            "trips",
        ]
    ].sort_values(["pickup_date", "dispatching_base"])
    write_csv(df, DASHBOARD / "uber_jan_feb_2015_daily_by_base.csv")
    return {"uber_jan_feb_2015_daily_by_base": len(df)}


def process_other_fhv_2015() -> dict[str, int]:
    path = OTHER_FHV_DATA / "other-FHV-data-jan-aug-2015.csv"
    df = pd.read_csv(path, skiprows=5)
    df = df.rename(
        columns={
            "Base Number": "base",
            "Base Name": "base_name",
            "Pick Up Date": "pickup_date",
            "Number of Trips": "trips",
            "Number of Vehicles": "vehicles",
        }
    )
    df = df[["base", "base_name", "pickup_date", "trips", "vehicles"]].dropna(
        subset=["base", "pickup_date"]
    )
    df["base"] = df["base"].astype(str).str.upper().str.strip()
    df["base_name"] = df["base_name"].astype(str).str.strip()
    df["pickup_date"] = pd.to_datetime(df["pickup_date"], errors="coerce").dt.date
    df["pickup_month"] = pd.to_datetime(df["pickup_date"]).dt.to_period("M").astype(str)
    df["trips"] = clean_number(df["trips"])
    df["vehicles"] = clean_number(df["vehicles"])
    df = df[
        ["pickup_date", "pickup_month", "base", "base_name", "trips", "vehicles"]
    ].sort_values(["pickup_date", "base"])
    write_csv(df, DASHBOARD / "fhv_2015_daily_by_base.csv")
    return {"fhv_2015_daily_by_base": len(df)}


def write_manifest(counts: dict[str, int]) -> None:
    manifest = {
        "source_repo": "https://github.com/fivethirtyeight/uber-tlc-foil-response",
        "source_dir": str(SOURCE.relative_to(ROOT)),
        "processed_dir": str(OUT.relative_to(ROOT)),
        "dashboard_dir": str(DASHBOARD.relative_to(ROOT)),
        "tables": counts,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            "Expected cloned data repo at data_source/uber-tlc-foil-response"
        )

    ensure_dirs()
    counts: dict[str, int] = {}
    zones = load_taxi_zones()
    write_uber_bases()
    counts.update(process_uber_2014())
    counts.update(process_uber_2015(zones))
    counts.update(process_uber_jan_feb_2015())
    counts.update(process_other_fhv_2015())
    write_manifest(counts)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
