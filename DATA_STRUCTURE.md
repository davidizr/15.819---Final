# Uber TLC Dashboard Data Structure

Raw source data is cloned from FiveThirtyEight into:

```text
data_source/uber-tlc-foil-response/
```

Processed dashboard tables are generated into:

```text
data/processed/
data/processed/dashboard/
```

Run the preparation step with:

```powershell
python scripts/prepare_uber_data.py
```

Run the supplemental context pull with:

```powershell
python scripts/fetch_supplemental_data.py
```

## Core Tables

`data/processed/dim_uber_bases.csv`

- `base`: TLC base code
- `base_name`: readable Uber base name from the FiveThirtyEight README

`data/processed/dim_taxi_zones.csv`

- `location_id`: TLC taxi zone ID
- `borough`: borough name
- `zone`: zone name

`data/processed/uber_2014_trips.parquet`

- Trip-level Uber pickups from April through September 2014
- Columns: `pickup_datetime`, `lat`, `lon`, `base`, `base_name`
- Best for map views or detailed geospatial analysis

## Dashboard Tables

`data/processed/dashboard/uber_2014_daily_by_base.csv`

- Daily Uber pickup counts by base for April through September 2014

`data/processed/dashboard/uber_2014_hourly_by_base.csv`

- Hourly Uber pickup counts by day and base for April through September 2014

`data/processed/dashboard/uber_2014_weekday_hour_by_base.csv`

- Weekday/hour heatmap table by Uber base for April through September 2014

`data/processed/dashboard/uber_2015_daily_by_zone.csv`

- Daily Uber pickup counts by TLC taxi zone for January through June 2015

`data/processed/dashboard/uber_2015_daily_by_borough.csv`

- Daily Uber pickup counts by borough for January through June 2015

`data/processed/dashboard/uber_2015_daily_by_dispatch_base.csv`

- Daily Uber pickup counts by dispatching base for January through June 2015

`data/processed/dashboard/uber_2015_daily_by_affiliated_base.csv`

- Daily Uber pickup counts by affiliated base for January through June 2015

`data/processed/dashboard/uber_jan_feb_2015_daily_by_base.csv`

- Daily Uber trips and active vehicles by base from the FOIL aggregate file

`data/processed/dashboard/fhv_2015_daily_by_base.csv`

- Daily trips and vehicles for 329 FHV companies from January through August 2015

`data/processed/manifest.json`

- Source and output paths plus row counts for generated tables

## Supplemental Context Tables

`data/processed/context/nyc_daily_weather.csv`

- Daily historical weather for New York City from April 1, 2014 through August 31, 2015
- Source: Open-Meteo Historical Weather API
- Useful for explaining rainy-day, snowy-day, heat, and wind effects on demand

`data/processed/context/calendar_features.csv`

- Daily calendar attributes over the same date range
- Includes weekday/weekend flags, month boundaries, federal holidays, and a few manually tagged NYC demand moments

`data/processed/dashboard/daily_context.csv`

- Dashboard-ready join of weather and calendar features by date
- Join to trip tables on `pickup_date`

`data/processed/dashboard/*_with_context.csv`

- Enriched daily dashboard tables with weather and calendar columns already joined
- Useful for charts like rainy-day trip lift, holiday demand, weekend behavior, and weather-aware growth comparisons

`data/external/tlc/taxi_zone_lookup.csv`

- Latest official TLC taxi zone lookup table

`data/external/tlc/taxi_zones.zip`

- Official TLC taxi zone shapefile archive
- Useful later for borough/zone choropleth maps if the dashboard stack supports shapefiles or GeoJSON

## Larger Optional Sources

The official TLC trip-record portal also has monthly yellow taxi, green taxi, FHV, and high-volume FHV Parquet files. These are excellent for competitive benchmarking, but they can add gigabytes quickly. For this project, prefer downloading only the months and vehicle types you plan to compare directly.

## Modern TLC HVFHV Data

The FiveThirtyEight FOIL data stops in 2015. For modern Uber demand, use NYC TLC High Volume FHV records, which begin in February 2019 and identify companies by `hvfhs_license_num`.

The official TLC page can be scraped into a local link manifest with:

```powershell
python scripts/discover_tlc_trip_record_links.py
```

This writes `data/processed/tlc_trip_record_links.csv`.

```powershell
python scripts/fetch_modern_hvfhv_data.py --overwrite
```

By default, this processes Uber only (`HV0003`) from February 2019 through the latest TLC month discovered before today. The script downloads one monthly Parquet file at a time, aggregates it, and deletes the raw file unless `--keep-raw` is passed.

Modern output tables:

- `data/processed/dashboard/modern_hvfhv_daily_by_company.csv`
- `data/processed/dashboard/modern_hvfhv_daily_by_borough.csv`
- `data/processed/dashboard/modern_hvfhv_daily_by_zone.csv`
- `data/processed/dashboard/modern_hvfhv_hourly_by_company.csv`
- `data/processed/dashboard/modern_hvfhv_weekday_hour_by_company.csv`
- `data/processed/modern_hvfhv_manifest.json`

Useful options:

```powershell
python scripts/fetch_modern_hvfhv_data.py --companies HV0003,HV0005 --overwrite
python scripts/fetch_modern_hvfhv_data.py --limit-months 3 --overwrite
```

## Gap-Year TLC FHV Uber Data

For 2015 through 2018, Uber can be approximated from public TLC FHV records by filtering to known Uber dispatching base codes from the FiveThirtyEight FOIL README.

```powershell
python scripts/fetch_gap_fhv_uber_data.py --start 2015-01 --end 2018-12
```

Gap-year output tables:

- `data/processed/dashboard/gap_fhv_uber_daily_by_company.csv`
- `data/processed/dashboard/gap_fhv_uber_daily_by_borough.csv`
- `data/processed/dashboard/gap_fhv_uber_daily_by_zone.csv`
- `data/processed/dashboard/gap_fhv_uber_hourly_by_company.csv`
- `data/processed/dashboard/gap_fhv_uber_weekday_hour_by_company.csv`
- `data/processed/gap_fhv_uber_manifest.json`

## Unified Dashboard Tables

Build the canonical all-year tables with:

```powershell
python scripts/build_unified_uber_tables.py
```

Unified output tables:

- `data/processed/dashboard/unified/unified_uber_daily_by_company.csv`
- `data/processed/dashboard/unified/unified_uber_daily_by_borough.csv`
- `data/processed/dashboard/unified/unified_uber_daily_by_zone.csv`
- `data/processed/dashboard/unified/unified_uber_hourly_by_company.csv`
- `data/processed/dashboard/unified/unified_uber_annual_summary.csv`

The unified builder de-duplicates overlapping records by source priority:

1. TLC HVFHV Uber records for 2019 onward
2. TLC FHV Uber-base records for 2015-2018
3. FiveThirtyEight 2015 FOIL records
4. FiveThirtyEight 2014 GPS records
