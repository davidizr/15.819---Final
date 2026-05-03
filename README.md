# Uber NYC Performance Dashboard

Streamlit dashboard for exploring historical NYC Uber pickup demand using the FiveThirtyEight TLC FOIL response dataset, enriched with weather, calendar, and TLC zone context.

## Run

```powershell
streamlit run streamlit_app.py
```

If `streamlit` is not on PATH:

```powershell
python -m streamlit run streamlit_app.py
```

## Current Dashboard

- Default view: unified Uber demand across every loaded source/year
- Overview: trips, daily trend, 7-day average, monthly volume, base/borough mix
- Timing: 2014 hourly demand and weekday/hour heatmap, plus 2015 weekday patterns
- Geography: 2014 pickup density map and 2015 borough/zone ranking
- Context: weather, precipitation, holiday, and event demand cuts
- Market: Jan-Feb 2015 trips per active vehicle and FHV competitor rankings

## Data Prep

```powershell
python scripts/prepare_uber_data.py
python scripts/fetch_supplemental_data.py
```

Discover TLC monthly Parquet links from the official page:

```powershell
python scripts/discover_tlc_trip_record_links.py
```

Modern TLC HVFHV data:

```powershell
python scripts/fetch_modern_hvfhv_data.py --overwrite
```

Use `--limit-months 3` for a quick trial, or `--companies HV0003,HV0005` to add Lyft alongside Uber.

Gap-year TLC FHV Uber-base data:

```powershell
python scripts/fetch_gap_fhv_uber_data.py --start 2015-01 --end 2018-12
```

Unified dashboard tables:

```powershell
python scripts/build_unified_uber_tables.py
```

The TLC download scripts cache month-level aggregate shards under `data/processed/monthly/`, so reruns resume without reprocessing completed months unless `--overwrite` is passed.
