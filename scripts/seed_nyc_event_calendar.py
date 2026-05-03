from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "processed" / "context" / "nyc_event_calendar.csv"


def add_events(
    rows: list[dict[str, str]],
    event_name: str,
    event_category: str,
    dates: list[str],
    venue: str,
    borough: str,
) -> None:
    rows.extend(
        {
            "date": date,
            "event_name": event_name,
            "event_category": event_category,
            "venue": venue,
            "borough": borough,
        }
        for date in dates
    )


def build_events() -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    add_events(
        rows,
        "NYC Half Marathon",
        "Sports",
        [
            "2015-03-15",
            "2016-03-20",
            "2017-03-19",
            "2018-03-18",
            "2019-03-17",
            "2022-03-20",
            "2023-03-19",
            "2024-03-17",
            "2025-03-16",
            "2026-03-15",
        ],
        "Manhattan/Brooklyn route",
        "Multiple boroughs",
    )
    add_events(
        rows,
        "US Open Starts",
        "Sports",
        [
            "2015-08-31",
            "2016-08-29",
            "2017-08-28",
            "2018-08-27",
            "2019-08-26",
            "2020-08-31",
            "2021-08-30",
            "2022-08-29",
            "2023-08-28",
            "2024-08-26",
            "2025-08-24",
        ],
        "USTA Billie Jean King National Tennis Center",
        "Queens",
    )
    add_events(
        rows,
        "US Open Final",
        "Sports",
        [
            "2015-09-13",
            "2016-09-11",
            "2017-09-10",
            "2018-09-09",
            "2019-09-08",
            "2020-09-13",
            "2021-09-12",
            "2022-09-11",
            "2023-09-10",
            "2024-09-08",
            "2025-09-07",
        ],
        "USTA Billie Jean King National Tennis Center",
        "Queens",
    )
    add_events(
        rows,
        "NYC Marathon",
        "Sports",
        [
            "2015-11-01",
            "2016-11-06",
            "2017-11-05",
            "2018-11-04",
            "2019-11-03",
            "2021-11-07",
            "2022-11-06",
            "2023-11-05",
            "2024-11-03",
            "2025-11-02",
        ],
        "Five-borough route",
        "Multiple boroughs",
    )

    # NY Mets home games — Citi Field, Queens
    add_events(
        rows,
        "Mets Home Opener",
        "Sports",
        [
            "2015-04-13",
            "2016-04-08",
            "2017-04-06",
            "2018-04-05",
            "2019-04-04",
            "2021-04-08",
            "2022-04-15",
            "2023-04-06",
            "2024-03-29",
            "2025-03-27",
        ],
        "Citi Field",
        "Queens",
    )
    add_events(
        rows,
        "Mets NLDS 2015",
        "Sports",
        ["2015-10-12", "2015-10-13"],
        "Citi Field",
        "Queens",
    )
    add_events(
        rows,
        "Mets NLCS 2015",
        "Sports",
        ["2015-10-17", "2015-10-18"],
        "Citi Field",
        "Queens",
    )
    add_events(
        rows,
        "Mets World Series 2015",
        "Sports",
        ["2015-10-30", "2015-10-31", "2015-11-01"],
        "Citi Field",
        "Queens",
    )
    add_events(
        rows,
        "Mets Wild Card 2022",
        "Sports",
        ["2022-10-07", "2022-10-08"],
        "Citi Field",
        "Queens",
    )
    add_events(
        rows,
        "Mets Wild Card 2024",
        "Sports",
        ["2024-10-01", "2024-10-02"],
        "Citi Field",
        "Queens",
    )
    add_events(
        rows,
        "Mets NLDS 2024",
        "Sports",
        ["2024-10-08", "2024-10-09", "2024-10-11"],
        "Citi Field",
        "Queens",
    )
    add_events(
        rows,
        "Mets NLCS 2024",
        "Sports",
        ["2024-10-17", "2024-10-18", "2024-10-20"],
        "Citi Field",
        "Queens",
    )

    # NY Knicks home games — Madison Square Garden, Manhattan
    add_events(
        rows,
        "Knicks Home Opener",
        "Sports",
        [
            "2015-10-28",
            "2016-10-26",
            "2017-10-19",
            "2018-10-20",
            "2019-10-24",
            "2021-10-21",
            "2022-10-20",
            "2023-10-25",
            "2024-10-22",
        ],
        "Madison Square Garden",
        "Manhattan",
    )
    # 2021 R1 vs Hawks (home games G3, G4)
    add_events(
        rows,
        "Knicks Playoffs 2021",
        "Sports",
        ["2021-05-27", "2021-05-29"],
        "Madison Square Garden",
        "Manhattan",
    )
    # 2023 R1 vs Cavaliers (home G3, G4); R2 vs Heat (home G1, G2, G5)
    add_events(
        rows,
        "Knicks Playoffs 2023",
        "Sports",
        [
            "2023-04-20", "2023-04-22",
            "2023-05-02", "2023-05-04", "2023-05-10",
        ],
        "Madison Square Garden",
        "Manhattan",
    )
    # 2024 R1 vs Sixers; R2 vs Pacers; ECF vs Celtics (home games)
    add_events(
        rows,
        "Knicks Playoffs 2024",
        "Sports",
        [
            "2024-04-20", "2024-04-22", "2024-04-30",
            "2024-05-07", "2024-05-09", "2024-05-11", "2024-05-16",
            "2024-05-21", "2024-05-23", "2024-05-25",
        ],
        "Madison Square Garden",
        "Manhattan",
    )

    # NY Rangers home games — Madison Square Garden, Manhattan
    add_events(
        rows,
        "Rangers Home Opener",
        "Sports",
        [
            "2015-10-13",
            "2016-10-13",
            "2017-10-09",
            "2018-10-04",
            "2019-10-03",
            "2021-01-14",
            "2021-10-14",
            "2022-10-13",
            "2023-10-12",
            "2024-10-10",
        ],
        "Madison Square Garden",
        "Manhattan",
    )
    # 2022 R1 vs Penguins (home G1, G2, G5, G7); R2 vs Hurricanes; ECF vs Lightning
    add_events(
        rows,
        "Rangers Playoffs 2022",
        "Sports",
        [
            "2022-05-03", "2022-05-05", "2022-05-11", "2022-05-17",
            "2022-05-26", "2022-05-28",
            "2022-06-01", "2022-06-03",
        ],
        "Madison Square Garden",
        "Manhattan",
    )
    # 2024 R1 vs Capitals; R2 vs Hurricanes; ECF vs Panthers (home games)
    add_events(
        rows,
        "Rangers Playoffs 2024",
        "Sports",
        [
            "2024-04-22", "2024-04-24", "2024-04-30",
            "2024-05-05", "2024-05-07", "2024-05-13",
            "2024-05-22", "2024-05-24", "2024-05-28",
        ],
        "Madison Square Garden",
        "Manhattan",
    )

    add_events(
        rows,
        "Governors Ball",
        "Concerts & music festivals",
        [
            "2015-06-05",
            "2015-06-06",
            "2015-06-07",
            "2016-06-03",
            "2016-06-04",
            "2016-06-05",
            "2017-06-02",
            "2017-06-03",
            "2017-06-04",
            "2018-06-01",
            "2018-06-02",
            "2018-06-03",
            "2019-05-31",
            "2019-06-01",
            "2019-06-02",
        ],
        "Randall's Island Park",
        "Manhattan",
    )
    add_events(
        rows,
        "Governors Ball",
        "Concerts & music festivals",
        [
            "2021-09-24",
            "2021-09-25",
            "2021-09-26",
            "2022-06-10",
            "2022-06-11",
            "2022-06-12",
        ],
        "Citi Field",
        "Queens",
    )
    add_events(
        rows,
        "Governors Ball",
        "Concerts & music festivals",
        [
            "2023-06-09",
            "2023-06-10",
            "2023-06-11",
            "2024-06-07",
            "2024-06-08",
            "2024-06-09",
            "2025-06-06",
            "2025-06-07",
            "2025-06-08",
        ],
        "Flushing Meadows Corona Park",
        "Queens",
    )
    add_events(
        rows,
        "Global Citizen Festival",
        "Concerts & music festivals",
        [
            "2015-09-26",
            "2016-09-24",
            "2017-09-23",
            "2018-09-29",
            "2019-09-28",
            "2021-09-25",
            "2022-09-24",
            "2023-09-23",
            "2024-09-28",
        ],
        "Central Park",
        "Manhattan",
    )
    add_events(
        rows,
        "Panorama Music Festival",
        "Concerts & music festivals",
        [
            "2016-07-22",
            "2016-07-23",
            "2016-07-24",
            "2017-07-28",
            "2017-07-29",
            "2017-07-30",
            "2018-07-27",
            "2018-07-28",
            "2018-07-29",
        ],
        "Randall's Island Park",
        "Manhattan",
    )

    add_events(
        rows,
        "NYC Pride March",
        "Parades & civic festivals",
        [
            "2015-06-28",
            "2016-06-26",
            "2017-06-25",
            "2018-06-24",
            "2019-06-30",
            "2022-06-26",
            "2023-06-25",
            "2024-06-30",
            "2025-06-29",
        ],
        "Greenwich Village/Manhattan route",
        "Manhattan",
    )
    add_events(
        rows,
        "St. Patrick's Day Parade",
        "Parades & civic festivals",
        [
            "2015-03-17",
            "2016-03-17",
            "2017-03-17",
            "2018-03-17",
            "2019-03-17",
            "2022-03-17",
            "2023-03-17",
            "2024-03-17",
            "2025-03-17",
            "2026-03-17",
        ],
        "Fifth Avenue",
        "Manhattan",
    )
    add_events(
        rows,
        "Puerto Rican Day Parade",
        "Parades & civic festivals",
        [
            "2015-06-14",
            "2016-06-12",
            "2017-06-11",
            "2018-06-10",
            "2019-06-09",
            "2022-06-12",
            "2023-06-11",
            "2024-06-09",
            "2025-06-08",
        ],
        "Fifth Avenue",
        "Manhattan",
    )
    add_events(
        rows,
        "Village Halloween Parade",
        "Parades & civic festivals",
        [
            "2015-10-31",
            "2016-10-31",
            "2017-10-31",
            "2018-10-31",
            "2019-10-31",
            "2021-10-31",
            "2022-10-31",
            "2023-10-31",
            "2024-10-31",
            "2025-10-31",
        ],
        "Greenwich Village/Manhattan route",
        "Manhattan",
    )

    add_events(
        rows,
        "New York Comic Con",
        "Convention / expo",
        [
            "2015-10-08",
            "2015-10-09",
            "2015-10-10",
            "2015-10-11",
            "2016-10-06",
            "2016-10-07",
            "2016-10-08",
            "2016-10-09",
            "2017-10-05",
            "2017-10-06",
            "2017-10-07",
            "2017-10-08",
            "2018-10-04",
            "2018-10-05",
            "2018-10-06",
            "2018-10-07",
            "2019-10-03",
            "2019-10-04",
            "2019-10-05",
            "2019-10-06",
            "2021-10-07",
            "2021-10-08",
            "2021-10-09",
            "2021-10-10",
            "2022-10-06",
            "2022-10-07",
            "2022-10-08",
            "2022-10-09",
            "2023-10-12",
            "2023-10-13",
            "2023-10-14",
            "2023-10-15",
            "2024-10-17",
            "2024-10-18",
            "2024-10-19",
            "2024-10-20",
            "2025-10-09",
            "2025-10-10",
            "2025-10-11",
            "2025-10-12",
        ],
        "Javits Center",
        "Manhattan",
    )

    add_events(
        rows,
        "Times Square New Year's Eve",
        "Civic / city event",
        [
            "2015-12-31",
            "2016-12-31",
            "2017-12-31",
            "2018-12-31",
            "2019-12-31",
            "2020-12-31",
            "2021-12-31",
            "2022-12-31",
            "2023-12-31",
            "2024-12-31",
            "2025-12-31",
        ],
        "Times Square",
        "Manhattan",
    )

    events = pd.DataFrame(rows).drop_duplicates()
    events["date"] = pd.to_datetime(events["date"])
    events = events.sort_values(["date", "event_category", "event_name"])
    events["date"] = events["date"].dt.strftime("%Y-%m-%d")
    return events


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    events = build_events()
    events.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(events):,} events to {OUTPUT}")
    print(events["event_category"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
