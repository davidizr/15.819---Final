from __future__ import annotations

import csv
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "tlc_trip_record_links.csv"
PAGE_URL = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href:
            self._href = href
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href:
            self.links.append((self._href, " ".join(self._text).strip()))
            self._href = None
            self._text = []


def record_type_from_url(url: str) -> str | None:
    filename = url.rsplit("/", maxsplit=1)[-1].lower()
    if filename.startswith("fhvhv_tripdata_"):
        return "hvfhv"
    if filename.startswith("fhv_tripdata_"):
        return "fhv"
    if filename.startswith("yellow_tripdata_"):
        return "yellow"
    if filename.startswith("green_tripdata_"):
        return "green"
    return None


def month_from_url(url: str) -> str | None:
    match = re.search(r"_(\d{4}-\d{2})\.parquet(?:\?|$)", url.lower())
    return match.group(1) if match else None


def main() -> None:
    request = urllib.request.Request(PAGE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        html = response.read().decode("utf-8", errors="replace")

    parser = LinkParser()
    parser.feed(html)

    rows = []
    for href, label in parser.links:
        if "trip-data" not in href or not href.lower().endswith(".parquet"):
            continue
        record_type = record_type_from_url(href)
        month = month_from_url(href)
        if not record_type or not month:
            continue
        rows.append(
            {
                "month": month,
                "year": month[:4],
                "record_type": record_type,
                "label": label,
                "url": href,
            }
        )

    rows = sorted(rows, key=lambda row: (row["month"], row["record_type"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["month", "year", "record_type", "label", "url"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} links to {OUT}")


if __name__ == "__main__":
    main()
