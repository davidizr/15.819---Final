from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd


ROOT = Path(__file__).resolve().parents[1]
ZONE_SHAPEFILE = ROOT / "data" / "external" / "tlc" / "taxi_zones.zip"
OUT = ROOT / "data" / "processed" / "dim_taxi_zone_centroids.csv"


def main() -> None:
    if not ZONE_SHAPEFILE.exists():
        raise FileNotFoundError(f"Missing TLC taxi zone shapefile: {ZONE_SHAPEFILE}")

    zones = gpd.read_file(f"zip://{ZONE_SHAPEFILE}!taxi_zones/taxi_zones.shp")
    centroids = zones.to_crs(epsg=4326).copy()
    centroids["centroid"] = zones.to_crs(epsg=2263).centroid.to_crs(epsg=4326)
    centroids["longitude"] = centroids["centroid"].x
    centroids["latitude"] = centroids["centroid"].y
    out = centroids.rename(
        columns={"LocationID": "location_id", "borough": "borough", "zone": "zone"}
    )[["location_id", "borough", "zone", "latitude", "longitude"]]
    out = out.sort_values("location_id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(json.dumps({"path": str(OUT.relative_to(ROOT)), "rows": len(out)}, indent=2))


if __name__ == "__main__":
    main()
