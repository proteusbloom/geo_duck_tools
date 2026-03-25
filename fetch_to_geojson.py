"""
Fetches 1,000 rows from NYC Open Data (Socrata SODA API) and saves as GeoJSON.
Uses the native `the_geom` geometry column returned by the API.
Requires: pip install requests geopandas
"""

import requests
import geopandas as gpd

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ID = "5zhs-2jue"
BASE_URL   = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
LIMIT      = 1000
OUT_FILE   = "output.geojson"
APP_TOKEN  = ""   # optional but recommended
# ─────────────────────────────────────────────────────────────────────────────


def fetch_rows() -> list[dict]:
    headers = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}
    params  = {"$limit": LIMIT, "$offset": 0}

    print(f"Fetching {LIMIT} rows from:\n  {BASE_URL}\n")
    resp = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    rows = fetch_rows()
    print(f"✓ Got {len(rows)} rows.")

    gdf = gpd.GeoDataFrame.from_features(
        [{"type": "Feature", "geometry": row["the_geom"], "properties": {k: v for k, v in row.items() if k != "the_geom"}}
         for row in rows if "the_geom" in row],
        crs="EPSG:4326"
    )

    print(f"✓ Built GeoDataFrame with {len(gdf)} features.")
    print(f"\nPreview:\n{gdf.head()}\n")

    gdf.to_file(OUT_FILE, driver="GeoJSON")
    print(f"✅  Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
