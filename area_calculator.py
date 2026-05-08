#!/usr/bin/env python3
"""
area_calculator.py — Polygon area calculator using DuckDB spatial + Polars.
Reprojects polygon geometries to EPSG:2263 and outputs area in square feet.

Usage:  python area_calculator.py <input_file> <output_csv> [--crs EPSG_CODE]
Input:  .geojson, .csv (WKT or EWKT geometry column), or .txt (one WKT/EWKT geometry per line)
Output: CSV with columns id, area_sqft, area_acres
"""

import argparse
import sys
from pathlib import Path
import duckdb
import polars as pl

ID_NAMES = {"id", "fid", "ogc_fid", "gid", "objectid"}
GEOM_NAMES = {"geometry", "geom", "wkt", "shape", "the_geom", "wkb_geometry"}

# DuckDB / PROJ treats EPSG geographic CRS codes with authority axis order (lat, lon).
# WKT stores coordinates as (x=lon, y=lat), so geographic CRS must be specified with
# an explicit lon/lat-forcing string to avoid silently swapped axes and wrong areas.
_GEOGRAPHIC_CRS_STRINGS = {
    4326: "OGC:CRS84",
    4269: "+proj=longlat +datum=NAD83 +no_defs",
}


def crs_transform_string(epsg: int) -> str:
    return _GEOGRAPHIC_CRS_STRINGS.get(epsg, f"EPSG:{epsg}")


def infer_crs_from_bbox(minx: float, maxx: float, miny: float, maxy: float) -> int | None:
    if -180 <= minx <= 0 and 24 <= miny <= 50:
        return 4326
    if 500_000 <= minx <= 700_000 and 4_000_000 <= miny <= 5_500_000:
        return 32618
    if 800_000 <= minx <= 1_100_000 and 100_000 <= miny <= 300_000:
        return 2263
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate polygon areas in sq ft using EPSG:2263."
    )
    parser.add_argument("input_file", help="Input .geojson, .csv, or .txt file")
    parser.add_argument("output_csv", help="Output CSV file path")
    parser.add_argument(
        "--crs", type=int, default=None,
        help="Override source CRS EPSG code (e.g. --crs 4326)"
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    ext = input_path.suffix.lower()

    if ext not in (".csv", ".geojson", ".txt"):
        sys.exit(f"Error: Only .csv, .geojson, and .txt files are supported. Got: '{ext}'")
    if not input_path.exists():
        sys.exit(f"Error: File not found: {input_path}")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    # DuckDB requires forward slashes on Windows
    safe_path = str(input_path.resolve()).replace("\\", "/")
    ewkt_srid: int | None = None

    if ext == ".geojson":
        con.execute(f"CREATE OR REPLACE VIEW src AS SELECT * FROM ST_Read('{safe_path}');")

    elif ext == ".txt":
        lines = [
            ln.strip()
            for ln in input_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if not lines:
            sys.exit("Error: .txt file is empty or contains no geometry.")

        # Detect EWKT ("SRID=4326;MULTIPOLYGON(...)") on the first line
        if lines[0].upper().startswith("SRID="):
            try:
                ewkt_srid = int(lines[0].split(";")[0].split("=")[1])
                lines = [ln.split(";", 1)[1] if ";" in ln else ln for ln in lines]
                print(f"Detected EWKT format — embedded SRID: {ewkt_srid}")
            except (IndexError, ValueError):
                pass

        # Parameterized inserts avoid any quoting issues inside WKT strings
        con.execute("CREATE OR REPLACE TABLE _wkt_input (wkt_text VARCHAR)")
        con.executemany("INSERT INTO _wkt_input VALUES (?)", [(ln,) for ln in lines])
        con.execute(
            "CREATE OR REPLACE VIEW src AS "
            "SELECT ST_GeomFromText(wkt_text) AS __geom__ FROM _wkt_input"
        )

    else:  # .csv
        # Inspect raw column names before parsing geometry
        raw_cols = con.execute(
            f"DESCRIBE SELECT * FROM read_csv('{safe_path}', AUTO_DETECT=TRUE)"
        ).fetchall()
        col_names = [r[0] for r in raw_cols]

        wkt_col = next((c for c in col_names if c.lower() in GEOM_NAMES), None)
        if wkt_col is None:
            sys.exit(
                f"Error: No geometry column found in CSV.\n"
                f"Columns present: {col_names}\n"
                f"Expected a column named one of: {sorted(GEOM_NAMES)}"
            )

        # Detect EWKT format: "SRID=4326;POLYGON(...)"
        # ST_GeomFromText does not support EWKT — strip the prefix if present
        sample = con.execute(
            f"SELECT {wkt_col} FROM read_csv('{safe_path}', AUTO_DETECT=TRUE) "
            f"WHERE {wkt_col} IS NOT NULL LIMIT 1"
        ).fetchone()

        if sample and isinstance(sample[0], str) and sample[0].upper().startswith("SRID="):
            try:
                ewkt_srid = int(sample[0].split(";")[0].split("=")[1])
                geom_expr = (
                    f"ST_GeomFromText(regexp_replace({wkt_col}, '^SRID=\\d+;', ''))"
                )
                print(f"Detected EWKT format — embedded SRID: {ewkt_srid}")
            except (IndexError, ValueError):
                geom_expr = f"ST_GeomFromText({wkt_col})"
        else:
            geom_expr = f"ST_GeomFromText({wkt_col})"

        con.execute(
            f"CREATE OR REPLACE VIEW src AS "
            f"SELECT *, {geom_expr} AS __geom__ "
            f"FROM read_csv('{safe_path}', AUTO_DETECT=TRUE);"
        )

    # --- Identify GEOMETRY-typed column ---
    desc = con.execute("DESCRIBE src").fetchall()
    geom_cols = [r[0] for r in desc if r[1].upper() == "GEOMETRY"]

    if not geom_cols:
        sys.exit(f"Error: No GEOMETRY column found in {input_path}.")
    if len(geom_cols) > 1:
        print(f"Warning: Multiple geometry columns: {geom_cols}. Using '{geom_cols[0]}'.")
    geom_col = geom_cols[0]

    # --- Identify ID column, or fall back to row numbers ---
    id_col = next((r[0] for r in desc if r[0].lower() in ID_NAMES), None)
    if id_col:
        id_expr = id_col
    else:
        id_expr = "ROW_NUMBER() OVER ()"
        print("No ID column found -- using sequential row numbers.")

    # --- Warn on NULL geometries (includes malformed WKT on CSV path) ---
    null_count = con.execute(
        f"SELECT COUNT(*) FROM src WHERE {geom_col} IS NULL"
    ).fetchone()[0]
    if null_count:
        print(f"Warning: {null_count} NULL/invalid geometries will be skipped.")

    # --- Determine source CRS ---
    if args.crs:
        src_crs = args.crs
        print(f"Using user-specified CRS: EPSG:{src_crs}")
    elif ewkt_srid:
        src_crs = ewkt_srid
    else:
        print("Inferring CRS from bounding box...")
        row = con.execute(
            f"SELECT MIN(ST_XMin({geom_col})), MAX(ST_XMax({geom_col})), "
            f"MIN(ST_YMin({geom_col})), MAX(ST_YMax({geom_col})) "
            f"FROM src WHERE {geom_col} IS NOT NULL"
        ).fetchone()
        minx, maxx, miny, maxy = row
        src_crs = infer_crs_from_bbox(minx, maxx, miny, maxy)
        if src_crs is None:
            sys.exit(
                f"Error: Cannot infer CRS.\n"
                f"Bounding box: x=[{minx:.4f}, {maxx:.4f}], "
                f"y=[{miny:.4f}, {maxy:.4f}]\n"
                "Use --crs <EPSG_CODE> to specify the source CRS manually."
            )
        print(f"Inferred EPSG:{src_crs} from bounding box.")

    # --- Build area query ---
    # CTE avoids computing ST_Transform twice (for sqft and acres)
    if src_crs == 2263:
        area_expr = f"ST_Area({geom_col})"
    else:
        src_crs_str = crs_transform_string(src_crs)
        area_expr = (
            f"ST_Area(ST_Transform({geom_col}, '{src_crs_str}', 'EPSG:2263'))"
        )

    sql = f"""
        WITH base AS (
            SELECT
                {id_expr} AS id,
                {area_expr} AS area_sqft
            FROM src
            WHERE {geom_col} IS NOT NULL
        )
        SELECT
            id,
            area_sqft,
            area_sqft / 43560.0 AS area_acres
        FROM base
    """

    df = con.execute(sql).pl()
    df.write_csv(args.output_csv)

    area = df["area_sqft"]
    print(f"\nWritten {len(df):,} rows -> {args.output_csv}")
    print(f"area_sqft  min: {area.min():>14,.1f}")
    print(f"           max: {area.max():>14,.1f}")
    print(f"          mean: {area.mean():>14,.1f}")
    print("\nExpected ranges (NYC building footprints):")
    print("  rowhouse 1,000-2,500  |  commercial 5,000-20,000  |  city block ~150,000")


if __name__ == "__main__":
    main()
