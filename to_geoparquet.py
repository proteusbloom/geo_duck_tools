"""
Convert a buildings file (CSV or GeoJSON) to GeoParquet using DuckDB spatial.

Usage:
    python duckdb_fun_tools/to_geoparquet.py <input_file> <output_file>

Examples:
    python duckdb_fun_tools/to_geoparquet.py buildings.csv out.parquet
    python duckdb_fun_tools/to_geoparquet.py buildings.geojson out.parquet
"""

import argparse
import json
import sys
import duckdb

sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to input CSV or GeoJSON file")
    parser.add_argument("output_file", help="Path for output GeoParquet file")
    args = parser.parse_args()

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    if args.input_file.endswith(".geojson") or args.input_file.endswith(".json"):
        con.execute(f"""
            COPY (
                SELECT * FROM ST_Read('{args.input_file}')
            ) TO '{args.output_file}' (FORMAT PARQUET)
        """)
    else:
        # CSV: expects a the_geom column containing GeoJSON strings
        con.execute(f"""
            COPY (
                SELECT
                    * EXCLUDE (the_geom),
                    ST_GeomFromGeoJSON(the_geom) AS geometry
                FROM read_csv_auto('{args.input_file}')
                WHERE the_geom IS NOT NULL
            ) TO '{args.output_file}' (FORMAT PARQUET)
        """)

    print(f"Written to: {args.output_file}\n")

    # --- GeoParquet metadata ---
    print("--- GeoParquet metadata ---")
    meta = con.execute(f"SELECT * FROM parquet_kv_metadata('{args.output_file}')").fetchdf()
    geo_rows = meta[meta["key"].apply(lambda k: bytes(k) == b"geo")]
    if not geo_rows.empty:
        geo_meta = json.loads(bytes(geo_rows["value"].iloc[0]))
        print(f"  version        : {geo_meta.get('version')}")
        print(f"  primary_column : {geo_meta.get('primary_column')}")
        col_info = geo_meta.get("columns", {}).get("geometry", {})
        print(f"  geometry_types : {col_info.get('geometry_types')}")
        print(f"  crs            : {col_info.get('crs', {}).get('id', 'n/a')}")
        print(f"  bbox           : {col_info.get('bbox')}")
    else:
        print("  (no 'geo' key found)")

    # --- Schema ---
    print("\n--- Schema ---")
    schema = con.execute(f"DESCRIBE SELECT * FROM '{args.output_file}'").fetchdf()
    print(schema.to_string(index=False))

    # --- Sample rows ---
    print("\n--- Sample rows (5) ---")
    sample = con.execute(f"""
        SELECT * EXCLUDE (geometry), ST_AsText(geometry) AS geometry
        FROM '{args.output_file}'
        LIMIT 5
    """).fetchdf()
    print(sample.to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()
