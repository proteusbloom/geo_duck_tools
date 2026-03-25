"""
Compute the centroid (lat/lon) of building polygons identified by BIN.

Usage:
    python bin_centroid.py <csv_path> (--bin BIN | --bins-csv PATH)
                           [--geom-col COL] [--input-crs EPSG] [--output PATH]
"""

import argparse
import sys
from pathlib import Path

import duckdb
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

_GEOM_COL_NAMES = {"geom", "geometry", "the_geom", "shape", "wkt", "geom_wkt"}


def _to_duckdb_path(p) -> str:
    """Convert a path to a forward-slash string safe for DuckDB SQL."""
    return str(p).replace("\\", "/")


def _detect_geom_col(columns: list[str]) -> str:
    """Find geometry column by common names (case-insensitive)."""
    lower_map = {c.lower(): c for c in columns}
    col = next((lower_map[k] for k in _GEOM_COL_NAMES if k in lower_map), None)
    if col is None:
        raise ValueError(
            f"Cannot auto-detect geometry column. Tried: {sorted(_GEOM_COL_NAMES)}. "
            f"Columns found: {columns}. Use --geom-col to specify."
        )
    return col


def _check_duplicate_bins(df: pl.DataFrame) -> None:
    """Warn if any BIN appears more than once. Remove this function when no longer needed."""
    dupes = (
        df.group_by("bin")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") >= 2)
    )
    if len(dupes) > 0:
        print("Warning: duplicate BINs found in input CSV:", file=sys.stderr)
        print(dupes, file=sys.stderr)


def _build_connection() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with the spatial extension loaded."""
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    return con


def compute_centroids(
    con: duckdb.DuckDBPyConnection,
    csv_path,
    geom_col: str,
    bins: list[str],
    input_crs: str = "EPSG:4326",
) -> pl.DataFrame:
    """
    Compute centroids for buildings identified by BIN.

    Returns a Polars DataFrame with columns: bin, lat, lon.
    """
    csv_p = _to_duckdb_path(csv_path)

    # Detect geometry format by peeking at the first non-null value
    peek_sql = f"""
    SELECT {geom_col}
    FROM read_csv_auto('{csv_p}')
    WHERE {geom_col} IS NOT NULL
    LIMIT 1
    """
    peek_result = con.execute(peek_sql).fetchone()
    if peek_result and str(peek_result[0]).strip().startswith("{"):
        geom_fn = "ST_GeomFromGeoJSON"
    else:
        geom_fn = "ST_GeomFromText"

    # Build reprojected geometry expression
    parsed_geom = f"{geom_fn}({geom_col})"
    if input_crs.upper() != "EPSG:4326":
        reprojected_geom = f"ST_Transform({parsed_geom}, '{input_crs}', 'EPSG:4326')"
    else:
        reprojected_geom = parsed_geom

    # Build bins placeholder
    bins_placeholder = ", ".join(f"'{b}'" for b in bins)

    sql = f"""
    SELECT
        bin,
        ST_Y(ST_Centroid({reprojected_geom})) AS lat,
        ST_X(ST_Centroid({reprojected_geom})) AS lon
    FROM read_csv_auto('{csv_p}')
    WHERE CAST(bin AS VARCHAR) IN ({bins_placeholder})
    """

    return con.execute(sql).pl()


def _load_bins_csv(bins_csv_path) -> list[str]:
    """Load BINs from a single-column CSV, handling optional header."""
    # Read with no header assumption first
    raw = pl.read_csv(bins_csv_path, has_header=False, infer_schema_length=0)
    first_val = raw[raw.columns[0]][0]
    # If first value is not numeric, it's a header — skip it
    try:
        int(first_val)
        has_header = False
    except (ValueError, TypeError):
        has_header = True

    df = pl.read_csv(bins_csv_path, has_header=has_header, infer_schema_length=0)
    return df[df.columns[0]].to_list()


def main():
    parser = argparse.ArgumentParser(
        description="Compute centroids of building polygons identified by BIN."
    )
    parser.add_argument("csv_path", help="Input CSV with 'bin' and geometry columns")

    bin_group = parser.add_mutually_exclusive_group(required=True)
    bin_group.add_argument("--bin", help="Single BIN value to look up")
    bin_group.add_argument("--bins-csv", metavar="PATH", help="CSV of BINs to look up")

    parser.add_argument(
        "--geom-col",
        default=None,
        help="Geometry column name (auto-detected if omitted)",
    )
    parser.add_argument(
        "--input-crs",
        default="EPSG:4326",
        metavar="EPSG",
        help="CRS of the input geometry (default: EPSG:4326)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: bin_centroids.csv next to input; "
             "for --bin mode, prints to stdout if omitted)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Load input CSV to check for duplicate BINs and detect geom col
    input_df = pl.read_csv(csv_path, infer_schema_length=0)

    if "bin" not in input_df.columns:
        print("Error: input CSV has no 'bin' column.", file=sys.stderr)
        sys.exit(1)

    _check_duplicate_bins(input_df)

    # Resolve geometry column
    if args.geom_col:
        if args.geom_col not in input_df.columns:
            print(
                f"Error: geometry column {args.geom_col!r} not found in CSV. "
                f"Columns: {input_df.columns}",
                file=sys.stderr,
            )
            sys.exit(1)
        geom_col = args.geom_col
    else:
        try:
            geom_col = _detect_geom_col(input_df.columns)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Resolve BINs to look up
    single_mode = args.bin is not None
    if single_mode:
        bins = [str(args.bin)]
    else:
        bins_csv_path = Path(args.bins_csv)
        if not bins_csv_path.exists():
            print(f"Error: bins CSV not found: {bins_csv_path}", file=sys.stderr)
            sys.exit(1)
        bins = _load_bins_csv(bins_csv_path)

    con = _build_connection()
    try:
        result = compute_centroids(con, csv_path, geom_col, bins, args.input_crs)
    finally:
        con.close()

    if single_mode and args.output is None:
        # Print tab-separated to stdout
        if len(result) == 0:
            print(f"No result found for BIN {args.bin}", file=sys.stderr)
        else:
            print("bin\tlat\tlon")
            for row in result.iter_rows():
                print(f"{row[0]}\t{row[1]}\t{row[2]}")
    else:
        # Write CSV
        output_path = Path(args.output) if args.output else csv_path.parent / "bin_centroids.csv"
        result.write_csv(output_path)
        print(f"Wrote {len(result)} rows to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
