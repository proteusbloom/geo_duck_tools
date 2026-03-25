"""
Spatially join a CSV of lat/lon coordinates to NYC building footprints.

Each input row is matched to the building polygon it falls within.
A ``neighbour_bins`` column lists BINs of buildings that touch the matched
polygon — useful as a soft human-error check when a point was clicked near
a boundary.

Usage:
    python point_in_polygon.py <csv_path> [--lat-col LAT] [--lon-col LON]
                               [--geojson PATH] [--output PATH]
"""

import argparse
import sys
from pathlib import Path

import duckdb
import pyarrow
import pyarrow.compute as pc
import pyarrow.csv as pa_csv

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_GEOJSON = Path(__file__).parent / "output.geojson"

_LAT_NAMES = {"lat", "latitude", "lat_dd", "y"}
_LON_NAMES = {"lon", "lng", "longitude", "long", "lon_dd", "x"}


def _to_duckdb_path(p) -> str:
    """Convert a path to a forward-slash string safe for DuckDB SQL."""
    return str(p).replace("\\", "/")


def _detect_lat_lon_cols(columns: list[str]) -> tuple[str, str]:
    """Auto-detect latitude and longitude column names (case-insensitive)."""
    lower_map = {c.lower(): c for c in columns}
    lat_col = next((lower_map[k] for k in _LAT_NAMES if k in lower_map), None)
    lon_col = next((lower_map[k] for k in _LON_NAMES if k in lower_map), None)
    if lat_col is None:
        raise ValueError(
            f"Cannot auto-detect latitude column. Tried: {sorted(_LAT_NAMES)}. "
            f"Columns found: {columns}. Use --lat-col to specify."
        )
    if lon_col is None:
        raise ValueError(
            f"Cannot auto-detect longitude column. Tried: {sorted(_LON_NAMES)}. "
            f"Columns found: {columns}. Use --lon-col to specify."
        )
    return lat_col, lon_col


def _build_connection() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with the spatial extension loaded."""
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    return con


def find_polygon(
    con: duckdb.DuckDBPyConnection,
    csv_path,
    geojson_path,
    lat_col: str,
    lon_col: str,
) -> pyarrow.Table:
    """
    Spatially join ``csv_path`` coordinates to building polygons in ``geojson_path``.

    Parameters
    ----------
    con:
        An open DuckDB connection with the spatial extension loaded.
    csv_path:
        Path to the input CSV containing coordinate columns.
    geojson_path:
        Path to the buildings GeoJSON file (default: output.geojson next to script).
    lat_col:
        Name of the latitude column in the CSV.
    lon_col:
        Name of the longitude column in the CSV.

    Returns
    -------
    pyarrow.Table with all CSV columns plus building attributes and ``neighbour_bins``.
    Unmatched rows have NULL building attributes and an empty ``neighbour_bins`` list.
    """
    csv_p = _to_duckdb_path(csv_path)
    geo_p = _to_duckdb_path(geojson_path)

    sql = f"""
    WITH pts AS (
        SELECT *, row_number() OVER () AS _row_id
        FROM read_csv_auto('{csv_p}')
    ),
    bldgs AS (
        SELECT bin, doitt_id, height_roof, construction_year,
               base_bbl, shape_area, ground_elevation,
               last_status_type, geom_source, geom
        FROM ST_Read('{geo_p}')
    ),
    neighbours AS (
        SELECT p._row_id,
               LIST(n.bin ORDER BY n.bin) FILTER (WHERE n.bin IS NOT NULL)
                   AS neighbour_bins
        FROM pts p
        JOIN bldgs m ON ST_Within(ST_Point(p.{lon_col}, p.{lat_col}), m.geom)
        LEFT JOIN bldgs n ON ST_Touches(m.geom, n.geom) AND n.bin <> m.bin
        GROUP BY p._row_id
    )
    SELECT p.* EXCLUDE (_row_id),
           b.bin, b.doitt_id, b.height_roof, b.construction_year,
           b.base_bbl, b.shape_area, b.ground_elevation,
           b.last_status_type, b.geom_source,
           COALESCE(nb.neighbour_bins, []) AS neighbour_bins
    FROM pts p
    LEFT JOIN bldgs b ON ST_Within(ST_Point(p.{lon_col}, p.{lat_col}), b.geom)
    LEFT JOIN neighbours nb USING (_row_id)
    ORDER BY p._row_id
    """

    result = con.execute(sql).arrow()
    if hasattr(result, "read_all"):
        result = result.read_all()
    return result


def _print_summary(table: pyarrow.Table) -> None:
    """Print match statistics to stderr."""
    n_total = len(table)

    if "bin" in table.schema.names:
        bin_col = table.column("bin")
        n_matched = bin_col.null_count
        n_matched = n_total - bin_col.null_count
    else:
        n_matched = 0

    n_unmatched = n_total - n_matched

    if "neighbour_bins" in table.schema.names and n_matched > 0:
        nb_col = table.column("neighbour_bins")
        lengths = pc.list_value_length(nb_col)
        # Only average over matched rows (where bin is not null)
        matched_mask = pc.invert(pc.is_null(table.column("bin")))
        matched_lengths = pc.filter(lengths, matched_mask)
        avg_nb = pc.mean(matched_lengths).as_py() if len(matched_lengths) > 0 else 0.0
    else:
        avg_nb = 0.0

    print(
        f"Total: {n_total} | Matched: {n_matched} | Unmatched: {n_unmatched} "
        f"| Avg neighbours (matched): {avg_nb:.1f}",
        file=sys.stderr,
    )


def _write_csv(table: pyarrow.Table, output_path) -> None:
    """
    Write the result table to CSV.

    The ``neighbour_bins`` list column is serialised as a pipe-delimited string.
    """
    if "neighbour_bins" in table.schema.names:
        nb_idx = table.schema.get_field_index("neighbour_bins")
        nb_col = table.column("neighbour_bins")

        # Serialise each list to a pipe-delimited string using Python
        def _join_list(val):
            if val is None:
                return ""
            return "|".join(str(v) for v in val)

        nb_joined = pyarrow.array(
            [_join_list(nb_col[i].as_py()) for i in range(len(nb_col))],
            type=pyarrow.string(),
        )
        table = table.set_column(nb_idx, "neighbour_bins", nb_joined)

    pa_csv.write_csv(table, str(output_path))
    print(f"CSV written to {output_path}", file=sys.stderr)


def _peek_csv_columns(csv_path) -> list[str]:
    """Read just the header of the CSV to get column names."""
    opts = pa_csv.ReadOptions(block_size=65536)
    with pa_csv.open_csv(str(csv_path), read_options=opts) as reader:
        batch = reader.read_next_batch()
    return batch.schema.names


def main():
    parser = argparse.ArgumentParser(
        description="Spatially join CSV coordinates to NYC building footprints."
    )
    parser.add_argument("csv_path", help="Input CSV with coordinate columns")
    parser.add_argument(
        "--lat-col",
        default=None,
        help="Latitude column name (auto-detected if omitted)",
    )
    parser.add_argument(
        "--lon-col",
        default=None,
        help="Longitude column name (auto-detected if omitted)",
    )
    parser.add_argument(
        "--geojson",
        default=None,
        help="Buildings GeoJSON path (default: output.geojson next to script)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: point_in_polygon_results.csv next to input)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    geojson_path = Path(args.geojson) if args.geojson else DEFAULT_GEOJSON

    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if not geojson_path.exists():
        print(f"Error: GeoJSON not found: {geojson_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = csv_path.parent / "point_in_polygon_results.csv"

    # Auto-detect lat/lon columns if not provided
    if args.lat_col is None or args.lon_col is None:
        columns = _peek_csv_columns(csv_path)
        lat_col, lon_col = _detect_lat_lon_cols(columns)
        if args.lat_col:
            lat_col = args.lat_col
        if args.lon_col:
            lon_col = args.lon_col
    else:
        lat_col = args.lat_col
        lon_col = args.lon_col

    print(f"Using lat_col={lat_col!r}, lon_col={lon_col!r}", file=sys.stderr)

    con = _build_connection()
    try:
        table = find_polygon(con, csv_path, geojson_path, lat_col, lon_col)
    finally:
        con.close()

    _print_summary(table)
    _write_csv(table, output_path)


if __name__ == "__main__":
    main()
