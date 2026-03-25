"""
Join a BIN lookup CSV against a buildings reference file.

Usage:
    python fetch_and_join.py <lookup_csv> <buildings_file> [-o OUTPUT] [-f FORMAT]

    lookup_csv:     CSV or GeoJSON with a 'bin' column
    buildings_file: Buildings reference CSV or GeoJSON
"""

import argparse
import sys
import duckdb
import polars as pl
import pyarrow

sys.stdout.reconfigure(encoding="utf-8")


def load_data(path: str) -> pl.DataFrame:
    """Unified loader for CSV, GeoJSON, and JSON files.

    GeoJSON geometry is returned as WKT string via ST_AsText().
    """
    if path.endswith(".geojson") or path.endswith(".json"):
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")
        df = con.execute(
            f"SELECT * EXCLUDE (geom), ST_AsText(geom) AS geom FROM ST_Read('{path}')"
        ).pl()
        con.close()
        return df
    return pl.read_csv(path, infer_schema_length=0)


def join_on_bin(lookup: pl.DataFrame, buildings: pl.DataFrame) -> pl.DataFrame:
    """Left-join lookup onto buildings on the 'bin' column. Prints match stats."""
    if "bin" in lookup.columns:
        lookup = lookup.with_columns(pl.col("bin").cast(pl.Utf8))
    if "bin" in buildings.columns:
        buildings = buildings.with_columns(pl.col("bin").cast(pl.Utf8))

    joined = lookup.join(buildings, on="bin", how="left")

    non_bin_cols = [c for c in buildings.columns if c != "bin"]
    if non_bin_cols:
        matched = joined.filter(
            pl.all_horizontal(pl.col(c).is_not_null() for c in non_bin_cols)
        ).height
    else:
        matched = len(joined)

    print(
        f"Rows: {len(joined)} | Matched: {matched} | Unmatched: {len(joined) - matched}",
        file=sys.stderr,
    )
    return joined


def _enforce_crs(gdf, expected_crs="EPSG:4326"):
    """Raise ValueError if gdf's CRS does not match expected_crs. (Internal use only.)"""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS set.")
    from pyproj import CRS
    if not CRS(gdf.crs).equals(CRS(expected_crs)):
        raise ValueError(f"CRS mismatch: expected {expected_crs}, got {gdf.crs}")


def _transform_crs(gdf, target_crs):
    """Reproject gdf to target_crs and return the reprojected GeoDataFrame. (Internal use only.)"""
    return gdf.to_crs(target_crs)


def write_output(df: pl.DataFrame, output_path: str, fmt: str) -> None:
    """Write joined DataFrame to CSV or GeoJSON."""
    if fmt == "geojson":
        import geopandas as gpd
        from shapely import wkt as shapely_wkt

        geom_col = next(
            (c for c in df.columns if c in ("geom", "geometry", "the_geom")), None
        )
        if geom_col is None:
            raise ValueError(
                "No geometry column found (expected 'geom', 'geometry', or 'the_geom')."
            )

        pandas_df = df.to_pandas()
        pandas_df["geometry"] = gpd.GeoSeries.from_wkt(pandas_df[geom_col])
        if geom_col != "geometry":
            pandas_df = pandas_df.drop(columns=[geom_col])

        gdf = gpd.GeoDataFrame(pandas_df, geometry="geometry", crs="EPSG:4326")
        gdf.to_file(output_path, driver="GeoJSON")
        print(f"GeoJSON written to {output_path}", file=sys.stderr)
    else:
        df.write_csv(output_path)
        print(f"CSV written to {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Join a BIN lookup file against a buildings reference file."
    )
    parser.add_argument("lookup_csv", help="CSV or GeoJSON with a 'bin' column")
    parser.add_argument("buildings_file", help="Buildings reference CSV or GeoJSON")
    parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    parser.add_argument(
        "--format",
        "-f",
        choices=["csv", "geojson"],
        default="csv",
        help="Output format: csv (default) or geojson",
    )
    # CRS flags — defined but commented out; use _enforce_crs/_transform_crs internally
    # parser.add_argument("--enforce-crs", default="EPSG:4326", help="Expected input CRS")
    # parser.add_argument("--target-crs", help="Reproject output to this CRS")
    args = parser.parse_args()

    lookup = load_data(args.lookup_csv)
    buildings = load_data(args.buildings_file)

    joined = join_on_bin(lookup, buildings)

    if args.output:
        write_output(joined, args.output, args.format)
    else:
        print(joined)


if __name__ == "__main__":
    main()
