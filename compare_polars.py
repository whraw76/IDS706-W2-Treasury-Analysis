# Repeat the Treasury filtering and grouping in Pandas and Polars.

from pathlib import Path
from statistics import median
from timeit import repeat

import pandas as pd
import polars as pl

DATA_PATH = Path(__file__).resolve().parent / "data" / "treasury_yields_2023_2025.csv"


def pandas_analysis(data):
    valid = data.dropna(subset=["yield_2y", "yield_10y"]).copy()
    valid["spread"] = valid["yield_10y"] - valid["yield_2y"]
    valid["year"] = valid["date"].dt.year
    valid["inverted"] = valid["spread"] < 0
    filtered = valid[(valid["year"] == 2023) & (valid["spread"] < 0)]
    yearly = valid.groupby("year").agg(
        observations=("spread", "count"),
        mean_spread_pp=("spread", "mean"),
        inverted_days=("inverted", "sum"),
    ).reset_index()
    return filtered, yearly


def polars_analysis(data):
    valid = data.drop_nulls(subset=["yield_2y", "yield_10y"]).with_columns(
        (pl.col("yield_10y") - pl.col("yield_2y")).alias("spread"),
        pl.col("date").dt.year().alias("year"),
    ).with_columns((pl.col("spread") < 0).alias("inverted"))
    filtered = valid.filter((pl.col("year") == 2023) & (pl.col("spread") < 0))
    yearly = valid.group_by("year").agg(
        pl.col("spread").count().alias("observations"),
        pl.col("spread").mean().alias("mean_spread_pp"),
        pl.col("inverted").sum().alias("inverted_days"),
    ).sort("year")
    return filtered, yearly


if __name__ == "__main__":
    pandas_data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    polars_data = pl.read_csv(DATA_PATH, try_parse_dates=True)
    pd_filtered, pd_yearly = pandas_analysis(pandas_data)
    pl_filtered, pl_yearly = polars_analysis(polars_data)

    # Check that both versions give the same results.
    assert pd_filtered["date"].dt.date.tolist() == pl_filtered["date"].to_list()
    pd.testing.assert_frame_equal(
        pd_yearly, pd.DataFrame(pl_yearly.to_dicts()),
        check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10,
    )
    print(f"Results match: {len(pd_filtered)} filtered dates and the yearly summary")
    print(pd_yearly.round(4).to_string(index=False))
    print(f"\nPandas {pd.__version__} | Polars {pl.__version__}")
    print("Timing: 5 batches of 100 runs, median time per run.")
    print("Timed: dropping missing values, new columns, filtering, grouping, sorting.")
    print("CSV loading, result checks, printing, plotting, and ML are not timed.")
    for name, operation, data in [
        ("Pandas", pandas_analysis, pandas_data),
        ("Polars", polars_analysis, polars_data),
    ]:
        operation(data)  # Warm up before timing.
        timings = repeat(lambda: operation(data), number=100, repeat=5)
        print(f"{name}: {median(timings) / 100 * 1000:.3f} ms/run")
