"""IDS 706: basic Treasury yield analysis with Pandas and linear regression."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save the chart without opening a separate window.
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "treasury_yields_2023_2025.csv"
PLOT_PATH = BASE_DIR / "figures" / "treasury_yields_2023_2025.png"


def inspect_data(data):
    """Inspect the source before handling missing values."""
    print("FIRST FIVE ROWS\n", data.head())
    print("\nSHAPE:", data.shape)
    print("\nDATA TYPES AND NON-MISSING COUNTS")
    data.info()
    print("\nSUMMARY STATISTICS\n", data.select_dtypes("number").describe())
    print("\nMISSING VALUES\n", data.isna().sum())
    print("\nDUPLICATE ROWS:", data.duplicated().sum())
    print("DUPLICATE DATES:", data["date"].duplicated().sum())


def filter_and_group(data):
    """Find inverted dates and compare their frequency across years."""
    valid = data.dropna(subset=["yield_2y", "yield_10y"]).copy()
    valid["spread"] = valid["yield_10y"] - valid["yield_2y"]
    valid["year"] = valid["date"].dt.year
    valid["inverted"] = valid["spread"] < 0

    # Both conditions must be true: the year is 2023 and the spread is negative.
    filtered = valid[(valid["year"] == 2023) & (valid["spread"] < 0)]
    print("\nINVERTED DATES IN 2023:", len(filtered))
    print(filtered[["date", "yield_2y", "yield_10y", "spread"]].head())

    yearly = valid.groupby("year").agg(
        observations=("spread", "count"),
        mean_spread_pp=("spread", "mean"),
        inverted_days=("inverted", "sum"),
    )
    print("\nYEARLY SUMMARY\n", yearly.round(4))
    return yearly


def plot_yields(data):
    """Compare the two yields over time; missing values remain gaps."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data["date"], data["yield_2y"], color="#bd5b16", label="2-year yield")
    ax.plot(data["date"], data["yield_10y"], color="#205b9d", label="10-year yield")
    ax.fill_between(
        data["date"], data["yield_10y"], data["yield_2y"],
        where=data["yield_2y"] > data["yield_10y"],
        color="#e7a66b", alpha=0.3, label="Inverted: 2-year > 10-year",
    )
    ax.set(title="U.S. Treasury zero-coupon yields, 2023–2025",
           xlabel="Date", ylabel="Annualized yield (%)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print("\nCHART SAVED:", PLOT_PATH)


def train_model(data):
    """Use the current 10-year yield to predict its next observed value."""
    samples = data[["date", "yield_10y"]].dropna().sort_values("date").copy()
    samples["next_yield_10y"] = samples["yield_10y"].shift(-1)
    samples["target_date"] = samples["date"].shift(-1)
    samples = samples.dropna()  # The last date has no next observation.

    # Split by the predicted date so 2025 outcomes are kept out of training.
    train = samples[samples["target_date"] < "2025-01-01"]
    test = samples[samples["target_date"] >= "2025-01-01"]
    model = LinearRegression()
    model.fit(train[["yield_10y"]], train["next_yield_10y"])
    predictions = model.predict(test[["yield_10y"]])
    mae = mean_absolute_error(test["next_yield_10y"], predictions)

    print("\nLINEAR REGRESSION: CURRENT 10-YEAR YIELD -> NEXT OBSERVED YIELD")
    print("Training samples:", len(train), "| Test samples:", len(test))
    print(f"Test MAE: {mae:.4f} percentage points")
    examples = pd.DataFrame({"target_date": test["target_date"],
                             "actual": test["next_yield_10y"], "predicted": predictions})
    print(examples.head().to_string(index=False, float_format="{:.4f}".format))
    return model, mae


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")
    inspect_data(data)
    filter_and_group(data)
    plot_yields(data)
    train_model(data)
