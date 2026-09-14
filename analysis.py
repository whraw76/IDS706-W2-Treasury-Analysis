# IDS 706 - Treasury yields, 2023 to 2025

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save the plot to a file.
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "treasury_yields_2023_2025.csv"
PLOT_PATH = BASE_DIR / "figures" / "treasury_yields_2023_2025.png"


def inspect_data(data):
    print("First five rows\n", data.head())
    print("\nRows and columns:", data.shape)
    print("\nColumn information")
    data.info()
    print("\nSummary statistics\n", data.select_dtypes("number").describe())
    print("\nMissing values\n", data.isna().sum())
    print("\nDuplicate rows:", data.duplicated().sum())
    print("Duplicate dates:", data["date"].duplicated().sum())


def filter_and_group(data):
    valid = data.dropna(subset=["yield_2y", "yield_10y"]).copy()
    valid["spread"] = valid["yield_10y"] - valid["yield_2y"]
    valid["year"] = valid["date"].dt.year
    valid["inverted"] = valid["spread"] < 0

    # Inverted dates in 2023.
    filtered = valid[(valid["year"] == 2023) & (valid["spread"] < 0)]
    print("\nInverted dates in 2023:", len(filtered))
    print(filtered[["date", "yield_2y", "yield_10y", "spread"]].head())

    yearly = valid.groupby("year").agg(
        observations=("spread", "count"),
        mean_spread_pp=("spread", "mean"),
        inverted_days=("inverted", "sum"),
    )
    print("\nYearly summary\n", yearly.round(4))
    return yearly


def plot_yields(data):
    # Leave missing values as gaps in the lines.
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
    print("\nPlot saved to:", PLOT_PATH)


def train_model(data):
    # Pair each yield with the next available observation.
    samples = data[["date", "yield_10y"]].dropna().sort_values("date").copy()
    samples["next_yield_10y"] = samples["yield_10y"].shift(-1)
    samples["target_date"] = samples["date"].shift(-1)
    samples = samples.dropna()  # No target for the final row.

    # Keep all 2025 target values in the test set.
    train = samples[samples["target_date"] < "2025-01-01"]
    test = samples[samples["target_date"] >= "2025-01-01"]
    model = LinearRegression()
    model.fit(train[["yield_10y"]], train["next_yield_10y"])
    predictions = model.predict(test[["yield_10y"]])
    mae = mean_absolute_error(test["next_yield_10y"], predictions)

    print("\nLinear regression: current 10-year yield -> next observed yield")
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
