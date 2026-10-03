# IDS 706 - Treasury yields, 2023 to 2025

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without a desktop display.
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "treasury_yields_2023_2025.csv"
OUTPUT_DIR = BASE_DIR / "results"


def load_data(path=DATA_PATH):
    """Read one row per observation date; retain missing yields."""
    data = pd.read_csv(path)
    required = {"date", "yield_2y", "yield_10y"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if data["date"].isna().any():
        raise ValueError("Dates must not be missing.")
    if data["date"].duplicated().any():
        raise ValueError("Duplicate dates need review before analysis.")
    for column in data.columns:
        if column.startswith("yield_"):
            data[column] = pd.to_numeric(data[column], errors="raise")
            if data[column].isin([float("inf"), float("-inf")]).any():
                raise ValueError(f"{column} contains infinite values.")
    return data.sort_values("date").reset_index(drop=True)


def inspect_data(data):
    print("First five rows\n", data.head())
    print("\nRows and columns:", data.shape)
    print("\nColumn information")
    data.info()
    print("\nSummary statistics\n", data.select_dtypes("number").describe())
    print("\nMissing values\n", data.isna().sum())
    print("\nDuplicate rows:", data.duplicated().sum())
    print("Duplicate dates:", data["date"].duplicated().sum())
    changes_bp = data["yield_10y"].dropna().diff() * 100
    print(
        f"Largest absolute 10-year change between observations: "
        f"{changes_bp.abs().max():.2f} bp"
    )
    print("Finite yields are retained; no clipping or winsorization is applied.")


def filter_and_group(data):
    """Return inverted 2023 dates and annual statistics without changing data."""
    valid = data.dropna(subset=["yield_2y", "yield_10y"]).copy()
    valid["spread"] = valid["yield_10y"] - valid["yield_2y"]
    valid["year"] = valid["date"].dt.year
    valid["inverted"] = valid["spread"] < 0
    filtered = valid[(valid["year"] == 2023) & (valid["spread"] < 0)]
    yearly = valid.groupby("year").agg(
        observations=("spread", "count"),
        mean_spread_pp=("spread", "mean"),
        inverted_days=("inverted", "sum"),
    )
    return filtered, yearly


def plot_yields(data, output_path):
    # Leave missing values as gaps in the lines.
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data["date"], data["yield_2y"], color="#bd5b16", label="2-year yield")
    ax.plot(data["date"], data["yield_10y"], color="#205b9d", label="10-year yield")
    ax.fill_between(
        data["date"],
        data["yield_10y"],
        data["yield_2y"],
        where=data["yield_2y"] > data["yield_10y"],
        color="#e7a66b",
        alpha=0.3,
        label="Inverted: 2-year > 10-year",
    )
    ax.set(
        title="U.S. Treasury zero-coupon yields, 2023–2025",
        xlabel="Date",
        ylabel="Annualized yield (%)",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def prepare_model_samples(data):
    # Pair each yield with the next available observation, not calendar day.
    samples = data[["date", "yield_10y"]].dropna().sort_values("date").copy()
    samples["next_yield_10y"] = samples["yield_10y"].shift(-1)
    samples["target_date"] = samples["date"].shift(-1)
    return samples.dropna()


def split_model_samples(samples, cutoff="2025-01-01"):
    """Split on target date, keeping future answers out of training."""
    cutoff = pd.Timestamp(cutoff)
    if pd.isna(cutoff):
        raise ValueError("The cutoff must be a valid date.")
    train = samples[samples["target_date"] < cutoff].copy()
    test = samples[samples["target_date"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Need both training and test samples around the cutoff.")
    return train, test


def train_model(training_samples):
    """Fit using only the samples passed in by the chronological split."""
    model = LinearRegression()
    model.fit(training_samples[["yield_10y"]], training_samples["next_yield_10y"])
    return model


def evaluate_forecasts(test_samples, regression_predictions):
    """Compare both forecasts against the same next-observation targets."""
    if test_samples.empty:
        raise ValueError("Need test samples to evaluate forecasts.")
    if len(regression_predictions) != len(test_samples):
        raise ValueError("Need one prediction for each test sample.")
    predictions = test_samples[["date", "target_date"]].reset_index(drop=True).copy()
    predictions["actual"] = test_samples["next_yield_10y"].to_numpy()
    predictions["regression"] = list(regression_predictions)
    predictions["unchanged_yield"] = test_samples["yield_10y"].to_numpy()
    metrics = []
    for name in ["regression", "unchanged_yield"]:
        mae_pp = mean_absolute_error(predictions["actual"], predictions[name])
        metrics.append({"model": name, "mae_pp": mae_pp, "mae_bp": mae_pp * 100})
    return pd.DataFrame(metrics), predictions


def plot_forecast_comparison(comparison, output_path):
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Linear regression", "Unchanged yield"]
    bars = ax.bar(labels, comparison["mae_bp"], color=["#205b9d", "#bd5b16"])
    ax.bar_label(bars, fmt="%.3f bp", padding=4)
    ax.set(
        title="Next-observation forecast errors in 2025",
        ylabel="Mean absolute error (basis points)",
    )
    ax.set_ylim(0, max(1, comparison["mae_bp"].max() * 1.25))
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(data_path=DATA_PATH, output_dir=OUTPUT_DIR):
    data = load_data(data_path)
    inspect_data(data)
    filtered, yearly = filter_and_group(data)
    print("\nInverted dates in 2023:", len(filtered))
    print(filtered[["date", "yield_2y", "yield_10y", "spread"]].head())
    print("\nYearly summary\n", yearly.round(4))

    samples = prepare_model_samples(data)
    train, test = split_model_samples(samples)
    model = train_model(train)
    comparison, predictions = evaluate_forecasts(
        test, model.predict(test[["yield_10y"]])
    )
    print("\nTraining samples:", len(train), "| Test samples:", len(test))
    print(
        "\nSame test dates for both forecasts\n",
        comparison.to_string(index=False, float_format="{:.6f}".format),
    )
    print("\nFirst five predictions\n", predictions.head().to_string(index=False))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_yields(data, output_dir / "treasury_yields_2023_2025.png")
    plot_forecast_comparison(comparison, output_dir / "forecast_mae.png")
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    print("\nResults saved to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the bundled Treasury yields.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    main(arguments.data, arguments.output_dir)
