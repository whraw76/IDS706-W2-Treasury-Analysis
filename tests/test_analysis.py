from datetime import datetime

import pandas as pd
import polars as pl
import pytest

import analysis
from analysis import (
    evaluate_forecasts,
    filter_and_group,
    load_data,
    prepare_model_samples,
    split_model_samples,
    train_model,
)
from compare_polars import pandas_analysis, polars_analysis


@pytest.fixture
def small_dataset():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-12-27",
                    "2023-12-28",
                    "2023-12-29",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                ]
            ),
            "yield_1y": [None, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            "yield_2y": [4.0, 4.0, 3.0, 3.0, 5.0, None, 4.0],
            "yield_10y": [3.0, 4.0, 4.0, 4.0, 3.0, 5.0, None],
        }
    )


def test_load_data_sorts_dates_and_keeps_missing_values(tmp_path):
    path = tmp_path / "yields.csv"
    path.write_text("date,yield_2y,yield_10y\n2024-01-03,4.2,\n2024-01-02,4.0,3.8\n")

    data = load_data(path)

    assert data["date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert data.loc[0, "yield_10y"] == pytest.approx(3.8)
    assert pd.isna(data.loc[1, "yield_10y"])
    assert len(data) == 2


def test_load_data_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "missing.csv")


def test_filter_and_group_handles_missing_and_zero_spreads(small_dataset):
    original = small_dataset.copy(deep=True)

    filtered, yearly = filter_and_group(small_dataset)

    assert filtered["date"].tolist() == [pd.Timestamp("2023-12-27")]
    assert filtered["spread"].tolist() == pytest.approx([-1.0])
    assert yearly.loc[2023, "observations"] == 3
    assert yearly.loc[2023, "mean_spread_pp"] == pytest.approx(0.0)
    assert yearly.loc[2023, "inverted_days"] == 1  # A zero spread is not inverted.
    assert yearly.loc[2024, "observations"] == 2
    assert yearly.loc[2024, "mean_spread_pp"] == pytest.approx(-0.5)
    assert (
        yearly.loc[2024, "inverted_days"] == 1
    )  # Inverted, but outside the filter year.
    pd.testing.assert_frame_equal(small_dataset, original)


def test_plot_uses_both_yields_and_keeps_missing_gaps(
    small_dataset, tmp_path, monkeypatch
):
    # Keep access to the real axes so we can check the plotted data.
    fig, ax = analysis.plt.subplots()
    monkeypatch.setattr(analysis.plt, "subplots", lambda **kwargs: (fig, ax))
    analysis.plot_yields(small_dataset, tmp_path / "yields.png")

    lines = {line.get_label(): line for line in ax.get_lines()}
    assert set(lines) == {"2-year yield", "10-year yield"}
    for label, column in [("2-year yield", "yield_2y"), ("10-year yield", "yield_10y")]:
        assert (
            pd.to_datetime(lines[label].get_xdata()).tolist()
            == small_dataset["date"].tolist()
        )
        pd.testing.assert_series_equal(
            pd.Series(lines[label].get_ydata()),
            small_dataset[column],
            check_names=False,
        )
    assert ax.get_ylabel() == "Annualized yield (%)"


def test_model_samples_use_next_observation_after_missing_dates():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-03", "2024-12-31", "2025-01-02", "2024-12-30"]
            ),
            "yield_10y": [4.3, 4.1, None, 4.0],
        }
    )

    samples = prepare_model_samples(data)

    assert samples["date"].tolist() == [
        pd.Timestamp("2024-12-30"),
        pd.Timestamp("2024-12-31"),
    ]
    assert samples["target_date"].tolist() == [
        pd.Timestamp("2024-12-31"),
        pd.Timestamp("2025-01-03"),
    ]
    assert samples["yield_10y"].tolist() == pytest.approx([4.0, 4.1])
    assert samples["next_yield_10y"].tolist() == pytest.approx([4.1, 4.3])


def test_model_training_predictions_and_mae_exclude_future_targets():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-12-27", "2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]
            ),
            "yield_10y": [1.0, 2.0, 3.0, 20.0, 30.0],
        }
    )

    train, test = split_model_samples(prepare_model_samples(data))
    model = train_model(train)
    metrics, forecasts = evaluate_forecasts(test, model.predict(test[["yield_10y"]]))
    mae = metrics.set_index("model").loc["regression", "mae_pp"]
    assert forecasts["target_date"].tolist() == [
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
    ]

    # Training pairs are 1 -> 2 and 2 -> 3, so the fitted line should be y = x + 1.
    assert model.coef_[0] == pytest.approx(1.0)
    assert model.intercept_ == pytest.approx(1.0)
    predictions = model.predict(pd.DataFrame({"yield_10y": [3.0, 20.0]}))
    assert predictions.tolist() == pytest.approx([4.0, 21.0])
    assert mae == pytest.approx(12.5)  # (|20 - 4| + |30 - 21|) / 2


@pytest.mark.parametrize("start", ["2024-01-01", "2025-01-01"])
def test_model_requires_both_training_and_test_samples(start):
    data = pd.DataFrame(
        {"date": pd.date_range(start, periods=3), "yield_10y": [3.0, 4.0, 5.0]}
    )

    with pytest.raises(ValueError, match="both training and test samples"):
        split_model_samples(prepare_model_samples(data))


def test_pandas_and_polars_match_expected_filtering_and_grouping():
    values = {
        "date": [
            datetime(2023, 1, 3),
            datetime(2023, 1, 4),
            datetime(2023, 1, 5),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
            datetime(2024, 1, 4),
        ],
        "yield_2y": [4.0, 4.0, None, 3.0, 5.0, 4.0],
        "yield_10y": [3.0, 4.0, 4.0, 4.0, 3.0, None],
    }
    pd_filtered, pd_yearly = pandas_analysis(pd.DataFrame(values))
    pl_filtered, pl_yearly = polars_analysis(pl.DataFrame(values))

    expected_dates = [datetime(2023, 1, 3)]
    assert pd_filtered["date"].tolist() == expected_dates
    assert pl_filtered["date"].to_list() == expected_dates
    # 2023 spreads are -1 and 0; 2024 spreads are +1 and -2.
    expected_yearly = pd.DataFrame(
        {
            "year": [2023, 2024],
            "observations": [2, 2],
            "mean_spread_pp": [-0.5, -0.5],
            "inverted_days": [1, 1],
        }
    )
    for result in [pd_yearly, pd.DataFrame(pl_yearly.to_dicts())]:
        pd.testing.assert_frame_equal(
            result,
            expected_yearly,
            check_dtype=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-10,
        )


@pytest.mark.parametrize(
    "csv_text, message",
    [
        ("date,yield_2y\n2024-01-02,4.0\n", "Missing required columns"),
        ("date,yield_2y,yield_10y\n,4.0,3.8\n", "Dates must not be missing"),
        (
            "date,yield_2y,yield_10y\n2024-01-02,4,3\n2024-01-02,5,4\n",
            "Duplicate dates",
        ),
        ("date,yield_2y,yield_10y\n2024-01-02,inf,3.8\n", "infinite"),
        ("date,yield_2y,yield_10y\n2024-01-02,4.0,not-a-yield\n", None),
        ("date,yield_2y,yield_10y\nnot-a-date,4.0,3.8\n", None),
    ],
)
def test_load_data_rejects_ambiguous_or_invalid_input(tmp_path, csv_text, message):
    path = tmp_path / "invalid.csv"
    path.write_text(csv_text)
    with pytest.raises(ValueError, match=message):
        load_data(path)


@pytest.mark.parametrize("values", [[], [None, None], [4.0]])
def test_model_samples_allow_no_available_pairs(values):
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(values)),
            "yield_10y": values,
        }
    )
    assert prepare_model_samples(data).empty


def test_split_uses_target_date_at_the_exact_cutoff():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-01"]),
            "yield_10y": [3.0, 4.0, 5.0],
        }
    )
    train, test = split_model_samples(prepare_model_samples(data))
    assert train["target_date"].tolist() == [pd.Timestamp("2024-12-31")]
    assert test["date"].tolist() == [pd.Timestamp("2024-12-31")]
    assert test["target_date"].tolist() == [pd.Timestamp("2025-01-01")]


def test_baseline_and_regression_use_same_targets_and_correct_units():
    test = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-12-31", "2025-01-02"]),
            "target_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "yield_10y": [3.0, 20.0],
            "next_yield_10y": [20.0, 30.0],
        },
        index=[8, 15],
    )
    original = test.copy(deep=True)
    metrics, predictions = evaluate_forecasts(test, [4.0, 21.0])
    metrics = metrics.set_index("model")
    # Regression: (16 + 9) / 2. Baseline: (17 + 10) / 2.
    assert metrics.loc["regression", "mae_pp"] == pytest.approx(12.5)
    assert metrics.loc["unchanged_yield", "mae_pp"] == pytest.approx(13.5)
    assert metrics.loc["regression", "mae_bp"] == pytest.approx(1250.0)
    assert predictions["unchanged_yield"].tolist() == [3.0, 20.0]
    assert predictions["actual"].tolist() == [20.0, 30.0]
    pd.testing.assert_frame_equal(test, original)


def test_perfect_baseline_keeps_zero_error():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3),
            "yield_10y": [4.0, 4.0, 4.0],
        }
    )
    metrics, _ = evaluate_forecasts(prepare_model_samples(data), [4.5, 4.5])
    metrics = metrics.set_index("model")
    assert metrics.loc["unchanged_yield", "mae_pp"] == 0
    assert metrics.loc["regression", "mae_pp"] == pytest.approx(0.5)


def test_evaluation_rejects_missing_predictions():
    samples = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=2),
            "yield_10y": [4.0, 4.1],
        }
    )
    with pytest.raises(ValueError, match="one prediction"):
        evaluate_forecasts(prepare_model_samples(samples), [])


def test_evaluation_rejects_empty_test_set():
    with pytest.raises(ValueError, match="Need test samples"):
        evaluate_forecasts(pd.DataFrame(), [])


def test_forecast_chart_uses_mae_in_basis_points(tmp_path, monkeypatch):
    comparison = pd.DataFrame(
        {
            "model": ["regression", "unchanged_yield"],
            "mae_pp": [0.04, 0.05],
            "mae_bp": [4.0, 5.0],
        }
    )
    fig, ax = analysis.plt.subplots()
    monkeypatch.setattr(analysis.plt, "subplots", lambda **kwargs: (fig, ax))
    analysis.plot_forecast_comparison(comparison, tmp_path / "comparison.png")
    assert [bar.get_height() for bar in ax.patches] == pytest.approx([4.0, 5.0])
    assert ax.get_ylim()[0] == 0
    assert ax.get_ylabel() == "Mean absolute error (basis points)"
