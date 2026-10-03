from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib.image as mpimg
import pandas as pd
import pytest


def test_complete_analysis_runs_and_saves_outputs(tmp_path):
    source = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    (project / "data").mkdir(parents=True)
    shutil.copy2(source / "analysis.py", project / "analysis.py")
    csv_path = project / "data" / "treasury_yields_2023_2025.csv"
    shutil.copy2(source / "data" / csv_path.name, csv_path)
    original_csv = csv_path.read_bytes()
    output = tmp_path / "custom results"

    # Run the real command from another folder, including a path with spaces.
    result = subprocess.run(
        [sys.executable, str(project / "analysis.py"), "--output-dir", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Inverted dates in 2023: 250" in result.stdout
    assert "Training samples: 499 | Test samples: 249" in result.stdout

    comparison = pd.read_csv(output / "model_comparison.csv").set_index("model")
    assert comparison.loc["regression", "mae_pp"] == pytest.approx(0.0404, abs=0.0001)
    predictions = pd.read_csv(output / "predictions.csv", parse_dates=["target_date"])
    assert len(predictions) == 249
    assert predictions["target_date"].min() >= pd.Timestamp("2025-01-01")
    assert predictions["target_date"].is_unique

    # Independently reconstruct baseline pairs from the bundled data.
    valid = pd.read_csv(csv_path).dropna(subset=["yield_10y"])
    current = valid["yield_10y"].to_numpy()[:-1]
    actual = valid["yield_10y"].to_numpy()[1:]
    target_dates = pd.to_datetime(valid["date"].iloc[1:]).reset_index(drop=True)
    mask = target_dates >= pd.Timestamp("2025-01-01")
    assert predictions["target_date"].tolist() == target_dates[mask].tolist()
    assert predictions["unchanged_yield"].tolist() == pytest.approx(current[mask])
    assert predictions["actual"].tolist() == pytest.approx(actual[mask])
    baseline_mae = abs(actual[mask] - current[mask]).mean()
    assert comparison.loc["unchanged_yield", "mae_pp"] == pytest.approx(baseline_mae)
    for name in ["regression", "unchanged_yield"]:
        independent_mae = abs(predictions["actual"] - predictions[name]).mean()
        assert comparison.loc[name, "mae_pp"] == pytest.approx(independent_mae)
        assert comparison.loc[name, "mae_bp"] == pytest.approx(independent_mae * 100)

    for filename in ["treasury_yields_2023_2025.png", "forecast_mae.png"]:
        chart = mpimg.imread(output / filename)
        assert chart.shape[0] > 100 and chart.shape[1] > 100
        assert chart[:, :, :3].std() > 0.01
    assert csv_path.read_bytes() == original_csv
    assert not (project / "results").exists()
