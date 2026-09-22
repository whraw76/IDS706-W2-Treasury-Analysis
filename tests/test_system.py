from pathlib import Path
import re
import shutil
import subprocess
import sys

import matplotlib.image as mpimg
import pytest


def test_complete_analysis_runs_and_saves_a_chart(tmp_path):
    source = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    (project / "data").mkdir(parents=True)
    shutil.copy2(source / "analysis.py", project / "analysis.py")
    csv_path = project / "data" / "treasury_yields_2023_2025.csv"
    shutil.copy2(source / "data" / csv_path.name, csv_path)
    original_csv = csv_path.read_bytes()

    # Run the actual script from outside its folder, using the bundled dataset.
    result = subprocess.run(
        [sys.executable, str(project / "analysis.py")],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Inverted dates in 2023: 250" in result.stdout
    assert "Training samples: 499 | Test samples: 249" in result.stdout
    error = re.search(r"Test MAE: ([0-9.]+)", result.stdout)
    assert error is not None, result.stdout
    assert float(error.group(1)) == pytest.approx(0.0404, abs=0.0001)

    chart = mpimg.imread(project / "figures" / "treasury_yields_2023_2025.png")
    assert chart.shape[0] > 100 and chart.shape[1] > 100
    assert chart[:, :, :3].std() > 0.01  # Check that this is not a blank image.
    assert csv_path.read_bytes() == original_csv
