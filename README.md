# Treasury Yield Analysis — IDS 706

This beginner project uses Pandas to inspect U.S. Treasury yields, filter inverted dates, summarize by year, create one chart, and try a simple linear regression.

## Data

Source: [Federal Reserve GSW nominal yield curve data](https://www.federalreserve.gov/data/nominal-yield-curve.htm) ([original CSV](https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv)).

The included `data/treasury_yields_2023_2025.csv` has **783 rows and 6 columns**: a date and 1-, 2-, 5-, 10-, and 30-year zero-coupon yields, covering 2023–2025. Each row is one date listed in the source. Yields use annualized percentages: `4.25` means 4.25%.

These are fitted, continuously compounded yields from a Fed staff research dataset, which may be revised. The included subset comes from a source downloaded on 2026-09-08. Source metadata is retained in `data/source_metadata.json`.

## Run

Use Python 3.10 or newer. From this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py
python compare_polars.py
```

`analysis.py` prints the results and saves a chart in `figures/`. `compare_polars.py` checks the Pandas/Polars results and prints timing comparisons. Both run offline after installing the dependencies.

## Analysis and findings

**Import and inspect.** `pd.read_csv(..., parse_dates=["date"])` reads the data. The script uses `.head()`, `.shape`, `.info()`, `.describe()`, `.isna().sum()`, and `.duplicated()`. There are **749 complete rows**, **34 rows missing all five yields**, and **no duplicate rows or dates**. The original CSV retains missing values. Calculations exclude rows missing the required yields, without filling them.

**Filter and group.** Define the spread as `yield_10y - yield_2y`. A negative spread means the 10-year yield is below the 2-year yield, an inversion between these maturities. The script combines two conditions to find inverted dates in 2023, then uses `groupby("year").agg(...)` to calculate counts and average spreads.

| Year | Valid dates | Mean spread (percentage points) | Inverted dates |
| --- | ---: | ---: | ---: |
| 2023 | 250 | -0.5896 | 250 |
| 2024 | 250 | -0.1288 | 165 |
| 2025 | 249 | 0.5458 | 0 |

The mean spread changed from negative in 2023 to positive in 2025. Inversion occurred on every valid date in 2023, on 165 dates in 2024, and on no valid dates in 2025.

**Visualization.** A line chart shows how the two yields change over time. Orange is the 2-year yield, blue is the 10-year yield, and shading marks inversion. Missing observations remain gaps. Shared axes make it easy to compare the yields and see their ordering change during 2024.

![2-year and 10-year Treasury yields with shaded inversions](figures/treasury_yields_2023_2025.png)

**Machine learning.** The algorithm is scikit-learn's `LinearRegression`, which fits a straight-line relationship between an input and a numerical target.

- **Input X:** current 10-year yield, one feature.
- **Target y:** next observed 10-year yield, created with `.shift(-1)`.
- **Preparation:** sort dates, exclude missing observations, and remove the final row because it has no future target.
- **Split:** 499 training samples with target dates in 2023–2024; 249 test samples with target dates in 2025. Splitting by the target date keeps 2025 outcomes out of training.
- **Evaluation:** test mean absolute error (**MAE**) is **0.0404 percentage points**. MAE measures the average size of the prediction error; lower is better. The script also prints five actual and predicted values.

The model is fitted once and predicts the next observation using the current observed yield. This is a basic learning exercise on a small historical sample; the error alone does not establish useful forecasting skill.

## Pandas and Polars comparison

[`compare_polars.py`](compare_polars.py) imports the same CSV with both libraries, removes missing 2-year/10-year yields, creates the spread and year columns, filters inverted dates in 2023, and calculates the same yearly summary. It checks that the **250 filtered dates match** and that all yearly statistics agree within floating-point tolerance.

| Operation | Pandas | Polars |
| --- | --- | --- |
| Remove missing yields | `dropna(subset=[...])` | `drop_nulls(subset=[...])` |
| Create a column | `df["spread"] = ...` | `with_columns((...).alias("spread"))` |
| Filter rows | `df[(condition1) & (condition2)]` | `filter((condition1) & (condition2))` |
| Summarize by year | `groupby("year").agg(...)` | `group_by("year").agg(...)` |

Pandas uses column indexing and assignment here; Polars uses expressions such as `pl.col("yield_10y")` inside `with_columns`, `filter`, and `agg`. See the [Polars grouping documentation](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.group_by.html) for the expression syntax.

**Timing method:** after a warm-up, run each complete cleaning/filtering/grouping function 100 times per batch for 5 batches. Report the median batch time divided by 100. CSV loading, result checks, printing, plotting, and ML are outside the timing. Both libraries start with the same 783-row dataset already loaded in memory and produce year-sorted summaries.

| Library | Version | Sample runtime per run |
| --- | --- | ---: |
| Pandas | 3.0.5 | 2.454 ms |
| Polars | 1.44.2 | 1.057 ms |

This run used Python 3.14.3 on an Apple Silicon Mac. Polars took less time in this small experiment, but both completed the operations in a few milliseconds. Timings vary with the machine and current load; this small dataset does not establish a general performance advantage.

## Rust notebook

[`rust_vs_python_intro.ipynb`](rust_vs_python_intro.ipynb) is adapted from [the course notebook](https://github.com/Kedar-V/data-processing-frameworks-demo/blob/75772a44ed1cbc61e66396fad49a9aa0c913c8ef/notebooks/rust_vs_python_intro.ipynb). Its lesson and exercises are retained, with completed **Your turn** cells and short experiment notes.

| Experiment | Change and observed result |
| --- | --- |
| Mutability | Added `mut`: the value changes from 1000 to 2000. The immutable version raises E0384. |
| Ownership and cloning | Moving a vector and reusing its original name raises E0382. Appending to a `.clone()` changes the copy and leaves the original unchanged. |
| Borrowing | Two shared references can read the vector; a later mutable reference can append a value. |
| Borrow conflict | Changed the loop to `for rating in &ratings` to demonstrate E0502, matching the lesson's explanation. A working version writes to a separate vector and removes both 2s. |
| Optional scope example | Changed 64 MB to 16 MB; `Drop` prints `FREE 16 MB` when the inner scope ends. |

All **17 code cells were executed with the Rust kernel**: 14 ran normally and 3 intentionally produced the compiler errors above. Their actual outputs are saved in the notebook. Those three marked error cells are part of the experiments; continue to the following working cells after reading them.

To rerun, open the notebook in VS Code with the Jupyter extension and choose **Rust** as the kernel. Rust and the kernel are already installed on this computer. For a new computer, follow the [course Rust setup instructions](https://github.com/Kedar-V/data-processing-frameworks-demo/blob/master/SETUP.md). The notebook uses small built-in examples and does not read the MovieLens dataset.
