# FinStat


## Stock Correlation and Factor Analysis

A Python package for analysing how stocks move together. It loads daily price
data, computes covariance and correlation matrices of returns, and uses
principal component analysis (PCA) with Marchenko-Pastur noise filtering to
separate genuine market factors from statistical noise.

### Features

- Load daily price data for many tickers from CSV files.
- Align tickers on their shared trading dates before any calculation.
- Covariance and correlation matrices of daily returns, with tables, pair
  rankings and heatmaps.
- PCA of the correlation matrix: variance explained, loadings and
  eigenportfolios.
- Marchenko-Pastur noise band to show how many components are real.
- Plots: correlation heatmap, scree plot, loadings chart and eigenvalue
  histogram against the noise density.

## Installation

From the repository root, install the package in editable mode:

```sh
python -m pip install -e .
```

The package uses NumPy and Matplotlib.

## Data format

Place one CSV file per ticker in a folder (the demonstration uses a folder
called `data`). Each file must be in yfinance format:

- Three header rows (column names, ticker, date label).
- One row per trading day with the columns `Date`, `Close`, `High`, `Low`,
  `Open`, `Volume`.

The filename without its extension is used as the ticker name, for example
`AAPL_stock_data.csv` becomes `AAPL_stock_data`.

## Quick start

```python
from FinStat import Corr
from FinStat import factors

# Load the data
fs = Corr("data")
fs.load_data()

# Correlation of daily returns
tickers, corr = fs.correlation_matrix()
names = [t.replace("_stock_data", "") for t in tickers]
fs.print_matrix(corr, names)
fs.plot(corr, names)

# Principal component analysis
pca = factors(fs)
pca.fit()

share, cumulative = pca.explained_variance()
above, below = pca.n_significant(adjust=True)
print(f"{above} components lie above the noise band")

pca.plot_scree()
pca.plot_loadings(k=0)
pca.plot_vs_mp()
```

## Main methods

**`Corr`** (data loading and correlation)

| Method | Description |
| --- | --- |
| `load_data()` | Load every CSV in the data folder |
| `covariance(a, b)` | Covariance of daily returns for two tickers |
| `covariance_matrix()` | Covariance matrix of daily returns |
| `correlation_matrix()` | Correlation matrix of daily returns |
| `print_matrix()` | Print a matrix as a labelled table |
| `ranking()` | Most and least correlated pairs |
| `plot()` | Correlation heatmap |

**`factors`** (PCA and noise analysis). Call `fit()` first.

| Method | Description |
| --- | --- |
| `fit()` | Eigen-decomposition of the correlation matrix |
| `explained_variance()` | Share and cumulative share of variance per component |
| `loading(k)` | Correlation of each stock with component `k` |
| `eigenportfolio(k)` | Portfolio weights for component `k` |
| `eigenportfolio_returns(k)` | Daily returns of that portfolio |
| `mp_bounds(adjust)` | Marchenko-Pastur noise band |
| `n_significant(adjust)` | Number of eigenvalues above and below the band |
| `plot_scree()` | Scree plot with noise threshold |
| `plot_loadings(k)` | Loadings bar chart for component `k` |
| `plot_vs_mp(adjust)` | Eigenvalue histogram against the noise density |

Component indices are zero-based, so `k=0` is the largest component (the
market factor). Plot titles number components from 1.

## Demonstration

The complete demonstration can be run from the root directory using:

```sh
python -m demo.demo1
python -m demo.demo2
```