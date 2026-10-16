"""
Financial Stastical Module: 

Loads daily price data (Date, Close, High, Low, Open, Volume) from
yfinance-format CSV files into NumPy arrays and computes return covariances.
"""

import csv 
import numpy as np
import os


class FinStat:

    """Holds price data for multiple tickers.

    Attributes:
        data_directory: Folder containing the CSV files.
        data: Maps each ticker (CSV filename without extension) to a dict of
            NumPy arrays keyed by "Date", "Close", "High", "Low", "Open"
            and "Volume".
    """
    def __init__(self, data_directory = None):
        """
        Set the data folder, defaulting to the folder containing this file.
        """
        if data_directory is None:
            current_directory = data_directory = os.path.dirname(os.path.abspath(__file__))
            data_directory = os.path.join(current_directory, "data")
        self.data_directory = data_directory
        self.data = {}


    def load_data(self):
        csv_files = [f for f in os.listdir(self.data_directory) if f.lower().endswith(".csv")]

        for filename in csv_files:
            name = os.path.splitext(filename)[0]
            file_path = os.path.join(self.data_directory, filename)

            with open(file_path, newline="", encoding="utf-8") as csvfile:
                # Skip the three header rows: column names, ticker, date label
                rows = [r for r in list(csv.reader(csvfile))[3:] if r]

                # Column order in the file: Date, Close, High, Low, Open, Volume
            self.data[name] = {
                "Date":   np.array([r[0] for r in rows], dtype="datetime64[D]"),
                "Close":  np.array([r[1] for r in rows], dtype=float),
                "High":   np.array([r[2] for r in rows], dtype=float),
                "Low":    np.array([r[3] for r in rows], dtype=float),
                "Open":   np.array([r[4] for r in rows], dtype=float),
                "Volume": np.array([r[5] for r in rows], dtype=float),
                    }
        return self.data

    def _aligned_closes(self, tickers):
        """Return a (n_dates, n_tickers) array of closes on shared dates only."""
        common = self.data[tickers[0]]["Date"]
        for t in tickers[1:]:
            common = np.intersect1d(common, self.data[t]["Date"])

        return np.column_stack([
            self.data[t]["Close"][np.isin(self.data[t]["Date"], common)]
            for t in tickers
        ])

    def _simple_returns(self, prices):
        return prices[1:] / prices[:-1] - 1

    def covariance(self, ticker_a, ticker_b):
        """Return the sample covariance of daily returns for two tickers."""
        returns = self._simple_returns(self._aligned_closes([ticker_a, ticker_b]))
        return np.cov(returns, rowvar=False)[0, 1]

    def covariance_matrix(self, tickers=None):
        """Return (tickers, covariance matrix) of daily returns.

        Defaults to every loaded ticker. Rows and columns follow the order
        of the returned ticker list.
        """
        if tickers is None:
            tickers = list(self.data)
        returns = self._simple_returns(self._aligned_closes(tickers))
        return tickers, np.cov(returns, rowvar=False)
    
    def correlation_matrix(self, tickers=None):
        """Return (tickers, correlation matrix) of daily returns."""
        tickers, cov = self.covariance_matrix(tickers)
        std = np.sqrt(np.diag(cov))          # each stock's daily volatility
        return tickers, cov / np.outer(std, std)
    def print_matrix(matrix, names, decimals=2, width=6):
        """Print a square matrix as a labelled table."""
        print(" " * width + "".join(f"{n:>{width}}" for n in names))
        for name, row in zip(names, matrix):
            print(f"{name:<{width}}" + "".join(f"{v:>{width}.{decimals}f}" for v in row))

    def print_matrix(self,matrix, names, decimals=2, width=6):
        """Print a square matrix as a labelled table."""
        print(" " * width + "".join(f"{n:>{width}}" for n in names))
        for name, row in zip(names, matrix):
            print(f"{name:<{width}}" + "".join(f"{v:>{width}.{decimals}f}" for v in row))

    def ranking(self):
        n = len(self.names)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):         # upper triangle only
                pairs.append((self.corr[i, j], self.names[i], self.names[j]))
        pairs.sort(reverse=True)              # highest correlation first
        
        print("\nMost correlated pairs:")
        for c, a, b in pairs[:10]:
            print(f"  {a}-{b}: {c:.2f}")
        
        print("\nLeast correlated pairs:")
        for c, a, b in pairs[-10:]:
            print(f"  {a}-{b}: {c:.2f}")