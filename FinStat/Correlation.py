"""Financial statistics module.
 
Loads daily price data (Date, Close, High, Low, Open, Volume) from
yfinance-format CSV files into NumPy arrays, then computes covariance and
correlation matrices of daily returns and presents them as tables, rankings
and heatmaps.
 
Example:
    fs = FinStat("data")
    fs.load_data()
    tickers, corr = fs.correlation_matrix()
    names = [t.replace("_stock_data", "") for t in tickers]
    fs.print_matrix(corr, names)
"""

import csv 
import numpy as np
import os
import matplotlib.pyplot as plt


class FinStat:

    """Load price data for multiple tickers and analyse their returns.
 
    Attributes:
        data_directory: Folder containing the CSV files.
        data: Maps each ticker (CSV filename without extension) to a dict of
            NumPy arrays keyed by "Date", "Close", "High", "Low", "Open"
            and "Volume". Empty until load_data() is called.
    """
    def __init__(self, data_directory = None):
        """Set the folder that the CSV files are read from.
 
        Args:
            data_directory: Path to the folder of CSV files. A relative path
                is resolved from the current working directory. If None, a
                "data" folder next to this module is used.
        """
        if data_directory is None:
            current_directory = data_directory = os.path.dirname(os.path.abspath(__file__))
            data_directory = os.path.join(current_directory, "data")
        self.data_directory = data_directory
        self.data = {}


    def load_data(self):
        """Load every CSV in the data directory into self.data.
 
        Each file must be in yfinance format: three header rows (column
        names, ticker, date label) followed by one row per trading day with
        the columns Date, Close, High, Low, Open, Volume.
 
        Returns:
            dict: self.data, keyed by ticker.
        """
        csv_files = [f for f in os.listdir(self.data_directory) if f.lower().endswith(".csv")]

        for filename in csv_files:
            # The filename without its extension is used as the ticker key
            name = os.path.splitext(filename)[0]
            file_path = os.path.join(self.data_directory, filename)

            with open(file_path, newline="", encoding="utf-8") as csvfile:
                # Skip the three header rows, and drop any blank lines
                rows = [r for r in list(csv.reader(csvfile))[3:] if r]

            # One array per column, so position i in every array is the same
            # trading day. Column order in the file: Date, Close, High, Low,
            # Open, Volume
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
        """Return closing prices for several tickers on their shared dates.
 
        NumPy matches values by position, not by date, so tickers with
        different trading days must be aligned before any calculation.
 
        Args:
            tickers: List of ticker keys present in self.data.
 
        Returns:
            numpy.ndarray: Shape (n_dates, n_tickers). Each column is one
                ticker and each row is the same trading day for all of them.
        """
        # Narrow down to the dates present in every requested ticker
        common = self.data[tickers[0]]["Date"]
        for t in tickers[1:]:
            common = np.intersect1d(common, self.data[t]["Date"])

        # Boolean mask keeps only the shared dates, preserving original order
        return np.column_stack([
            self.data[t]["Close"][np.isin(self.data[t]["Date"], common)]
            for t in tickers
        ])

    def _simple_returns(self, prices):
        """Return day-over-day simple returns, (p_t / p_{t-1}) - 1.
 
        Args:
            prices: Array of prices, ordered oldest to newest along axis 0.
 
        Returns:
            numpy.ndarray: Returns with one fewer row than prices, because
                the first day has no previous day to compare against.
        """
        return prices[1:] / prices[:-1] - 1

    def covariance(self, ticker_a, ticker_b):
        """Return the sample covariance of daily returns for two tickers.
 
        Args:
            ticker_a: First ticker key in self.data.
            ticker_b: Second ticker key in self.data.
 
        Returns:
            float: Covariance of the two return series, using n - 1.
        """
        returns = self._simple_returns(self._aligned_closes([ticker_a, ticker_b]))
        # np.cov returns a 2x2 matrix; the off-diagonal entry is the covariance
        return np.cov(returns, rowvar=False)[0, 1]

    def covariance_matrix(self, tickers=None):
        """Return the covariance matrix of daily returns.
 
        Only dates shared by all requested tickers are used, so a ticker with
        a short history shortens the data for every pair.
 
        Args:
            tickers: List of ticker keys. Defaults to every loaded ticker.
 
        Returns:
            tuple: (tickers, matrix). Rows and columns of the matrix follow
                the order of the returned ticker list.
        """
        if tickers is None:
            tickers = list(self.data)
        returns = self._simple_returns(self._aligned_closes(tickers))
        # rowvar=False: each column is a ticker and each row is a day
        return tickers, np.cov(returns, rowvar=False)
    
    def correlation_matrix(self, tickers=None):
        """Return the correlation matrix of daily returns.
 
        Args:
            tickers: List of ticker keys. Defaults to every loaded ticker.
 
        Returns:
            tuple: (tickers, matrix), with values between -1 and 1.
        """
        tickers, cov = self.covariance_matrix(tickers)
        # The diagonal of the covariance matrix holds each variance
        std = np.sqrt(np.diag(cov))          
        return tickers, cov / np.outer(std, std)
        
    def print_matrix(self,matrix, names, decimals=2, width=6):
        """Print a square matrix as a labelled table.
 
        Args:
            matrix: Square 2D array to print.
            names: Labels for the rows and columns, in matrix order.
            decimals: Number of decimal places for each value.
            width: Character width of each column.
        """
        print(" " * width + "".join(f"{n:>{width}}" for n in names))
        for name, row in zip(names, matrix):
            print(f"{name:<{width}}" + "".join(f"{v:>{width}.{decimals}f}" for v in row))

    def ranking(self,names,corr):
        n = len(names)
        pairs = []
        for i in range(n):
            # Start at i + 1 to skip the diagonal and avoid listing a pair
            # twice (A-B and B-A)
            for j in range(i + 1, n):         # upper triangle only
                pairs.append((corr[i, j], names[i], names[j]))
        pairs.sort(reverse=True)              # highest correlation first
        
        print("\nMost correlated pairs:")
        for c, a, b in pairs[:10]:
            print(f"  {a}-{b}: {c:.2f}")
        
        print("\nLeast correlated pairs:")
        for c, a, b in pairs[-10:]:
            print(f"  {a}-{b}: {c:.2f}")

        
    def plot(self, corr, names):
        """Plot a correlation matrix as a labelled heatmap.
 
        Args:
            corr: Square correlation matrix.
            names: Labels for the rows and columns of corr.
        """
        n = len(names)

        fig, ax = plt.subplots(figsize=(10, 8))
        # Fix the colour scale to the full correlation range so colours are
        # comparable between plots
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=90)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names)

        fig.colorbar(im, label="Correlation")
        ax.set_title("Correlation of daily returns")
        plt.tight_layout()
        plt.show()