"""Financial statistical methods.

Loads daily price data (Date, Close, High, Low, Open, Volume) from
yfinance-format CSV files into NumPy arrays for later analysis.
"""

# Importing Libraries
import numpy as np
import math as mt
import os
import csv


## Creating a Class containing the financial statistical methods
class finstat:

    """Holds price data for multiple tickers.

    Attributes:
        data_directory: Folder containing the CSV files.
        data: Maps each ticker (the CSV filename without extension) to a dict
            of NumPy arrays keyed by "Date", "Close", "High", "Low", "Open"
            and "Volume".
    """
    def __init__(self,data_directory=None):
        """Set the data folder, defaulting to ../data relative to this file."""
        if data_directory is None:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            data_directory = os.path.join(current_directory, "..", "data")
        self.data_directory = data_directory
        self.data = {}

    ## Creating a function to load the data
    def load_data(self):
        """Load every CSV in the data directory into self.data.

        Returns:
            dict: The loaded data, keyed by ticker.
        """
        csv_files = [
            f for f in os.listdir(self.data_directory)
            if f.endswith(".csv")
        ]
    
        for filename in csv_files:
            name = os.path.splitext(filename)[0]
            file_path = os.path.join(self.data_directory, filename)

            ## Opening each csv file and loading the data into Numpy arrays
            with open(file_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)

                # Skip the three header rows: column names, ticker, date label
                next(reader)
                next(reader)
                next(reader)
                rows = [row for row in reader if row]

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

    def covariance(self, ticker_a, ticker_b):
        """Return the sample covariance of daily returns for two tickers."""
        _, closes = self._aligned_closes([ticker_a, ticker_b])
        returns = self._simple_returns(closes)
        return np.cov(returns, rowvar=False)[0, 1]

def main():
    fs = finstat()
    data = fs.load_data()
    cov_xom_aapl = fs.covariance("XOM_stock_data", "AAPL_stock_data")
    tickers, cov_matrix = fs.covariance_matrix()
    
    close = data["XOM_stock_data"]["Close"]    # already an array, no list comprehension needed
    returns = np.diff(close) / close[:-1]
    print(cov_xom_aapl)
    print(returns)


if __name__ == "__main__":
    main()