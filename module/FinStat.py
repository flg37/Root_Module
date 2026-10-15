import numpy as np
import math as mt
import os
import csv

class finstat:
    def __init__(self,data_directory=None):
        if data_directory is None:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            data_directory = os.path.join(current_directory, "..", "data")
        self.data_directory = data_directory
        self.data = {}

    def load_data(self):
        csv_files = [
            f for f in os.listdir(self.data_directory)
            if f.endswith(".csv")
        ]
        for filename in csv_files:
            name = os.path.splitext(filename)[0]
            file_path = os.path.join(self.data_directory, filename)
        
            with open(file_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                next(reader)
                next(reader)
                rows = [row for row in reader if row]

                self.data[name] = {
                    "Date":   np.array([r[0] for r in rows], dtype="datetime64[D]"),
                    "Close":  np.array([r[1] for r in rows], dtype=float),
                    "High":   np.array([r[2] for r in rows], dtype=float),
                    "Low":    np.array([r[3] for r in rows], dtype=float),
                    "Open":   np.array([r[4] for r in rows], dtype=float),
                    "Volume": np.array([r[5] for r in rows], dtype=float),
                    }
              

        return self.data


def main():
    fs = finstat()
    data = fs.load_data()
    #print({k: len(v) for k, v in data.items()})
    # list of dicts, one per day
    close = data["XOM_stock_data"]["Close"]    # already an array, no list comprehension needed
    returns = np.diff(close) / close[:-1]
    print(returns)


if __name__ == "__main__":
    main()