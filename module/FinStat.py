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

                next(reader)  # Column headings
                next(reader)  # Ticker row
                next(reader)  # Date marker row

                self.data[name] = [
                    {
                        "Date": row[0],
                        "Close": float(row[1]),
                        "High": float(row[2]),
                        "Low": float(row[3]),
                        "Open": float(row[4]),
                        "Volume": int(float(row[5])),
                    }
                    for row in reader
                    if row
                ]

        return self.data


def main():
    fs = finstat()
    data = fs.load_data()
    print({k: len(v) for k, v in data.items()})


if __name__ == "__main__":
    main()