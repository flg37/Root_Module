import numpy as np
import math as mt
import os
import csv

class finstat:
    def __init__(self,x):
        self.x = x

    def Data(self):
        current_directory = os.path.dirname(__file__)
        data_directory = os.path.join(current_directory, "..","data")

        csv_files = [
            filename for filename in os.listdir(data_directory)
            if filename.endswith(".csv")
        ]

        with open(file_path, newline-"", encoding ="utf-8") as csvfile:
            reader = csv.reader(csvfile)

            next(reader)  # Column headings
            next(reader)  # Ticker row
            next(reader)  # Date marker row

            data[name] = [
                    {
                        "Date": row[0],
                        "Close": float(row[1]),
                        "High": float(row[2]),
                        "Low": float(row[3]),
                        "Open": float(row[4]),
                        "Volume": int(row[5]),
                    }
                    for row in reader
                    if row
                ]

        return data



def main():
    x = finstat(None)
    x.Data()

if __name__ == "__main__":
    main()
