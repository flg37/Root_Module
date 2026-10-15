import numpy as np
import math as mt
import os


class finstat:
    def __init__(self,x):
        self.x = x

    def Data(self):
        current_directory = os.path.dirname(__file__)
        custom_directory = os.path.join(current_directory, "..", "data")
        file_path = os.path.join(current_directory, "AAPL_stock_data.csv")
        print(custom_directory)



def main():
    x = finstat(None)
    x.Data()


if __name__ == "__main__":
    main()
