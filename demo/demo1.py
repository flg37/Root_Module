"""
Demo file demonstrating how to run the FinStat Class

Inputs: Csv Data of 19 different Companies

Outputs: Correlation and Covarriance Matrix
"""

## Importing Class into python file
from FinStat import Corr


## 
def main():
    fs = Corr(data_directory="data")    # looks for CSVs in the same folder as this file
    fs.load_data()

    tickers, cov_matrix = fs.covariance_matrix()
    #print(tickers)
    #print(cov_matrix)
    tickers, corr = fs.correlation_matrix()
    names = [t.replace("_stock_data", "") for t in tickers]
    print("Correlation Matrix")
    fs.print_matrix(corr, names)
    print("--------------------------------------------------------------------------------------------------------------------------------------------")
    print("Covariance Matrix")
    fs.print_matrix(cov_matrix * 252, names, decimals=3, width=7)

    fs.ranking(names,corr)

    fs.plot(corr,names)

if __name__ == "__main__":
    main()
