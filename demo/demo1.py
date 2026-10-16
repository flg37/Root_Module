"""
Demo file demonstrating how to run the FinStat Class

Inputs: Csv Data of 19 different Companies

Outputs: Correlation and Covarriance Matrix
"""

## Importing Class into python file
from module import FinStat


## 
def main():
    fs = FinStat(data_directory="data")    # looks for CSVs in the same folder as this file
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
    n = len(names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):         # upper triangle only
            pairs.append((corr[i, j], names[i], names[j]))
    pairs.sort(reverse=True)              # highest correlation first

    print("\nMost correlated pairs:")
    for c, a, b in pairs[:10]:
        print(f"  {a}-{b}: {c:.2f}")

    print("\nLeast correlated pairs:")
    for c, a, b in pairs[-10:]:
        print(f"  {a}-{b}: {c:.2f}")

if __name__ == "__main__":
    main()
