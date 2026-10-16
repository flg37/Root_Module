def main():
    fs = FinStat()    # looks for CSVs in the same folder as this file
    fs.load_data()

    tickers, cov_matrix = fs.covariance_matrix()
    print(tickers)
    print(cov_matrix)
    tickers, corr = fs.correlation_matrix()
    names = [t.replace("_stock_data", "") for t in tickers]
    fs.print_matrix(corr, names)

if __name__ == "__main__":
    main()
