from module import FinStat

def main():
    fs = FinStat(data_directory="data")    # looks for CSVs in the same folder as this file
    fs.load_data()

    tickers, cov_matrix = fs.covariance_matrix()
    print(tickers)
    print(cov_matrix)
    tickers, corr = fs.correlation_matrix()
    names = [t.replace("_stock_data", "") for t in tickers]
    fs.print_matrix(corr, names)
    fs.print_matrix(cov_matrix,names)

if __name__ == "__main__":
    main()
