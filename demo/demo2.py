from FinStat import Corr
from FinStat import factors
import numpy as np



def main():
    fs = Corr("data")
    fs.load_data()
    pca = factors(fs)
    pca.fit()
    share, cumulative = pca.explained_variance()
    L = pca.loading(0)
    print("")
    print(L)
    w = pca.eigenportfolio(0)
    print("Eigen Portfolio of component 0")
    print(w)
    F0 = pca.eigenportfolio_returns(0)
    F1 = pca.eigenportfolio_returns(1)
    print("Eigen Portfolio of component 1")
    print(F1)
    lower, upper = pca.mp_bounds()
    print("Lower Bound")
    print(lower)
    print("Upper Bound")
    print(upper)
    lower, upper = pca.mp_bounds(adjust=True)
    print("Adjusted Lower Bound")
    print(lower)
    print("Adjusted Upper Bound")
    print(upper)
    f1 = pca.n_significant(adjust=True)
    print(f1)
    f2 = pca.n_significant(adjust = False)
    print(f2)
    pca.plot_scree()
    pca.plot_loadings()
    pca.plot_vs_mp()


if __name__ == "__main__":
    main()




