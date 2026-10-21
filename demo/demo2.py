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
    lower, upper = pca.mp_bounds(adjust=True)
    print("Lower Bound")
    print(lower)
    print("Upper Bound")
    print(upper)
    pca.plot_scree()
    pca.plot_loadings()
    pca.plot_vs_mp()


if __name__ == "__main__":
    main()




#print(pca.vals[:3]) 
#print("---------------------------------------------")               
#print(pca.vecs.sum(axis=0))  # every entry >= 0
#print("---------------------------------------------") 
#print(pca.sigma.shape)                   # (19,)
#print("---------------------------------------------")
#print(np.allclose(pca.vecs @ np.diag(pca.vals) @ pca.vecs.T,np.corrcoef(pca.R, rowvar=False)))
#print("---------------------------------------------")


#print(np.round(share[:5], 3))
#print("---------------------------------------------")
#print(np.round(cumulative[:5], 3))
#print("---------------------------------------------")
#print(cumulative[-1])


#print("---------------------------------------------")
#print(L.shape)                    # (19,)
#print("---------------------------------------------")
#print(np.round(L, 2))             # all positive, around 0.6
#print("---------------------------------------------")
#print((L ** 2).sum()) 


#print("---------------------------------------------")
#print(w.shape)                       # (19,)
#print("---------------------------------------------")
#print(np.round(w, 1))
#print("---------------------------------------------")
#print(np.round(pca.sigma, 3))


#t = pca.eigenportfolio(0)
#print(t.shape)
#print(np.round(t,1))
#print(np.round(pca.sigma,3))

#print("------------------------------------------------------")
#print(np.corrcoef(F0, F1)[0, 1])         # about 1e-15, essentially 0
#print(np.var(F0, ddof=1), pca.vals[0])   # should match: 6.7557...
#print(np.var(F1, ddof=1), pca.vals[1]) 

#print("-------------------------------------------------------")
#lower, upper = pca.mp_bounds()
#print(lower, upper)
#lowe, upper = pca.mp_bounds(adjust = True)
#print(lower,upper)

#print("-------------------------------------------------------")
#print(pca.n_significant())
#print(pca.n_significant(adjust=True))

#print("-------------------------------------------------------")
#pca.plot_scree()

#print("-------------------------------------------------------")
#pca.plot_loadings()

#print("-------------------------------------------------------")
#pca.plot_vs_mp()