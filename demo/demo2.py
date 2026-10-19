from FinStat import Corr
from FinStat import factors
import numpy as np

fs = Corr("data")
fs.load_data()
pca = factors(fs)

pca.fit()
print(pca.vals[:3]) 
print("---------------------------------------------")               
print(pca.vecs.sum(axis=0))  # every entry >= 0
print("---------------------------------------------") 
print(pca.sigma.shape)                   # (19,)
print("---------------------------------------------")
print(np.allclose(pca.vecs @ np.diag(pca.vals) @ pca.vecs.T,np.corrcoef(pca.R, rowvar=False)))
print("---------------------------------------------")

share, cumulative = pca.explained_variance()
print(np.round(share[:5], 3))
print("---------------------------------------------")
print(np.round(cumulative[:5], 3))
print("---------------------------------------------")
print(cumulative[-1])

L = pca.loading(0)
print("---------------------------------------------")
print(L.shape)                    # (19,)
print("---------------------------------------------")
print(np.round(L, 2))             # all positive, around 0.6
print("---------------------------------------------")
print((L ** 2).sum()) 

w = pca.eigenportfolio(0)
print("---------------------------------------------")
print(w.shape)                       # (19,)
print("---------------------------------------------")
print(np.round(w, 1))
print("---------------------------------------------")
print(np.round(pca.sigma, 3))


t = pca.eigenportfolio(0)
print(t.shape)
print(np.round(t,1))
print(np.round(pca.sigma,3))


F = pca.eigenportfolio_returns(0)
print(F.shape)                                          # (T,)
print(np.corrcoef(F, pca.R.mean(axis=1))[0, 1])   