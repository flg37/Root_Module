from FinStat import Corr
from FinStat import factors
import numpy as np

fs = Corr("data")
fs.load_data()
pca = factors(fs)

pca.fit()
print(pca.vals[:3])                      # about 6.76, 2.30, 1.19, decreasing
print(pca.vecs.sum(axis=0))              # every entry >= 0
print(pca.sigma.shape)                   # (19,)
print(np.allclose(pca.vecs @ np.diag(pca.vals) @ pca.vecs.T,np.corrcoef(pca.R, rowvar=False)))

share, cumulative = pca.explained_variance()
print(np.round(share[:5], 3))
print(np.round(cumulative[:5], 3))
print(cumulative[-1])
