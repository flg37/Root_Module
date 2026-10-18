import numpy as np
import matplotlib.pyplot as plt
from FinStat import Corr

class factors:
    def __init__(self,corr_obj, tickers=None):
        self.corr = corr_obj
        self.tickers = tickers if tickers is not None else list(corr_obj.data)

    def fit(self):
        prices = self.corr._aligned_closes(self.tickers)
        R = self.corr._simple_returns(prices)
        #print(R.shape)
        C = np.corrcoef(R, rowvar = False)
        #print(C.shape)
        vals, vecs = np.linalg.eigh(C)
        print(vals)
        order = np.argsort(vals)
        vals = vals[order[::-1]]
        vecs = vecs[:, order [::-1]]
        print(vals[0])
        print(vals.sum())
        print(np.round(vecs.T @ vecs, 6))
        print(np.allclose(vecs @ np.diag(vals) @ vecs.T, C))



M = np.array([[ 1., -2.,  3.],
              [ 4.,  5., -6.]])

sums = M.sum(axis=0)
print(sums)                       # one sum per column
print(np.sign(sums))              # +1 or -1 for each column
print(M * np.sign(sums))          # flip the columns whose sum is negative
print((M * np.sign(sums)).sum(axis=0))
        

