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
        #print(vals[0])
        #print(vals.sum())
        #print(np.round(vecs.T @ vecs, 6))
        #print(np.allclose(vecs @ np.diag(vals) @ vecs.T, C))
        sums = vecs.sum(axis=0)
        sign = np.where(sums >= 0, 1, -1)
        vecs = (vecs*sign)
        self.vals = vals
        self.vecs = vecs
        self.R = R
        self.sigma = R.std(axis=0, ddof=1)


