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
        order2 = vals[order[::-1]]
        order3 = vecs[:, order [::-1]]
        print(order2[0])
        print(order2.sum())
        print(np.round(vecs.T @ vecs, 6))
        #print(order2)
        #print(order3)
        

