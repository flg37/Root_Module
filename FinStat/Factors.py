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
        C = np.corrcoef(R, rowrar = False)
        print(C.shape)
        E = np.linalg.eig(C)
        print(E)
        print(E.shape)

