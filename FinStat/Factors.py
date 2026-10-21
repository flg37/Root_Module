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
        C = np.corrcoef(R, rowvar = False)
        vals, vecs = np.linalg.eigh(C)
        order = np.argsort(vals)
        vals = vals[order[::-1]]
        vecs = vecs[:, order [::-1]]
        sums = vecs.sum(axis=0)
        sign = np.where(sums >= 0, 1, -1)
        vecs = (vecs*sign)
        self.vals = vals
        self.vecs = vecs
        self.R = R
        self.sigma = R.std(axis=0, ddof=1)

    def explained_variance(self):
        shares = self.vals / self.vals.sum()
        cumulative = np.cumsum(shares)
        return shares, cumulative

    def loading(self,k=0):
        kth = np.sqrt(self.vals[k]) * self.vecs[:,k]
        return kth

    def eigenportfolio(self,k=0):
        recipe = self.vecs[:, k]
        return recipe / self.sigma

    def eigenportfolio_returns(self,k=0):
        w = self.eigenportfolio(k)
        eigenR = self.R @ w
        return eigenR

    def mp_bounds(self,adjust = False):
        T,N = self.R.shape
        q = N/T
        if adjust:
            s2 = s2 = 1 - self.vals[0]/N
        else:
            s2 = 1
        lower = s2*(1 - np.sqrt(q))**2
        upper = s2*(1 + np.sqrt(q))**2
        return lower, upper

    def n_significant(self, adjust = False):
        lower, upper = self.mp_bounds(adjust)
        above = (self.vals > upper).sum()
        below = (self.vals < lower).sum()
        return above , below

    def plot_scree(self):
        shares, cumulative = self.explained_variance()
        T,N = self.R.shape
        x = range(1,N+1)
        y = shares
        fig,ax = plt.subplots()
        ax.bar(x,y)
        ax.plot(x,cumulative, marker = 'o')
        ax.set_title('Scree Plot')
        ax.set_xlabel('Component Number')
        ax.set_ylabel('Share of Variance')
        plt.show()


       


