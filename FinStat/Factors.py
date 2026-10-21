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
        lower,upper = self.mp_bounds(adjust=True)
        lower = lower / self.vals.sum()
        upper = upper / self.vals.sum()
        T,N = self.R.shape
        x = range(1,N+1)
        y = shares
        fig,ax = plt.subplots()
        ax.bar(x,y)
        ax.plot(x,cumulative, marker = 'o')
        ax.axhline(upper,linestyle="--", color="red", label="threshold")
        ax.legend()
        ax.set_title('Scree Plot')
        ax.set_xlabel('Component Number')
        ax.set_ylabel('Share of Variance')
        ax.set_xticks(x)
        plt.show()

    def plot_loadings(self, k=0):
        L = self.loading(k)
        order = np.argsort(L)
        names = [n.replace("_stock_data", "") for n in self.tickers]
        names = ([names[i] for i in order])
        values = L[order]

        fig,ax = plt.subplots()
        ax.barh(names, values)
        ax.set_xlabel("Loading")
        plt.show()

    def plot_vs_mp(self, adjust = True):
        lower, upper = self.mp_bounds(adjust)
        T, N = self.R.shape
        q = N / T
        if adjust:
            s2 = 1 - self.vals[0]/N
        else:
            s2 = 1
        x = np.linspace(lower, upper, 200)
        density = np.sqrt((upper - x) * (x - lower)) / (2 * np.pi * s2 * q * x)
        xmax = 1.5
        fig, ax = plt.subplots()
        ax.hist(self.vals, bins=10, density=True, range=(0, xmax),
            alpha=0.6, label="Eigenvalues")
        ax.plot(x, density, color="orange", label="Marchenko-Pastur noise density")

        # vertical lines for eigenvalues above the noise edge (label only the first)
        real = self.vals[self.vals > upper]
        for i, v in enumerate(real):
            ax.axvline(v, color="red", linestyle="--",
                   label="Above noise edge" if i == 0 else None)

        # note for eigenvalues too large for the visible range
        off = self.vals[self.vals > xmax]
        order = np.argsort(off)
        off = off[order]
        if len(off) > 0:
            note = "Off scale: " + ", ".join(f"{v:.2f}" for v in off)
            ax.text(0.02, 0.95, note, transform=ax.transAxes, ha="left", va="top")

        band = "adjusted" if adjust else "plain"
        ax.set_title(f"Eigenvalues vs Marchenko-Pastur ({band} band)")
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, density.max() * 1.4)
        ax.legend(loc="upper right")
        plt.show()




       


