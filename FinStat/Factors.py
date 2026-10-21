"""Principal component analysis of stock return correlations.
 
Builds on the Corr class in FinStat. Takes the correlation matrix of daily
returns, extracts its eigenvalues and eigenvectors (the principal components),
and uses Random Matrix Theory (the Marchenko-Pastur distribution) to judge
which components carry real market structure and which are statistical noise.
 
"""



import numpy as np
import matplotlib.pyplot as plt
from FinStat import Corr

class factors:
    """Eigen-decomposition of the correlation matrix of daily returns.
 
    Call fit() before any other method; every other method relies on the
    attributes that fit() creates.
 
    Attributes:
        corr: The Corr object that supplies the price data.
        tickers: Ticker keys included in the analysis.
        vals: Eigenvalues of the correlation matrix, largest first. Set by
            fit().
        vecs: Matching unit-length eigenvectors, one per column, so vecs[:, k]
            belongs to vals[k]. Set by fit().
        R: Daily simple returns, shape (T days, N tickers). Set by fit().
        sigma: Sample standard deviation of each ticker's returns, shape (N,).
            Set by fit().
    """
    def __init__(self,corr_obj, tickers=None):
        """Store the data source and choose which tickers to analyse.
 
        Args:
            corr_obj: A Corr object whose load_data() has already been called.
            tickers: List of ticker keys present in corr_obj.data. Defaults
                to every loaded ticker.
        """
        self.corr = corr_obj
        self.tickers = tickers if tickers is not None else list(corr_obj.data)
        

    def fit(self):
        """Compute the eigenvalues and eigenvectors of the correlation matrix.
 
        Aligns the tickers on their shared trading dates, converts prices to
        daily returns, and diagonalises the correlation matrix. Results are
        stored on the instance and nothing is returned.
        """

        prices = self.corr._aligned_closes(self.tickers)
        R = self.corr._simple_returns(prices)

        # Correlation (not covariance) is used so that every stock carries
        # equal weight regardless of how volatile it is.
        # rowvar=False: each column is a ticker and each row is a day.
        C = np.corrcoef(R, rowvar = False)

        # eigh is for symmetric matrices and returns eigenvalues in ascending
        vals, vecs = np.linalg.eigh(C) 
        # order, so the order is reversed to put the largest first
        order = np.argsort(vals)
        vals = vals[order[::-1]]
        vecs = vecs[:, order [::-1]]

        # An eigenvector's sign is arbitrary (v and -v are equally valid).
        # Fix a convention so results are reproducible: each eigenvector's
        # entries must sum to a non-negative number. For the first component
        # this means every stock loads positively, i.e. the "market" factor.
        sums = vecs.sum(axis=0)
        sign = np.where(sums >= 0, 1, -1)
        vecs = (vecs*sign)
        self.vals = vals
        self.vecs = vecs
        self.R = R
        # ddof=1 gives the sample standard deviation, matching np.cov
        self.sigma = R.std(axis=0, ddof=1)

    def explained_variance(self):
        """Return the share of total variance explained by each component.
 
        Returns:
            tuple: (shares, cumulative). Both are arrays of length N.
                shares[k] is the fraction explained by component k, and
                cumulative[k] is the running total up to and including k.
        """
        # The eigenvalues of a correlation matrix sum to N, so
        # dividing by their sum turns each eigenvalue into a fraction
        shares = self.vals / self.vals.sum()
        cumulative = np.cumsum(shares)
        return shares, cumulative

    def loading(self,k=0):
        """Return the loading of every ticker on component k.
 
        The loading is the correlation between a ticker's returns and the
        component, so values lie between -1 and 1.
 
        Args:
            k: Zero-based component index (0 is the largest).
 
        Returns:
            numpy.ndarray: Loadings of shape (N,), in the order of
                self.tickers.
        """
        # Scaling the unit eigenvector by sqrt(eigenvalue) converts it from a
        # direction into a correlation with the component
        kth = np.sqrt(self.vals[k]) * self.vecs[:,k]
        return kth

    def eigenportfolio(self,k=0):
        """Return the weights of the eigenportfolio for component k.
 
        Args:
            k: Zero-based component index.
 
        Returns:
            numpy.ndarray: One weight per ticker, shape (N,). The weights are
                not normalised, so they do not sum to 1.
        """
        recipe = self.vecs[:, k]

        # Dividing by volatility works on raw returns what the eigenvector
        # does on standardised returns: it stops high-volatility stocks from
        # dominating the portfolio
        return recipe / self.sigma

    def eigenportfolio_returns(self,k=0):
        """Return the daily return series of the eigenportfolio for component k.
 
        Args:
            k: Zero-based component index.
 
        Returns:
            numpy.ndarray: Portfolio return for each day, shape (T,).
        """
        w = self.eigenportfolio(k)
        # (T x N) @ (N,) gives the weighted sum of returns for each day
        eigenR = self.R @ w
        return eigenR

    def mp_bounds(self,adjust = False):
        """Return the Marchenko-Pastur noise band for the eigenvalues.
 
        If the returns were pure noise, the eigenvalues of the correlation
        matrix would fall between these two bounds. Eigenvalues above the
        upper bound therefore indicate genuine structure.
 
        Args:
            adjust: If False, assume the noise has unit variance. If True,
                shrink the band to allow for the variance already taken by
                the largest (market) component.
 
        Returns:
            tuple: (lower, upper) bounds of the noise band.
        """
        T,N = self.R.shape
        # q is the ratio of tickers to observations; the band widens as q grows
        q = N/T
        if adjust:
            # The market mode absorbs a share vals[0] / N of the total
            # variance. Only the remainder is available to the noise, so the
            # noise variance is scaled down accordingly.
            s2 = s2 = 1 - self.vals[0]/N
        else:
            s2 = 1
        lower = s2*(1 - np.sqrt(q))**2
        upper = s2*(1 + np.sqrt(q))**2
        return lower, upper

    def n_significant(self, adjust = False):
        """Count the eigenvalues that fall outside the noise band.
 
        Args:
            adjust: Passed to mp_bounds().
 
        Returns:
            tuple: (above, below). The number of eigenvalues above the upper
                bound (real market factors) and below the lower bound.
        """
        lower, upper = self.mp_bounds(adjust)
        above = (self.vals > upper).sum()
        below = (self.vals < lower).sum()
        return above , below

    def plot_scree(self):
        """Plot a scree chart of the variance explained by each component.
 
        Bars show each component's share of variance, the line shows the
        cumulative share, and the dashed line marks the Marchenko-Pastur noise
        threshold (using the adjusted band).
        """
        shares, cumulative = self.explained_variance()
        lower,upper = self.mp_bounds(adjust=True)
        # The bounds are in eigenvalue units. Dividing by the eigenvalue sum
        # converts them to shares, matching the y-axis.
        lower = lower / self.vals.sum()
        upper = upper / self.vals.sum()
        T,N = self.R.shape
        # Components are numbered from 1 on the axis
        x = range(1,N+1)
        y = shares
        fig,ax = plt.subplots()
        ax.bar(x,y, label="Individual share")
        ax.plot(x,cumulative, marker = 'o', label="Cumulative share")
        ax.axhline(upper,linestyle="--", color="red", label="Marchenko-Pastur noise threshold")
        ax.legend()
        ax.set_title('Scree Plot : Variance Explained by Each Principal Component (N={N} stocks)')
        ax.set_xlabel('Component Number')
        ax.set_ylabel('Share of Variance')
        ax.set_xticks(x)
        plt.show()

    def plot_loadings(self, k=0):
        """Plot a horizontal bar chart of ticker loadings on component k.
 
        Args:
            k: Zero-based component index.
        """
        shares, cumulative = self.explained_variance()
        L = self.loading(k)
        # Sort so the bars run from the most negative to the most positive
        order = np.argsort(L)
        names = [n.replace("_stock_data", "") for n in self.tickers]
        names = ([names[i] for i in order])
        values = L[order]

        fig,ax = plt.subplots()
        ax.barh(names, values)
        ax.set_xlabel("Loading (correlation with component)")
        # k is zero-based, so add 1 for the human-readable component number
        ax.set_title(f"Loadings on Principal Component {k+1} ({shares[k]:.1%} of variance)")
        plt.show()

    def plot_vs_mp(self, adjust = True):
        """Plot the eigenvalue histogram against the Marchenko-Pastur density.
 
        Eigenvalues that sit above the theoretical noise edge are drawn as red
        dashed lines. Eigenvalues too large for the visible range are listed
        in a text note instead.
 
        Args:
            adjust: If True, use the adjusted noise band (see mp_bounds()).
        """
        lower, upper = self.mp_bounds(adjust)
        T, N = self.R.shape
        q = N / T
        # Noise variance must match the one used inside mp_bounds()
        if adjust:
            s2 = 1 - self.vals[0]/N
        else:
            s2 = 1
        # Marchenko-Pastur probability density, evaluated across the noise
        # band (it is zero outside the band)
        x = np.linspace(lower, upper, 200)
        density = np.sqrt((upper - x) * (x - lower)) / (2 * np.pi * s2 * q * x)
        # The market eigenvalue is often many times larger than the rest, so
        # the x-axis is capped to keep the noise region readable
        xmax = 1.5
        fig, ax = plt.subplots()
        ax.hist(self.vals, bins=10, density=True, range=(0, xmax),
            alpha=0.6, label="Eigenvalues")
        ax.plot(x, density, color="orange", label="Marchenko-Pastur noise density")

        # Vertical lines for eigenvalues above the noise edge. Only the first
        # line is labelled so the legend has a single entry for them.
        real = self.vals[self.vals > upper]
        for i, v in enumerate(real):
            ax.axvline(v, color="red", linestyle="--",
                   label="Above noise edge" if i == 0 else None)

        # Note for eigenvalues too large for the visible range
        off = self.vals[self.vals > xmax]
        order = np.argsort(off)
        off = off[order]
        if len(off) > 0:
            note = "Off scale: " + ", ".join(f"{v:.2f}" for v in off)
            ax.text(0.02, 0.95, note, transform=ax.transAxes, ha="left", va="top")

        band = "adjusted" if adjust else "plain"
        ax.set_title(f"Correlation Eigenvalues vs Marchenko-Pastur Noise ({band} band)")
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.set_xlim(0, xmax)
        # Headroom above the density peak so the legend does not cover it
        ax.set_ylim(0, density.max() * 1.7)
        ax.legend(loc="upper right")
        plt.show()




       


