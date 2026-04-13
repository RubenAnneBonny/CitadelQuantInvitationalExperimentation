"""
Online HMM Regime Detector
--------------------------
Wraps the HMM from Training_before_comp with online feature extraction
suitable for tick-level live trading.

Features fed to the HMM per tick:
  - return_1   : last 1-tick price return for the y-leg (CRZY)
  - volatility : rolling std of last VOL_WINDOW returns
  - zscore     : Kalman pairs z-score

Regimes are labelled by inspecting the fitted emission parameters:
  - 'mean_reverting' : low variance, near-zero mean return
  - 'trending'       : high |mean return|
  - 'crisis'         : highest total variance (trace of Sigma)

Strategy implication:
  mean_reverting → Bollinger + pairs both active
  trending       → pairs only (spread is direction-neutral), BB suppressed
  crisis         → flatten and stop trading
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


# ── HMM (ported from Training_before_comp/HMM.ipynb) ────────────────────────

class HMM:
    def __init__(self, K, X):
        kmeans = KMeans(n_clusters=K, n_init=10).fit(X)
        labels = kmeans.labels_

        self.X = X
        self.K = K
        self.mu    = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        self.Sigma = np.array([np.cov(X[labels == k].T) + np.eye(X.shape[1]) * 1e-6
                               for k in range(K)])
        self.pi    = np.array([(labels == k).mean() for k in range(K)])
        self.A     = np.full((K, K), 1 / K)

    def _emission(self, x, k):
        return multivariate_normal.pdf(x, mean=self.mu[k], cov=self.Sigma[k])

    def _forward(self):
        T     = len(self.X)
        alpha = np.zeros((T, self.K))
        for k in range(self.K):
            alpha[0, k] = self.pi[k] * self._emission(self.X[0], k)
        for t in range(1, T):
            for k in range(self.K):
                alpha[t, k] = (
                    self._emission(self.X[t], k)
                    * np.sum(alpha[t - 1] * self.A[:, k])
                )
        return alpha

    def _backward(self):
        T    = len(self.X)
        beta = np.ones((T, self.K))
        for t in range(T - 2, -1, -1):
            for k in range(self.K):
                beta[t, k] = sum(
                    self.A[k, j] * self._emission(self.X[t + 1], j) * beta[t + 1, j]
                    for j in range(self.K)
                )
        return beta

    def _posteriors(self, alpha, beta):
        T     = len(self.X)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((T - 1, self.K, self.K))
        em = np.array([[self._emission(self.X[t], k) for k in range(self.K)]
                       for t in range(T)])
        for t in range(T - 1):
            for j in range(self.K):
                for k in range(self.K):
                    xi[t, j, k] = (
                        alpha[t, j] * self.A[j, k]
                        * em[t + 1, k] * beta[t + 1, k]
                    )
            xi[t] /= xi[t].sum()
        return gamma, xi

    def _m_step(self, gamma, xi):
        self.pi = gamma[0]
        self.A  = xi.sum(axis=0) / gamma[:-1].sum(axis=0, keepdims=True).T
        for k in range(self.K):
            w          = gamma[:, k]
            self.mu[k] = (w[:, None] * self.X).sum(axis=0) / w.sum()
            diff       = self.X - self.mu[k]
            self.Sigma[k] = (
                (w[:, None] * diff).T @ diff / w.sum()
                + np.eye(self.X.shape[1]) * 1e-6
            )

    def _fit(self, tol=1e-4, max_iter=80):
        ll_prev = -np.inf
        for _ in range(max_iter):
            alpha         = self._forward()
            beta          = self._backward()
            gamma, xi     = self._posteriors(alpha, beta)
            ll            = np.sum(np.log(alpha.sum(axis=1) + 1e-300))
            if abs(ll - ll_prev) < tol:
                break
            ll_prev = ll
            self._m_step(gamma, xi)

    def predict_states(self, X_new):
        """Forward-only pass on X_new using fitted parameters. Returns soft gamma."""
        T     = len(X_new)
        alpha = np.zeros((T, self.K))
        for k in range(self.K):
            alpha[0, k] = self.pi[k] * multivariate_normal.pdf(
                X_new[0], mean=self.mu[k], cov=self.Sigma[k])
        for t in range(1, T):
            for k in range(self.K):
                alpha[t, k] = (
                    multivariate_normal.pdf(X_new[t], mean=self.mu[k], cov=self.Sigma[k])
                    * np.sum(alpha[t - 1] * self.A[:, k])
                )
        gamma = alpha / (alpha.sum(axis=1, keepdims=True) + 1e-300)
        return gamma


# ── Online wrapper ────────────────────────────────────────────────────────────

VOL_WINDOW   = 10    # ticks for rolling volatility
MIN_SAMPLES  = 30    # minimum observations before first HMM fit
REFIT_EVERY  = 25    # re-fit HMM every N new ticks after initial fit


class OnlineHMMRegime:
    """
    Feed one tick at a time via .update(); get back the current regime label.
    Returns None until MIN_SAMPLES ticks have been collected.
    """

    REGIMES = ("mean_reverting", "trending", "crisis")

    def __init__(self, K: int = 3):
        self.K            = K
        self.hmm          = None
        self.label_map    = {}        # HMM state index → regime string
        self._features    = []        # list of 3-d feature vectors
        self._returns     = []        # raw returns for volatility window
        self._ticks_since_fit = 0
        self.current_regime   = None

    def update(self, price_y: float, zscore: float) -> str | None:
        """
        Parameters
        ----------
        price_y : latest price of the y-leg (CRZY)
        zscore  : Kalman pair z-score at this tick

        Returns
        -------
        Regime string or None if not enough data yet.
        """
        # Build return
        if self._returns:
            ret = (price_y - self._returns[-1]) / (self._returns[-1] + 1e-9)
        else:
            ret = 0.0
        self._returns.append(price_y)

        # Volatility of recent returns
        ret_series = np.diff(self._returns[-VOL_WINDOW - 1:]) if len(self._returns) > 1 else [0.0]
        vol = float(np.std(ret_series)) if len(ret_series) > 1 else 0.0

        self._features.append(np.array([ret, vol, zscore]))
        self._ticks_since_fit += 1

        n = len(self._features)
        if n < MIN_SAMPLES:
            return None

        # Fit / refit
        if self.hmm is None or self._ticks_since_fit >= REFIT_EVERY:
            self._refit()
            self._ticks_since_fit = 0

        # Predict current regime using forward-only pass
        X   = np.array(self._features)
        gamma = self.hmm.predict_states(X)
        state = int(gamma[-1].argmax())
        self.current_regime = self.label_map.get(state, "mean_reverting")
        return self.current_regime

    def _refit(self):
        X = np.array(self._features)
        self.hmm = HMM(K=self.K, X=X)
        self.hmm._fit()
        self._build_label_map()

    def _build_label_map(self):
        """
        Label each HMM state by its statistical character.
        Highest total variance → crisis
        Highest |mean return| among the rest → trending
        Remaining → mean_reverting
        """
        variances    = [np.trace(self.hmm.Sigma[k]) for k in range(self.K)]
        mean_returns = [abs(self.hmm.mu[k][0]) for k in range(self.K)]

        if self.K == 2:
            hi_var = int(np.argmax(variances))
            self.label_map = {
                hi_var:     "trending",
                1 - hi_var: "mean_reverting",
            }
        else:  # K == 3
            crisis_k  = int(np.argmax(variances))
            remaining = [k for k in range(self.K) if k != crisis_k]
            trending_k = remaining[int(np.argmax([mean_returns[k] for k in remaining]))]
            mr_k       = [k for k in remaining if k != trending_k][0]
            self.label_map = {
                crisis_k:   "crisis",
                trending_k: "trending",
                mr_k:       "mean_reverting",
            }
