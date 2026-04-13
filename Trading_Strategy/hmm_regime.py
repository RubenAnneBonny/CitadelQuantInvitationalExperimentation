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

def _safe_cov(X_k, D, fallback_cov):
    """Covariance of cluster X_k; falls back to fallback_cov when too few points."""
    if len(X_k) < 2:
        return fallback_cov.copy()
    c = np.cov(X_k.T)
    if c.ndim == 0:          # single feature edge case
        c = np.array([[float(c)]])
    c = np.where(np.isfinite(c), c, 0.0)
    return c + np.eye(D) * 1e-4


class HMM:
    def __init__(self, K, X):
        kmeans = KMeans(n_clusters=K, n_init=10).fit(X)
        labels = kmeans.labels_
        D = X.shape[1]

        self.X = X
        self.K = K

        fallback = np.cov(X.T) + np.eye(D) * 1e-4   # whole-dataset cov as fallback

        self.mu = np.array([
            X[labels == k].mean(axis=0) if (labels == k).any() else X.mean(axis=0)
            for k in range(K)
        ])
        self.Sigma = np.array([
            _safe_cov(X[labels == k], D, fallback) for k in range(K)
        ])
        counts = np.array([(labels == k).sum() for k in range(K)], dtype=float)
        self.pi = np.maximum(counts / counts.sum(), 1e-6)
        self.pi /= self.pi.sum()
        self.A  = np.full((K, K), 1 / K)

    def _emission(self, x, k):
        try:
            v = multivariate_normal.pdf(x, mean=self.mu[k], cov=self.Sigma[k])
            return v if np.isfinite(v) and v > 0 else 1e-300
        except Exception:
            return 1e-300

    def _forward(self):
        T     = len(self.X)
        alpha = np.zeros((T, self.K))
        for k in range(self.K):
            alpha[0, k] = self.pi[k] * self._emission(self.X[0], k)
        # Normalise each row to prevent underflow over long sequences
        row = alpha[0].sum()
        if row > 0:
            alpha[0] /= row
        for t in range(1, T):
            for k in range(self.K):
                alpha[t, k] = (
                    self._emission(self.X[t], k)
                    * np.sum(alpha[t - 1] * self.A[:, k])
                )
            row = alpha[t].sum()
            if row > 0:
                alpha[t] /= row
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
        D = self.X.shape[1]
        self.pi = np.maximum(gamma[0], 1e-6)
        self.pi /= self.pi.sum()

        denom_A = gamma[:-1].sum(axis=0, keepdims=True).T
        self.A  = xi.sum(axis=0) / np.maximum(denom_A, 1e-300)
        # Ensure rows sum to 1 and no NaNs
        row_sums = self.A.sum(axis=1, keepdims=True)
        self.A   = np.where(row_sums > 0, self.A / row_sums, 1 / self.K)

        fallback = np.cov(self.X.T) + np.eye(D) * 1e-4
        for k in range(self.K):
            w     = gamma[:, k]
            w_sum = w.sum()
            if w_sum < 1e-6:   # collapsed cluster — reset to global stats
                self.mu[k]    = self.X.mean(axis=0)
                self.Sigma[k] = fallback.copy()
                continue
            self.mu[k] = (w[:, None] * self.X).sum(axis=0) / w_sum
            diff       = self.X - self.mu[k]
            S          = (w[:, None] * diff).T @ diff / w_sum + np.eye(D) * 1e-4
            self.Sigma[k] = np.where(np.isfinite(S), S, fallback)

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
            v = multivariate_normal.pdf(X_new[0], mean=self.mu[k], cov=self.Sigma[k])
            alpha[0, k] = self.pi[k] * (v if np.isfinite(v) else 1e-300)
        row = alpha[0].sum()
        if row > 0:
            alpha[0] /= row
        for t in range(1, T):
            for k in range(self.K):
                v = multivariate_normal.pdf(X_new[t], mean=self.mu[k], cov=self.Sigma[k])
                alpha[t, k] = (v if np.isfinite(v) else 1e-300) * np.sum(alpha[t - 1] * self.A[:, k])
            row = alpha[t].sum()
            if row > 0:
                alpha[t] /= row
        gamma = alpha / (alpha.sum(axis=1, keepdims=True) + 1e-300)
        return gamma


# ── Online wrapper ────────────────────────────────────────────────────────────

MOM_WINDOW   = 10    # ticks for rolling momentum
VOL_WINDOW   = 10    # ticks for rolling volatility
MIN_SAMPLES  = 30    # minimum observations before first HMM fit
REFIT_EVERY  = 30    # re-fit HMM every N new ticks after initial fit


class OnlineHMMRegime:
    """
    Two-regime HMM (trending / mean_reverting) fitted on (momentum, volatility).

    Features are deliberately kept to 2D and K=2 so there is always a clear
    binary split even with limited tick data. K=3 consistently collapsed to
    one dominant regime on 30-100 tick samples.

    Regimes are labelled after each fit by comparing the mean momentum of the
    two states — the state with higher |mean momentum| is 'trending'.
    """

    def __init__(self):
        self.hmm              = None
        self.label_map        = {}
        self._prices          = []   # raw prices for return / momentum computation
        self._features        = []   # (momentum, volatility) vectors
        self._ticks_since_fit = 0
        self.current_regime   = "mean_reverting"   # safe default while warming up

    def update(self, price: float) -> str:
        """
        Ingest one new price tick. Returns the current regime label immediately
        (defaults to 'mean_reverting' during warmup).
        """
        self._prices.append(price)
        n_prices = len(self._prices)

        if n_prices < 2:
            return self.current_regime

        # Rolling momentum: mean of last MOM_WINDOW single-tick returns
        recent = self._prices[-MOM_WINDOW - 1:]
        rets   = np.diff(recent) / (np.array(recent[:-1]) + 1e-9)
        mom    = float(rets.mean())

        # Rolling volatility: std of those same returns
        vol    = float(rets.std()) if len(rets) > 1 else 0.0

        self._features.append(np.array([mom, vol]))
        self._ticks_since_fit += 1

        n = len(self._features)
        if n < MIN_SAMPLES:
            return self.current_regime

        # Fit / refit
        if self.hmm is None or self._ticks_since_fit >= REFIT_EVERY:
            self._refit()
            self._ticks_since_fit = 0

        # Classify current tick
        X     = np.array(self._features)
        gamma = self.hmm.predict_states(X)
        state = int(gamma[-1].argmax())
        self.current_regime = self.label_map.get(state, "mean_reverting")
        return self.current_regime

    def _refit(self):
        X        = np.array(self._features)
        self.hmm = HMM(K=2, X=X)
        # Bias A toward persistence — regimes should last multiple ticks
        self.hmm.A = np.array([[0.90, 0.10],
                                [0.10, 0.90]])
        self.hmm._fit()
        self._build_label_map()

    def _build_label_map(self):
        """
        State with higher |mean momentum| (feature 0) → 'trending'.
        Other state → 'mean_reverting'.
        """
        mom0 = abs(self.hmm.mu[0][0])
        mom1 = abs(self.hmm.mu[1][0])
        if mom0 >= mom1:
            self.label_map = {0: "trending", 1: "mean_reverting"}
        else:
            self.label_map = {0: "mean_reverting", 1: "trending"}
        print(f"  [HMM refit]  state0: mom={self.hmm.mu[0][0]:+.5f} vol={self.hmm.mu[0][1]:.5f} → {self.label_map[0]}"
              f"  |  state1: mom={self.hmm.mu[1][0]:+.5f} vol={self.hmm.mu[1][1]:.5f} → {self.label_map[1]}")
