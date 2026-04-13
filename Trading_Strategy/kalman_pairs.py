"""
Kalman Filter for Pairs Trading — Rolling Beta Estimator
---------------------------------------------------------
Models the relationship between two assets as a linear regression with
time-varying coefficients estimated online via a Kalman filter:

    y_t = alpha_t + beta_t * x_t + epsilon_t

State:       theta_t = [alpha_t, beta_t]
Transition:  theta_t = theta_{t-1} + process_noise   (random walk on coefficients)
Observation: y_t = H_t @ theta_t + obs_noise
             where H_t = [1, x_t]

The innovation e_t = y_t - H_t @ theta_{t-1} is the model's estimate of the
current spread. Normalising by sqrt(Q_t) gives a z-score suitable for
entry/exit signals.

delta controls how fast the hedge ratio is allowed to drift:
  - small delta (e.g. 1e-4): slow adaptation, stable beta
  - large delta (e.g. 1e-2): fast adaptation, more reactive beta
"""

import numpy as np


class KalmanPairFilter:
    def __init__(self, delta: float = 1e-3, obs_noise: float = 1.0):
        """
        Parameters
        ----------
        delta     : controls process noise (how fast alpha/beta can drift)
                    Vw = delta / (1 - delta)
        obs_noise : initial observation noise variance (R); auto-updated online
        """
        self.delta = delta
        Vw = delta / (1 - delta)

        # State: [alpha, beta]
        self.theta = np.zeros(2)

        # State covariance
        self.P = np.eye(2)

        # Process noise covariance (fixed, diagonal)
        self.Q = Vw * np.eye(2)

        # Observation noise variance (scalar, updated online via running variance)
        self.R = obs_noise

        # Running stats for spread variance (used to keep R calibrated)
        self._spread_var_n   = 0
        self._spread_var_m   = 0.0   # Welford mean
        self._spread_var_M2  = 0.0   # Welford M2

        # History for diagnostics / signal generation
        self.spreads    = []   # raw innovations e_t
        self.zscores    = []   # e_t / sqrt(Q_t)
        self.betas      = []   # beta estimates
        self.alphas     = []   # alpha estimates

    # ── Online update ────────────────────────────────────────────────────────

    def update(self, y: float, x: float) -> dict:
        """
        Ingest one new (x, y) observation and return current spread statistics.

        Parameters
        ----------
        y : price of asset A (the "dependent" leg, e.g. CRZY)
        x : price of asset B (the "independent" leg, e.g. TAME)

        Returns
        -------
        dict with keys: alpha, beta, spread (innovation), zscore, spread_std
        """
        H = np.array([1.0, x])

        # ── Predict ──────────────────────────────────────────────────────────
        # Coefficients are a random walk so prediction = previous estimate
        theta_pred = self.theta.copy()
        P_pred     = self.P + self.Q

        # ── Innovation (spread) ──────────────────────────────────────────────
        y_hat  = H @ theta_pred
        e      = y - y_hat                   # raw spread

        # Innovation variance (scalar)
        Q_t    = float(H @ P_pred @ H) + self.R

        # ── Kalman gain ───────────────────────────────────────────────────────
        K = (P_pred @ H) / Q_t              # shape (2,)

        # ── Update ────────────────────────────────────────────────────────────
        self.theta = theta_pred + K * e
        self.P     = (np.eye(2) - np.outer(K, H)) @ P_pred

        # ── Update observation noise R via Welford online variance ────────────
        self._spread_var_n  += 1
        delta_mean           = e - self._spread_var_m
        self._spread_var_m  += delta_mean / self._spread_var_n
        self._spread_var_M2 += delta_mean * (e - self._spread_var_m)
        if self._spread_var_n >= 2:
            self.R = max(self._spread_var_M2 / (self._spread_var_n - 1), 1e-6)

        # ── z-score ──────────────────────────────────────────────────────────
        spread_std = np.sqrt(Q_t)
        zscore     = e / spread_std if spread_std > 0 else 0.0

        # ── Store history ────────────────────────────────────────────────────
        self.spreads.append(e)
        self.zscores.append(zscore)
        self.betas.append(self.theta[1])
        self.alphas.append(self.theta[0])

        return {
            "alpha":      self.theta[0],
            "beta":       self.theta[1],
            "spread":     e,
            "zscore":     zscore,
            "spread_std": spread_std,
        }

    # ── Convenience properties ───────────────────────────────────────────────

    @property
    def current_beta(self) -> float:
        return self.theta[1]

    @property
    def current_alpha(self) -> float:
        return self.theta[0]

    @property
    def n_obs(self) -> int:
        return self._spread_var_n
