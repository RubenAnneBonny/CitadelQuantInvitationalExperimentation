"""
KalmanModel: Kalman-filter pairs-trading signals.

Wraps Trading_Strategy/kalman_pairs.py (KalmanPairFilter).

Signal logic:
  zscore > PAIRS_ENTRY_Z  →  short spread: sell y, buy x
    CRZY signal = -1, TAME signal = +1
  zscore < -PAIRS_ENTRY_Z →  long spread: buy y, sell x
    CRZY signal = +1, TAME signal = -1
  |zscore| < PAIRS_EXIT_Z →  flat: both signals = 0

Signal is proportional to zscore magnitude between exit and entry thresholds.
"""
import os, sys
import pandas as pd
import numpy as np

# Add Trading_Strategy to path so we can import KalmanPairFilter
_TRADING_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Trading_Strategy")
if _TRADING_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_TRADING_DIR))

from kalman_pairs import KalmanPairFilter  # noqa: E402
from .base_model import BaseModel
from .model_registry import model_register
from ..data.market_snapshot import MarketSnapshot
from .. import config


@model_register("kalman")
class KalmanModel(BaseModel):

    def __init__(self, delta: float = config.KF_DELTA,
                 entry_z: float = config.PAIRS_ENTRY_Z,
                 exit_z: float  = config.PAIRS_EXIT_Z,
                 min_obs: int   = config.PAIRS_MIN_OBS,
                 pair: tuple    = None):
        self.delta   = delta
        self.entry_z = entry_z
        self.exit_z  = exit_z
        self.min_obs = min_obs
        self.pair    = pair or config.PAIR   # (y_ticker, x_ticker)
        self._kf     = KalmanPairFilter(delta=self.delta)

    def fit(self, df: pd.DataFrame) -> None:
        pass  # online model, no batch training

    def predict(self, snapshot: MarketSnapshot) -> dict:
        ticker_y, ticker_x = self.pair
        y = snapshot.last.get(ticker_y)
        x = snapshot.last.get(ticker_x)
        if y is None or x is None:
            return {ticker_y: 0.0, ticker_x: 0.0}

        kf_state = self._kf.update(y=y, x=x)
        zscore   = kf_state["zscore"]

        if self._kf.n_obs < self.min_obs:
            return {ticker_y: 0.0, ticker_x: 0.0}

        # Normalise signal magnitude between exit_z and entry_z
        abs_z = abs(zscore)
        if abs_z < self.exit_z:
            raw = 0.0
        else:
            raw = np.clip((abs_z - self.exit_z) / (self.entry_z - self.exit_z), 0.0, 1.0)

        direction = -1.0 if zscore > 0 else 1.0  # positive z → sell y
        signal_y  = direction * raw
        signal_x  = -signal_y  # hedge leg is opposite

        return {ticker_y: float(signal_y), ticker_x: float(signal_x)}

    @property
    def last_zscore(self) -> float:
        if not self._kf.zscores:
            return 0.0
        return self._kf.zscores[-1]

    @property
    def last_beta(self) -> float:
        if not self._kf.betas:
            return 1.0
        return self._kf.betas[-1]

    def reset(self) -> None:
        self._kf = KalmanPairFilter(delta=self.delta)

    def get_params(self) -> dict:
        return {"entry_z": self.entry_z, "exit_z": self.exit_z}

    def set_params(self, params: dict) -> None:
        self.entry_z = params.get("entry_z", self.entry_z)
        self.exit_z  = params.get("exit_z",  self.exit_z)
