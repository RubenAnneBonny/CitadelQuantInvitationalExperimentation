"""
BollingerModel: Bollinger-band mean-reversion signals.

Signal logic (ported from Trading_Strategy/strategy.py):
  price < mean - width*std  →  +1.0 (oversold, buy)
  price > mean + width*std  →  -1.0 (overbought, sell)
  otherwise                 →   0.0 (flat)

Signal magnitude scales with how far price is outside the band.
"""
import numpy as np
import pandas as pd
from .base_model import BaseModel
from .model_registry import model_register
from ..data.market_snapshot import MarketSnapshot
from .. import config


@model_register("bollinger")
class BollingerModel(BaseModel):

    def __init__(self, window: int = config.BB_WINDOW, width: float = config.BB_WIDTH,
                 tickers: list = None):
        self.window   = window
        self.width    = width
        self.tickers  = tickers or config.TICKERS
        self._history: dict = {t: [] for t in self.tickers}

    def fit(self, df: pd.DataFrame) -> None:
        pass  # no offline training needed

    def predict(self, snapshot: MarketSnapshot) -> dict:
        signals = {}
        for ticker in self.tickers:
            price = snapshot.last.get(ticker)
            if price is None:
                signals[ticker] = 0.0
                continue
            self._history[ticker].append(price)
            arr = np.array(self._history[ticker][-self.window:])
            if len(arr) < self.window or arr.std() == 0:
                signals[ticker] = 0.0
                continue
            mean = arr.mean()
            std  = arr.std()
            z    = (price - mean) / (self.width * std)
            # Clip to [-1, 1]; negative z (below band) → positive (buy) signal
            signals[ticker] = float(np.clip(-z, -1.0, 1.0))
        return signals

    def reset(self) -> None:
        self._history = {t: [] for t in self.tickers}

    def get_params(self) -> dict:
        return {"window": self.window, "width": self.width}

    def set_params(self, params: dict) -> None:
        self.window = params.get("window", self.window)
        self.width  = params.get("width", self.width)
