"""
MomentumAlpha: simple MA-crossover directional signal.

Signal: +1 if short MA > long MA (uptrend), -1 if short MA < long MA (downtrend).
"""
import numpy as np
import pandas as pd
from .base_alpha import BaseAlpha
from .alpha_registry import register
from ..models.base_model import BaseModel
from ..models.model_registry import model_register
from ..data.market_snapshot import MarketSnapshot
from .. import config


@model_register("momentum")
class MomentumModel(BaseModel):

    def __init__(self, short: int = config.MOM_SHORT, long: int = config.MOM_LONG,
                 tickers: list = None):
        self.short   = short
        self.long    = long
        self.tickers = tickers or config.TICKERS
        self._history: dict = {t: [] for t in self.tickers}

    def fit(self, df: pd.DataFrame) -> None:
        pass

    def predict(self, snapshot: MarketSnapshot) -> dict:
        signals = {}
        for ticker in self.tickers:
            price = snapshot.last.get(ticker)
            if price is None:
                signals[ticker] = 0.0
                continue
            self._history[ticker].append(price)
            hist = self._history[ticker]
            if len(hist) < self.long:
                signals[ticker] = 0.0
                continue
            ma_short = np.mean(hist[-self.short:])
            ma_long  = np.mean(hist[-self.long:])
            if ma_long == 0:
                signals[ticker] = 0.0
                continue
            # Normalise by long MA so different price scales are comparable
            raw = (ma_short - ma_long) / ma_long
            signals[ticker] = float(np.clip(raw * 100, -1.0, 1.0))
        return signals

    def reset(self) -> None:
        self._history = {t: [] for t in self.tickers}

    def get_params(self) -> dict:
        return {"short": self.short, "long": self.long}

    def set_params(self, params: dict) -> None:
        self.short = params.get("short", self.short)
        self.long  = params.get("long",  self.long)


@register("momentum")
class MomentumAlpha(BaseAlpha):
    """MA-crossover trend-following alpha."""

    tickers = config.TICKERS

    def __init__(self):
        super().__init__(model=MomentumModel())
