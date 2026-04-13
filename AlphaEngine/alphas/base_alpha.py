"""
BaseAlpha: thin wrapper that connects a BaseModel to the SignalBus.

An alpha's job is simple:
  1. Call self.model.predict(snapshot)
  2. Return the resulting signal dict

All prediction logic lives in the model, not here.
This separation makes it trivial to swap the underlying model.
"""
from abc import ABC, abstractmethod
from ..data.market_snapshot import MarketSnapshot
from ..models.base_model import BaseModel


class BaseAlpha(ABC):

    name:    str  = ""       # set by @register decorator
    tickers: list = []

    def __init__(self, model: BaseModel):
        self.model = model

    def update(self, snapshot: MarketSnapshot) -> dict:
        """
        Called every tick. Returns {ticker: signal} in [-1, 1].
        Positive = buy pressure, negative = sell pressure, 0 = flat.
        """
        signals = self.model.predict(snapshot)
        self.model.update(snapshot)
        return signals

    def reset(self) -> None:
        self.model.reset()

    def get_params(self) -> dict:
        return self.model.get_params()

    def set_params(self, params: dict) -> None:
        self.model.set_params(params)
