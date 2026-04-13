"""
BaseModel: abstract interface for all predictive models.

A model takes market data and outputs a trading signal per ticker.
Signals are floats in [-1, 1]:  +1 = strong buy, -1 = strong sell, 0 = flat.

To add a new model (e.g. LSTM, XGBoost):
  1. Create models/lstm_model.py
  2. Subclass BaseModel and implement fit() + predict()
  3. Register with @model_register("lstm")
  4. Use it in an Alpha: alpha.model = LSTMModel()
"""
from abc import ABC, abstractmethod
import pandas as pd
from ..data.market_snapshot import MarketSnapshot


class BaseModel(ABC):

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """
        Train the model on historical data.
        df: Rotman-format DataFrame with columns tick, ticker, last, bid, ask, ...
        For online/stateless models this can be a no-op.
        """
        ...

    @abstractmethod
    def predict(self, snapshot: MarketSnapshot) -> dict:
        """
        Generate signals for the current tick.
        Returns {ticker: signal} where signal in [-1.0, 1.0].
        """
        ...

    def update(self, snapshot: MarketSnapshot) -> None:
        """
        Optional: incrementally update internal state each tick.
        Called by the alpha after predict() in the live loop.
        Override for online-learning models.
        """
        pass

    def reset(self) -> None:
        """Reset internal state. Called before backtest replay."""
        pass

    def get_params(self) -> dict:
        """Return optimizable hyperparameters. Override to enable grid search."""
        return {}

    def set_params(self, params: dict) -> None:
        """Apply hyperparameter dict. Override to enable grid search."""
        pass
