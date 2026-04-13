"""
MLModel: drop-in stub for ML-based models (LSTM, XGBoost, etc.).

To implement a real model:
  1. Copy this file (e.g., lstm_model.py)
  2. Fill in fit() with your training logic
  3. Fill in predict() with your inference logic
  4. Change @model_register("ml") to your model name
  5. Import it in models/__init__.py

The stub returns zero signals so it is safe to run without training.
"""
import numpy as np
import pandas as pd
from .base_model import BaseModel
from .model_registry import model_register
from ..data.market_snapshot import MarketSnapshot
from .. import config


@model_register("ml")
class MLModel(BaseModel):
    """
    Skeleton for a supervised ML model.

    fit() should:
      - Build features from the historical df
      - Train your model (LSTM, XGBoost, sklearn, etc.)
      - Store the trained model as self._model

    predict() should:
      - Extract features from the snapshot
      - Run inference with self._model
      - Return {ticker: signal} in [-1, 1]
    """

    def __init__(self, tickers: list = None):
        self.tickers = tickers or config.TICKERS
        self._model  = None          # set in fit()
        self._buffer: dict = {t: [] for t in self.tickers}  # rolling feature buffer

    def fit(self, df: pd.DataFrame) -> None:
        """
        Train on historical data.
        df columns: tick, ticker, last, bid, ask, bid_size, ask_size, volume
        """
        # TODO: implement feature engineering + model training
        # Example skeleton:
        #   features, labels = build_features(df)
        #   self._model = XGBClassifier().fit(features, labels)
        print("[MLModel] fit() not implemented — using zero signals")

    def predict(self, snapshot: MarketSnapshot) -> dict:
        """Run inference. Returns zero signals until fit() is implemented."""
        if self._model is None:
            return {t: 0.0 for t in self.tickers}

        # TODO: build feature vector from snapshot and self._buffer
        # signals = self._model.predict(feature_vector)
        return {t: 0.0 for t in self.tickers}

    def update(self, snapshot: MarketSnapshot) -> None:
        """Update rolling buffer for feature construction."""
        for ticker in self.tickers:
            price = snapshot.last.get(ticker)
            if price is not None:
                self._buffer[ticker].append(price)

    def reset(self) -> None:
        self._buffer = {t: [] for t in self.tickers}
