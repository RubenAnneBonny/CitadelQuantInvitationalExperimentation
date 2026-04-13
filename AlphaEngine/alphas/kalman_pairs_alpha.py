from .base_alpha import BaseAlpha
from .alpha_registry import register
from ..models.kalman_model import KalmanModel
from .. import config


@register("kalman_pairs")
class KalmanPairsAlpha(BaseAlpha):
    """Kalman-filter pairs-trading alpha. Active in all regimes."""

    tickers = list(config.PAIR)

    def __init__(self):
        super().__init__(model=KalmanModel())

    @property
    def last_zscore(self) -> float:
        return self.model.last_zscore

    @property
    def last_beta(self) -> float:
        return self.model.last_beta
