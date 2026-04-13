from .base_alpha import BaseAlpha
from .alpha_registry import register
from ..models.bollinger_model import BollingerModel
from .. import config


@register("bollinger")
class BollingerAlpha(BaseAlpha):
    """Bollinger-band mean-reversion alpha. Active in mean_reverting regime."""

    tickers = config.TICKERS

    def __init__(self):
        super().__init__(model=BollingerModel())
