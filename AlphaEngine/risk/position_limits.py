"""
PositionLimits: enforces per-ticker position caps before any order is placed.
"""
from .. import config


class PositionLimits:

    def __init__(self, max_position: int = config.MAX_POSITION):
        self.max_position = max_position

    def can_buy(self, ticker: str, qty: int, current_position: int) -> int:
        """Return the actual qty we're allowed to buy (may be less than requested)."""
        headroom = self.max_position - current_position
        return max(0, min(qty, headroom))

    def can_sell(self, ticker: str, qty: int, current_position: int) -> int:
        """Return the actual qty we're allowed to sell (may be less than requested)."""
        headroom = self.max_position + current_position   # max_position - (-current)
        return max(0, min(qty, headroom))
