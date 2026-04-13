"""
PnLTracker: algo-only P&L tracking.

Maintains a completely independent position ledger built only from fills
placed by AlphaEngine. Rotman's API unrealized field is never used,
so manual trades through the RIT UI do not contaminate this tracker.

Realized P&L uses FIFO matching.
Unrealized P&L = (current mid - avg_cost) × net_position.
Equity curve = list of (tick, total_pnl) for charting.
"""
from collections import deque
from .. import config


class _FIFOLedger:
    """FIFO lot queue for one ticker."""

    def __init__(self):
        self._lots: deque = deque()   # each entry: (qty, fill_price)
        self.realized = 0.0

    def buy(self, qty: int, price: float) -> None:
        self._lots.append((qty, price))

    def sell(self, qty: int, price: float) -> None:
        remaining = qty
        while remaining > 0 and self._lots:
            lot_qty, lot_price = self._lots[0]
            matched = min(remaining, lot_qty)
            self.realized += matched * (price - lot_price)
            remaining -= matched
            if matched == lot_qty:
                self._lots.popleft()
            else:
                self._lots[0] = (lot_qty - matched, lot_price)
        # If we sold more than we held (short), add to front at fill price
        if remaining > 0:
            self._lots.appendleft((-remaining, price))

    @property
    def net_position(self) -> int:
        return sum(q for q, _ in self._lots)

    @property
    def avg_cost(self) -> float:
        total_qty = sum(abs(q) for q, _ in self._lots)
        if total_qty == 0:
            return 0.0
        return sum(abs(q) * p for q, p in self._lots) / total_qty

    def unrealized(self, current_mid: float) -> float:
        pos = self.net_position
        if pos == 0 or self.avg_cost == 0:
            return 0.0
        return pos * (current_mid - self.avg_cost)


class PnLTracker:

    def __init__(self, tickers: list = None):
        self.tickers = tickers or config.TICKERS
        self._ledgers: dict = {t: _FIFOLedger() for t in self.tickers}
        self._mid: dict     = {t: 0.0            for t in self.tickers}
        self._equity_curve: list = []   # [(tick, total_pnl)]
        self._tick = 0

    def record_fill(self, ticker: str, action: str, qty: int, fill_price: float) -> None:
        """
        Called by OrderManager for every algo-placed fill.
        action: 'BUY' or 'SELL'
        """
        if ticker not in self._ledgers:
            return
        if action == "BUY":
            self._ledgers[ticker].buy(qty, fill_price)
        else:
            self._ledgers[ticker].sell(qty, fill_price)

    def update_prices(self, prices: dict, tick: int = None) -> None:
        """
        Called each tick with latest mid prices.
        Recomputes unrealized PnL and appends to equity curve.
        """
        for t, p in prices.items():
            if t in self._mid:
                self._mid[t] = p
        if tick is not None:
            self._tick = tick
        self._equity_curve.append((self._tick, self.total_pnl()))

    def realized_pnl(self) -> float:
        return sum(l.realized for l in self._ledgers.values())

    def unrealized_pnl(self) -> float:
        return sum(
            l.unrealized(self._mid[t])
            for t, l in self._ledgers.items()
        )

    def total_pnl(self) -> float:
        return self.realized_pnl() + self.unrealized_pnl()

    def equity_curve(self) -> list:
        """List of (tick, total_pnl) for the dashboard equity chart."""
        return list(self._equity_curve)

    def per_ticker_pnl(self) -> dict:
        """
        Returns {ticker: {realized, unrealized, position, avg_cost}} for the positions table.
        """
        result = {}
        for t, l in self._ledgers.items():
            result[t] = {
                "realized":   l.realized,
                "unrealized": l.unrealized(self._mid[t]),
                "position":   l.net_position,
                "avg_cost":   l.avg_cost,
            }
        return result

    def reset(self) -> None:
        self._ledgers      = {t: _FIFOLedger() for t in self.tickers}
        self._mid          = {t: 0.0            for t in self.tickers}
        self._equity_curve = []
        self._tick         = 0
