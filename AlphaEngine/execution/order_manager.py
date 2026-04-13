"""
OrderManager: wraps RITClient order placement and notifies PnLTracker of fills.

Assumes limit orders at mid price fill immediately (valid for small sizes
in a competition with thin books). For more accuracy, poll order status.
"""
from ..data.rit_client import RITClient
from ..pnl.pnl_tracker import PnLTracker
from .. import config


class OrderManager:

    def __init__(self, client: RITClient, pnl: PnLTracker):
        self._client = client
        self._pnl    = pnl

    def buy(self, ticker: str, qty: int, price: float) -> bool:
        """Place a BUY limit order and record the expected fill."""
        if qty <= 0:
            return False
        result = self._client.place_limit(ticker, "BUY", qty, price)
        if result:
            self._pnl.record_fill(ticker, "BUY", qty, price)
            print(f"  [ORDER] BUY  {qty:>5} {ticker} @ {price:.2f}")
        return bool(result)

    def sell(self, ticker: str, qty: int, price: float) -> bool:
        """Place a SELL limit order and record the expected fill."""
        if qty <= 0:
            return False
        result = self._client.place_limit(ticker, "SELL", qty, price)
        if result:
            self._pnl.record_fill(ticker, "SELL", qty, price)
            print(f"  [ORDER] SELL {qty:>5} {ticker} @ {price:.2f}")
        return bool(result)

    def cancel_all(self) -> None:
        self._client.cancel_all()

    def flatten_all(self, tickers: list = None) -> None:
        tickers = tickers or config.TICKERS
        self.cancel_all()
        for ticker in tickers:
            self._client.flatten_ticker(ticker)
