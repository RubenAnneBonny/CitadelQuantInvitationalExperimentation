"""
RITClient: live data feed + order execution for the Rotman RIT REST API.

Implements BaseFeed so it can be used interchangeably with CSVFeed.
Also exposes order-management methods used by OrderManager.
"""
import time
import requests
from .base_feed import BaseFeed
from .market_snapshot import MarketSnapshot
from .. import config


class RITClient(BaseFeed):

    def __init__(self, api_key: str = config.API_KEY, base: str = config.BASE):
        self._session = requests.Session()
        self._session.headers.update({"X-API-Key": api_key})
        self._base     = base
        self._last_tick = -1

    # ── BaseFeed ──────────────────────────────────────────────────────────────

    def next_snapshot(self) -> MarketSnapshot | None:
        """
        Blocks until a new tick arrives, then returns a snapshot.
        Returns None when the case ends.
        """
        while True:
            case = self.get_case()
            if case["status"] != "ACTIVE":
                return None
            tick = case["tick"]
            if tick == self._last_tick:
                time.sleep(config.POLL_INTERVAL)
                continue
            self._last_tick = tick
            return self._build_snapshot(tick)

    def reset(self) -> None:
        self._last_tick = -1

    # ── Case / securities ─────────────────────────────────────────────────────

    def get_case(self) -> dict:
        return self._session.get(f"{self._base}/case").json()

    def get_securities(self) -> list:
        return self._session.get(f"{self._base}/securities").json()

    def get_book(self, ticker: str) -> dict:
        return self._session.get(
            f"{self._base}/securities/book", params={"ticker": ticker}
        ).json()

    def get_positions(self, tickers: list = None) -> dict:
        tickers = tickers or config.TICKERS
        return {
            s["ticker"]: s["position"]
            for s in self.get_securities()
            if s["ticker"] in tickers
        }

    def get_mid(self, ticker: str) -> tuple:
        """Returns (mid, bid, ask) or (None, None, None) if book is empty."""
        book = self.get_book(ticker)
        bid  = book["bids"][0]["price"] if book["bids"] else None
        ask  = book["asks"][0]["price"] if book["asks"] else None
        if bid is None or ask is None:
            return None, None, None
        return (bid + ask) / 2, bid, ask

    # ── Orders ────────────────────────────────────────────────────────────────

    def place_limit(self, ticker: str, action: str, qty: int, price: float) -> dict:
        """Place a LIMIT order. action = 'BUY' or 'SELL'."""
        return self._session.post(f"{self._base}/orders", params={
            "ticker":   ticker,
            "type":     "LIMIT",
            "quantity": qty,
            "action":   action,
            "price":    round(price, 2),
        }).json()

    def cancel_all(self) -> None:
        try:
            self._session.post(f"{self._base}/commands/cancel", params={"all": 1})
        except Exception as e:
            print(f"[WARN] cancel_all: {e}")

    def flatten_ticker(self, ticker: str) -> None:
        pos = self.get_positions([ticker]).get(ticker, 0)
        if pos == 0:
            return
        qty    = min(abs(int(pos)), 25_000)
        action = "SELL" if pos > 0 else "BUY"
        mid, bid, ask = self.get_mid(ticker)
        price = bid if action == "SELL" else ask
        if price is None:
            return
        print(f"  [FLATTEN] {action} {qty} {ticker} @ {price:.2f}")
        self.place_limit(ticker, action, qty, price)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_snapshot(self, tick: int) -> MarketSnapshot:
        securities = self.get_securities()
        rows = {}
        for s in securities:
            if s["ticker"] in config.TICKERS:
                mid_val, bid_val, ask_val = self.get_mid(s["ticker"])
                rows[s["ticker"]] = {
                    "last":     s.get("last", 0),
                    "bid":      bid_val or s.get("bid", 0),
                    "ask":      ask_val or s.get("ask", 0),
                    "bid_size": s.get("bid_size", 0),
                    "ask_size": s.get("ask_size", 0),
                    "volume":   s.get("volume", 0),
                }
        return MarketSnapshot.from_dicts(tick, rows)
