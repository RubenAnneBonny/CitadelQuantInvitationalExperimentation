"""
MarketSnapshot: one tick's worth of market data for all tickers.
This is the single data structure that flows through the entire system —
feeds produce it, models consume it, the dashboard displays it.
"""
from dataclasses import dataclass, field


@dataclass
class MarketSnapshot:
    tick:      int
    last:      dict   # {ticker: float}
    bid:       dict   # {ticker: float}
    ask:       dict   # {ticker: float}
    bid_size:  dict   # {ticker: float}
    ask_size:  dict   # {ticker: float}
    volume:    dict   # {ticker: float}
    mid:       dict   # {ticker: float}  — computed: (bid+ask)/2

    # Enriched by the trading loop (not set by the feed)
    regime:    str   = "mean_reverting"
    zscore:    float = 0.0
    beta:      float = 1.0

    @classmethod
    def from_dicts(cls, tick: int, rows: dict) -> "MarketSnapshot":
        """
        Build a snapshot from a dict of {ticker: row_dict}.
        row_dict keys: last, bid, ask, bid_size, ask_size, volume
        """
        last     = {t: float(r.get("last", 0))     for t, r in rows.items()}
        bid      = {t: float(r.get("bid", 0))      for t, r in rows.items()}
        ask      = {t: float(r.get("ask", 0))      for t, r in rows.items()}
        bid_size = {t: float(r.get("bid_size", 0)) for t, r in rows.items()}
        ask_size = {t: float(r.get("ask_size", 0)) for t, r in rows.items()}
        volume   = {t: float(r.get("volume", 0))   for t, r in rows.items()}
        mid      = {
            t: (bid[t] + ask[t]) / 2 if bid[t] and ask[t] else last[t]
            for t in rows
        }
        return cls(
            tick=tick, last=last, bid=bid, ask=ask,
            bid_size=bid_size, ask_size=ask_size, volume=volume, mid=mid,
        )
