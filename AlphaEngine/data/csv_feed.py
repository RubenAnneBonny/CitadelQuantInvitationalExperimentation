"""
CSVFeed: replay any Rotman-format CSV file as a BaseFeed.

Expected CSV columns (long format, one row per ticker per tick):
  tick, ticker, last, bid, ask, bid_size, ask_size, volume

Optional columns (ignored but accepted): position, unrealized

Usage:
  feed = CSVFeed("Training_before_comp/rit_full_session.csv")
  while (snap := feed.next_snapshot()) is not None:
      process(snap)
  feed.reset()  # replay from start
"""
import pandas as pd
from .base_feed import BaseFeed
from .market_snapshot import MarketSnapshot
from .. import config


class CSVFeed(BaseFeed):

    def __init__(self, csv_path: str = config.DEFAULT_CSV, tickers: list = None):
        self._path    = csv_path
        self._tickers = tickers or config.TICKERS
        self._ticks   = []     # sorted list of tick numbers
        self._data    = {}     # {tick: {ticker: row_dict}}
        self._idx     = 0
        self._load()

    def _load(self) -> None:
        df = pd.read_csv(self._path)
        df = df[df["ticker"].isin(self._tickers)].copy()
        df["tick"] = df["tick"].astype(int)
        df = df.sort_values("tick")

        for tick, group in df.groupby("tick"):
            rows = {}
            for _, row in group.iterrows():
                ticker = row["ticker"]
                rows[ticker] = {
                    "last":     row.get("last", 0),
                    "bid":      row.get("bid", 0),
                    "ask":      row.get("ask", 0),
                    "bid_size": row.get("bid_size", 0),
                    "ask_size": row.get("ask_size", 0),
                    "volume":   row.get("volume", 0),
                }
            # Only include ticks where all tickers are present
            if all(t in rows for t in self._tickers):
                self._data[int(tick)] = rows

        self._ticks = sorted(self._data.keys())
        self._idx   = 0

    def next_snapshot(self) -> MarketSnapshot | None:
        if self._idx >= len(self._ticks):
            return None
        tick   = self._ticks[self._idx]
        self._idx += 1
        return MarketSnapshot.from_dicts(tick, self._data[tick])

    def reset(self) -> None:
        self._idx = 0

    def __len__(self) -> int:
        return len(self._ticks)
