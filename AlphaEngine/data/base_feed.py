"""
BaseFeed: abstract interface for all data sources.

Implementations:
  RITClient  — live feed from the Rotman RIT REST API
  CSVFeed    — replay any Rotman-format CSV file

To swap data source, change one line in your script:
  feed = RITClient()               # live trading
  feed = CSVFeed("session.csv")    # backtesting / replay
"""
from abc import ABC, abstractmethod
from .market_snapshot import MarketSnapshot


class BaseFeed(ABC):

    @abstractmethod
    def next_snapshot(self) -> MarketSnapshot | None:
        """
        Return the next MarketSnapshot, or None when the feed is exhausted.
        For live feeds this blocks until a new tick arrives.
        """
        ...

    def reset(self) -> None:
        """Restart the feed from the beginning (used in backtesting)."""
        pass
