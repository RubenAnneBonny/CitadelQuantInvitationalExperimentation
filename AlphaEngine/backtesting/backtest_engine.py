"""
BacktestEngine: replay any BaseFeed and evaluate a BaseAlpha.

Usage:
  feed   = CSVFeed("rit_full_session.csv")
  alpha  = BollingerAlpha()
  engine = BacktestEngine(feed)
  result = engine.run(alpha)
  print(result)
"""
import numpy as np
from ..data.base_feed import BaseFeed
from ..alphas.base_alpha import BaseAlpha
from ..pnl.pnl_tracker import PnLTracker
from .backtest_result import BacktestResult
from .. import config


class BacktestEngine:

    def __init__(
        self,
        feed:          BaseFeed,
        trade_size:    int   = config.BT_TRADE_SIZE,
        slippage_bps:  float = config.SLIPPAGE_BPS,
        max_position:  int   = config.MAX_POSITION,
        signal_thresh: float = config.SIGNAL_THRESHOLD,
    ):
        self._feed          = feed
        self.trade_size     = trade_size
        self.slippage_bps   = slippage_bps
        self.max_position   = max_position
        self.signal_thresh  = signal_thresh

    def run(self, alpha: BaseAlpha) -> BacktestResult:
        """Replay the feed and simulate fills. Returns a BacktestResult."""
        self._feed.reset()
        alpha.reset()

        pnl       = PnLTracker(alpha.tickers)
        positions = {t: 0 for t in alpha.tickers}
        trades    = []

        while True:
            snap = self._feed.next_snapshot()
            if snap is None:
                break

            signals = alpha.update(snap)

            for ticker in alpha.tickers:
                sig = signals.get(ticker, 0.0)
                mid = snap.mid.get(ticker)
                if mid is None:
                    continue

                pos = positions[ticker]
                action, qty = self._signal_to_action(sig, pos)
                if action is None:
                    continue

                slippage  = mid * self.slippage_bps / 10_000
                fill_price = mid + slippage if action == "BUY" else mid - slippage

                pnl.record_fill(ticker, action, qty, fill_price)
                positions[ticker] = pos + qty if action == "BUY" else pos - qty
                trades.append((snap.tick, ticker, action, qty, fill_price))

            pnl.update_prices(snap.mid, tick=snap.tick)

        result = BacktestResult(
            alpha_name   = alpha.name,
            params       = alpha.get_params(),
            equity_curve = pnl.equity_curve(),
            trades       = trades,
        )
        result.compute_metrics()
        return result

    def _signal_to_action(self, signal: float, position: int):
        """Convert a signal float to (action, qty) or (None, 0)."""
        if signal > self.signal_thresh and position < self.max_position:
            qty = min(self.trade_size, self.max_position - position)
            return "BUY", qty
        elif signal < -self.signal_thresh and position > -self.max_position:
            qty = min(self.trade_size, self.max_position + position)
            return "SELL", qty
        elif signal < 0 and position > 0:
            qty = min(abs(position), self.trade_size)
            return "SELL", qty
        elif signal > 0 and position < 0:
            qty = min(abs(position), self.trade_size)
            return "BUY", qty
        return None, 0
