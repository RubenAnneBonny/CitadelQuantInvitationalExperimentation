"""
Executor: converts SignalBus signals into limit orders each tick.

Signal → action rules:
  combined signal > +SIGNAL_THRESHOLD and position < MAX_POSITION  → BUY
  combined signal < -SIGNAL_THRESHOLD and position > -MAX_POSITION → SELL
  combined signal flips sign and position non-zero                  → CLOSE

Respects:
  - Trade cooldown (TRADE_COOLDOWN ticks between orders per ticker)
  - Position limits (MAX_POSITION)
  - Kill switch (no orders if halted)
  - Stop-trading tick (flatten all and halt)
"""
from ..execution.signal_bus import SignalBus
from ..execution.order_manager import OrderManager
from ..risk.kill_switch import KillSwitch
from ..risk.position_limits import PositionLimits
from ..pnl.pnl_tracker import PnLTracker
from ..data.market_snapshot import MarketSnapshot
from .. import config


class Executor:

    def __init__(
        self,
        order_manager: OrderManager,
        signal_bus:    SignalBus,
        kill_switch:   KillSwitch,
        pnl_tracker:   PnLTracker,
        tickers:       list = None,
    ):
        self._orders      = order_manager
        self._bus         = signal_bus
        self._kill        = kill_switch
        self._pnl         = pnl_tracker
        self._limits      = PositionLimits()
        self.tickers      = tickers or config.TICKERS
        self._last_traded: dict = {}    # {ticker: tick}
        self._positions:   dict = {t: 0 for t in self.tickers}

    def run_tick(self, snapshot: MarketSnapshot) -> None:
        """Called once per tick from the main loop."""
        tick = snapshot.tick

        # Auto-flatten near end of case
        if tick >= config.STOP_TRADING_TICK:
            print(f"\n[Tick {tick}] Stop-trading tick reached — flattening...")
            self._orders.flatten_all(self.tickers)
            return

        if self._kill.is_halted():
            return

        for ticker in self.tickers:
            signal = self._bus.combined_signal(ticker)
            self._act(ticker, signal, snapshot, tick)

    def _act(self, ticker: str, signal: float, snapshot: MarketSnapshot, tick: int) -> None:
        signal = -signal  # Contrarian: always do the opposite of what the model suggests
        # Cooldown check
        if tick - self._last_traded.get(ticker, -999) < config.TRADE_COOLDOWN:
            return

        mid = snapshot.mid.get(ticker)
        if mid is None:
            return

        pos = self._positions.get(ticker, 0)

        if signal > config.SIGNAL_THRESHOLD:
            qty = self._limits.can_buy(ticker, config.TRADE_SIZE, pos)
            if qty > 0 and self._orders.buy(ticker, qty, mid):
                self._positions[ticker] = pos + qty
                self._last_traded[ticker] = tick

        elif signal < -config.SIGNAL_THRESHOLD:
            qty = self._limits.can_sell(ticker, config.TRADE_SIZE, pos)
            if qty > 0 and self._orders.sell(ticker, qty, mid):
                self._positions[ticker] = pos - qty
                self._last_traded[ticker] = tick

        # Close position if signal flips and we have an open position
        elif pos > 0 and signal < 0:
            qty = min(abs(pos), config.TRADE_SIZE)
            qty = self._limits.can_sell(ticker, qty, pos)
            if qty > 0 and self._orders.sell(ticker, qty, mid):
                self._positions[ticker] = pos - qty
                self._last_traded[ticker] = tick

        elif pos < 0 and signal > 0:
            qty = min(abs(pos), config.TRADE_SIZE)
            qty = self._limits.can_buy(ticker, qty, pos)
            if qty > 0 and self._orders.buy(ticker, qty, mid):
                self._positions[ticker] = pos + qty
                self._last_traded[ticker] = tick

    def sync_positions(self, positions: dict) -> None:
        """
        Sync internal position tracker from the API.
        Call once per tick to stay accurate if manual trades occur.
        Only updates tickers we track — doesn't import manual-trade PnL.
        """
        for t in self.tickers:
            if t in positions:
                self._positions[t] = positions[t]
