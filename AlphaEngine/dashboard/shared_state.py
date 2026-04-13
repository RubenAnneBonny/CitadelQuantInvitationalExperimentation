"""
DashboardState: thread-safe container shared between the trading loop and Dash callbacks.

The trading thread writes via update().
Dash callbacks read via snapshot() — they get a consistent copy with no lock held.
"""
import threading
from collections import deque
from dataclasses import dataclass, field


MAX_HISTORY = 300   # number of ticks to keep in rolling charts


@dataclass
class DashboardState:
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Tick info
    tick:           int  = 0
    ticks_per_period: int = 300

    # Regime & kill switch
    regime:         str  = "mean_reverting"
    kill_switch_on: bool = False
    halt_reason:    str  = ""

    # Kalman
    zscore:         float = 0.0
    beta:           float = 1.0

    # Rolling histories for charts  {ticker: deque(maxlen=MAX_HISTORY)}
    price_history:  dict = field(default_factory=dict)
    bid_history:    dict = field(default_factory=dict)
    ask_history:    dict = field(default_factory=dict)
    signal_history: dict = field(default_factory=dict)   # {alpha_name: {ticker: deque}}
    tick_history:   list = field(default_factory=list)   # list of tick numbers

    # Z-score rolling history
    zscore_history: object = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))

    # PnL
    realized_pnl:   float = 0.0
    unrealized_pnl: float = 0.0
    pnl_history:    list  = field(default_factory=list)  # [(tick, realized, total)]

    # Positions table
    positions: dict = field(default_factory=dict)        # per_ticker_pnl() output

    def init_tickers(self, tickers: list, alpha_names: list) -> None:
        with self._lock:
            for t in tickers:
                if t not in self.price_history:
                    self.price_history[t] = deque(maxlen=MAX_HISTORY)
                    self.bid_history[t]   = deque(maxlen=MAX_HISTORY)
                    self.ask_history[t]   = deque(maxlen=MAX_HISTORY)
            for name in alpha_names:
                if name not in self.signal_history:
                    self.signal_history[name] = {t: deque(maxlen=MAX_HISTORY) for t in tickers}

    def push_tick(self, tick: int, snapshot, signals: dict, pnl_tracker,
                  kill_switch) -> None:
        """Called by the trading loop once per tick."""
        with self._lock:
            self.tick   = tick
            self.regime = snapshot.regime
            self.zscore = snapshot.zscore
            self.beta   = snapshot.beta

            self.kill_switch_on = kill_switch.is_halted()
            self.halt_reason    = kill_switch.halt_reason

            self.tick_history.append(tick)
            if len(self.tick_history) > MAX_HISTORY:
                self.tick_history = self.tick_history[-MAX_HISTORY:]

            for ticker in snapshot.last:
                if ticker in self.price_history:
                    self.price_history[ticker].append(snapshot.last[ticker])
                    self.bid_history[ticker].append(snapshot.bid.get(ticker, 0))
                    self.ask_history[ticker].append(snapshot.ask.get(ticker, 0))

            for alpha_name, ticker_sigs in signals.items():
                if alpha_name in self.signal_history:
                    for ticker, sig in ticker_sigs.items():
                        if ticker in self.signal_history[alpha_name]:
                            self.signal_history[alpha_name][ticker].append(sig)

            self.zscore_history.append(snapshot.zscore)

            self.realized_pnl   = pnl_tracker.realized_pnl()
            self.unrealized_pnl = pnl_tracker.unrealized_pnl()
            self.pnl_history.append((tick, self.realized_pnl, self.realized_pnl + self.unrealized_pnl))
            if len(self.pnl_history) > MAX_HISTORY:
                self.pnl_history = self.pnl_history[-MAX_HISTORY:]

            self.positions = pnl_tracker.per_ticker_pnl()

    def snapshot(self) -> dict:
        """Return a shallow-copied dict safe for the Dash callback thread."""
        with self._lock:
            return {
                "tick":             self.tick,
                "ticks_per_period": self.ticks_per_period,
                "regime":           self.regime,
                "kill_switch_on":   self.kill_switch_on,
                "halt_reason":      self.halt_reason,
                "zscore":           self.zscore,
                "beta":             self.beta,
                "tick_history":     list(self.tick_history),
                "price_history":    {t: list(v) for t, v in self.price_history.items()},
                "bid_history":      {t: list(v) for t, v in self.bid_history.items()},
                "ask_history":      {t: list(v) for t, v in self.ask_history.items()},
                "signal_history":   {
                    name: {t: list(v) for t, v in sigs.items()}
                    for name, sigs in self.signal_history.items()
                },
                "zscore_history":   list(self.zscore_history),
                "realized_pnl":     self.realized_pnl,
                "unrealized_pnl":   self.unrealized_pnl,
                "pnl_history":      list(self.pnl_history),
                "positions":        dict(self.positions),
            }
