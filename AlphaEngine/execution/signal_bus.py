"""
SignalBus: thread-safe container for alpha signals.

Alphas write their signals here; Executor reads from here.
This decouples signal generation from order execution.
"""
import threading


class SignalBus:

    def __init__(self):
        self._lock    = threading.Lock()
        self._signals: dict = {}   # {alpha_name: {ticker: float}}

    def set(self, alpha_name: str, ticker_signals: dict) -> None:
        """Write signals from one alpha. ticker_signals = {ticker: float in [-1,1]}."""
        with self._lock:
            self._signals[alpha_name] = ticker_signals.copy()

    def get_all(self) -> dict:
        """Return a snapshot of all current signals."""
        with self._lock:
            return {name: signals.copy() for name, signals in self._signals.items()}

    def combined_signal(self, ticker: str, weights: dict = None) -> float:
        """
        Weighted average of all alpha signals for a single ticker.
        weights = {alpha_name: float}; defaults to equal weight.
        """
        signals = self.get_all()
        if not signals:
            return 0.0
        if weights is None:
            weights = {name: 1.0 for name in signals}
        total_w = total_s = 0.0
        for name, ticker_map in signals.items():
            w = weights.get(name, 1.0)
            total_s += w * ticker_map.get(ticker, 0.0)
            total_w += abs(w)
        return total_s / total_w if total_w > 0 else 0.0
