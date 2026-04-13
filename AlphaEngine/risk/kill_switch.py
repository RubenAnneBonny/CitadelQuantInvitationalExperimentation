"""
KillSwitch: three-layer halt logic.

Layer 1 — Drawdown:      stop if realized drawdown > MAX_DRAWDOWN_USD
Layer 2 — Volatility:    stop if rolling vol > VOL_SPIKE_FACTOR × baseline
Layer 3 — HMM regime:    stop if OnlineHMMRegime returns "crisis"

Usage:
  ks = KillSwitch()
  ks.check(snapshot, pnl_tracker)
  if ks.is_halted():
      # skip all trading

  ks.reset()   # manually re-enable after reviewing
"""
import os, sys
import numpy as np
from collections import deque
from ..data.market_snapshot import MarketSnapshot
from .. import config

# Import OnlineHMMRegime from Trading_Strategy
_TRADING_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Trading_Strategy")
if _TRADING_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_TRADING_DIR))

from hmm_regime import OnlineHMMRegime   # noqa: E402


class KillSwitch:

    def __init__(
        self,
        max_drawdown_usd: float = config.MAX_DRAWDOWN_USD,
        vol_spike_factor: float = config.VOL_SPIKE_FACTOR,
        vol_window: int         = 20,
        baseline_window: int    = 60,
    ):
        self.max_drawdown_usd = max_drawdown_usd
        self.vol_spike_factor = vol_spike_factor
        self.vol_window       = vol_window
        self.baseline_window  = baseline_window

        self._hmm             = OnlineHMMRegime()
        self._price_history: deque = deque(maxlen=baseline_window)
        self._peak_pnl        = 0.0
        self._halted          = False
        self._halt_reason     = ""

    def check(self, snapshot: MarketSnapshot, pnl_tracker) -> bool:
        """
        Evaluate all halt conditions. Returns True if trading should stop.
        Sets self._halted and self._halt_reason.
        """
        if self._halted:
            return True

        # Update HMM with latest price
        ref_ticker = next(iter(snapshot.last))
        price = snapshot.last[ref_ticker]
        self._price_history.append(price)
        regime = self._hmm.update(price=price)

        # Layer 3: crisis regime
        if regime == "crisis":
            self._halt("HMM regime = crisis")
            return True

        # Layer 1: drawdown
        total_pnl = pnl_tracker.total_pnl()
        self._peak_pnl = max(self._peak_pnl, total_pnl)
        drawdown = self._peak_pnl - total_pnl
        if drawdown > self.max_drawdown_usd:
            self._halt(f"Drawdown ${drawdown:.0f} > limit ${self.max_drawdown_usd:.0f}")
            return True

        # Layer 2: volatility spike
        prices = list(self._price_history)
        if len(prices) >= self.baseline_window:
            recent_vol   = float(np.std(np.diff(prices[-self.vol_window:])))
            baseline_vol = float(np.std(np.diff(prices)))
            if baseline_vol > 0 and recent_vol > self.vol_spike_factor * baseline_vol:
                self._halt(f"Vol spike {recent_vol:.4f} > {self.vol_spike_factor}× baseline {baseline_vol:.4f}")
                return True

        return False

    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @property
    def current_regime(self) -> str:
        return getattr(self._hmm, "current_regime", "unknown")

    def reset(self) -> None:
        """Manually re-enable trading after kill switch triggered."""
        self._halted      = False
        self._halt_reason = ""
        self._peak_pnl    = 0.0
        print("[KillSwitch] Reset — trading re-enabled.")

    def _halt(self, reason: str) -> None:
        self._halted      = True
        self._halt_reason = reason
        print(f"[KillSwitch] HALTED: {reason}")
