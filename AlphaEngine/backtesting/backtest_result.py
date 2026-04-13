"""BacktestResult: output of a single backtest run."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class BacktestResult:
    alpha_name:    str
    params:        dict
    equity_curve:  list   # [(tick, cumulative_pnl)]
    trades:        list   # [(tick, ticker, action, qty, price)]
    sharpe:        float  = 0.0
    max_drawdown:  float  = 0.0
    total_pnl:     float  = 0.0
    n_trades:      int    = 0

    def compute_metrics(self, ticks_per_period: int = 300) -> None:
        """Compute Sharpe ratio and max drawdown from equity_curve."""
        if len(self.equity_curve) < 2:
            return
        pnls = np.array([p for _, p in self.equity_curve])
        rets = np.diff(pnls)
        if rets.std() == 0:
            self.sharpe = 0.0
        else:
            self.sharpe = float(rets.mean() / rets.std() * np.sqrt(ticks_per_period))
        # Max drawdown
        peak = pnls[0]
        max_dd = 0.0
        for p in pnls:
            peak   = max(peak, p)
            max_dd = max(max_dd, peak - p)
        self.max_drawdown = max_dd
        self.total_pnl    = float(pnls[-1])
        self.n_trades     = len(self.trades)

    def __repr__(self) -> str:
        return (
            f"BacktestResult({self.alpha_name} params={self.params} "
            f"sharpe={self.sharpe:.3f} pnl={self.total_pnl:.2f} "
            f"drawdown={self.max_drawdown:.2f} trades={self.n_trades})"
        )
