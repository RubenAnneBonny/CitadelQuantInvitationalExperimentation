"""
SharpeOptimizer: grid-search or scipy-based hyperparameter tuning.

Finds the parameter set for each alpha that maximizes Sharpe ratio
over the backtesting dataset.

Usage:
  feed      = CSVFeed("rit_full_session.csv")
  engine    = BacktestEngine(feed)
  optimizer = SharpeOptimizer(engine)

  # Grid search
  best = optimizer.optimize(
      BollingerAlpha(),
      param_grid={"window": [10, 15, 20, 30], "width": [1.0, 1.5, 2.0, 2.5]},
  )
  print(best["best_sharpe"], best["best_params"])

  # Continuous optimization (faster for large param spaces)
  best = optimizer.optimize(
      BollingerAlpha(),
      param_grid={"window": (5, 40), "width": (0.5, 3.0)},
      method="scipy",
  )
"""
import itertools
import numpy as np
from ..alphas.base_alpha import BaseAlpha
from .backtest_engine import BacktestEngine


class SharpeOptimizer:

    def __init__(self, engine: BacktestEngine):
        self._engine = engine

    def optimize(
        self,
        alpha:      BaseAlpha,
        param_grid: dict,
        method:     str = "grid",   # "grid" or "scipy"
        verbose:    bool = True,
    ) -> dict:
        """
        Optimize alpha hyperparameters to maximize Sharpe ratio.

        param_grid values:
          list  → grid search over those specific values
          tuple → (min, max) continuous range for scipy search

        Returns:
          {
            "best_params":  {...},
            "best_sharpe":  float,
            "best_result":  BacktestResult,
            "all_results":  [BacktestResult, ...],   # grid only
          }
        """
        if method == "scipy":
            return self._scipy_optimize(alpha, param_grid, verbose)
        return self._grid_search(alpha, param_grid, verbose)

    def _grid_search(self, alpha: BaseAlpha, param_grid: dict, verbose: bool) -> dict:
        keys   = list(param_grid.keys())
        values = [param_grid[k] if isinstance(param_grid[k], list)
                  else list(param_grid[k])     # treat tuple as list if not range
                  for k in keys]
        combos = list(itertools.product(*values))

        best_sharpe = -np.inf
        best_result = None
        all_results = []

        for combo in combos:
            params = dict(zip(keys, combo))
            alpha.set_params(params)
            result = self._engine.run(alpha)
            all_results.append(result)
            if verbose:
                print(f"  params={params}  sharpe={result.sharpe:.3f}  pnl={result.total_pnl:.2f}")
            if result.sharpe > best_sharpe:
                best_sharpe = result.sharpe
                best_result = result

        print(f"\n[SharpeOptimizer] Best: sharpe={best_sharpe:.3f}  params={best_result.params}")
        return {
            "best_params":  best_result.params if best_result else {},
            "best_sharpe":  best_sharpe,
            "best_result":  best_result,
            "all_results":  all_results,
        }

    def _scipy_optimize(self, alpha: BaseAlpha, param_grid: dict, verbose: bool) -> dict:
        from scipy.optimize import differential_evolution

        keys   = list(param_grid.keys())
        bounds = []
        for k in keys:
            v = param_grid[k]
            if isinstance(v, (list, tuple)) and len(v) == 2:
                bounds.append((float(v[0]), float(v[1])))
            elif isinstance(v, list):
                bounds.append((float(min(v)), float(max(v))))
            else:
                raise ValueError(f"scipy mode: param '{k}' must be (min, max) tuple")

        def objective(x):
            params = {k: float(v) for k, v in zip(keys, x)}
            alpha.set_params(params)
            result = self._engine.run(alpha)
            return -result.sharpe   # minimise negative Sharpe

        opt = differential_evolution(objective, bounds, seed=42, maxiter=50, tol=0.01,
                                     workers=1, updating="immediate")
        best_params = {k: float(v) for k, v in zip(keys, opt.x)}
        alpha.set_params(best_params)
        best_result = self._engine.run(alpha)

        print(f"\n[SharpeOptimizer] scipy best: sharpe={best_result.sharpe:.3f}  params={best_params}")
        return {
            "best_params":  best_params,
            "best_sharpe":  best_result.sharpe,
            "best_result":  best_result,
            "all_results":  [],
        }


def run_all_alphas(csv_path: str = None, verbose: bool = True) -> dict:
    """
    Convenience: run Sharpe optimization for every registered alpha.
    Returns {alpha_name: optimizer_result}.

    Example:
      python -m AlphaEngine.backtesting.sharpe_optimizer
    """
    from ..data.csv_feed import CSVFeed
    from ..alphas import alpha_registry
    from .. import config

    path   = csv_path or config.DEFAULT_CSV
    feed   = CSVFeed(path)
    engine = BacktestEngine(feed)
    opt    = SharpeOptimizer(engine)

    results = {}
    for alpha in alpha_registry.get_all():
        print(f"\n=== Optimizing {alpha.name} ===")
        params = alpha.get_params()
        if not params:
            result = engine.run(alpha)
            results[alpha.name] = {
                "best_params": {}, "best_sharpe": result.sharpe, "best_result": result
            }
            print(f"  (no params)  sharpe={result.sharpe:.3f}  pnl={result.total_pnl:.2f}")
            continue

        # Build a default grid ±50% around current params
        grid = {}
        for k, v in params.items():
            if isinstance(v, int):
                lo = max(2, int(v * 0.5))
                hi = int(v * 2.0)
                grid[k] = list(range(lo, hi + 1, max(1, (hi - lo) // 5)))
            else:
                lo = max(0.1, v * 0.5)
                hi = v * 2.0
                grid[k] = [round(lo + i * (hi - lo) / 4, 2) for i in range(5)]

        results[alpha.name] = opt.optimize(alpha, grid, verbose=verbose)

    return results


if __name__ == "__main__":
    run_all_alphas()
