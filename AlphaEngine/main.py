"""
AlphaEngine main entry point.

Starts two concurrent components:
  1. Trading loop (daemon thread) — polls RIT API, runs alphas, places orders
  2. Dash dashboard server       — serves http://127.0.0.1:8050 (blocking)

Usage:
  python AlphaEngine/main.py

Stop:  Ctrl+C
"""
import time
import threading



from .data.rit_client import RITClient
from .alphas import alpha_registry
from .alphas.kalman_pairs_alpha import KalmanPairsAlpha
from .execution.signal_bus import SignalBus
from .execution.order_manager import OrderManager
from .execution.executor import Executor
from .risk.kill_switch import KillSwitch
from .pnl.pnl_tracker import PnLTracker
from .dashboard.shared_state import DashboardState
from .dashboard.app import create_app
from . import config


def trading_loop(state: DashboardState, stop_event: threading.Event) -> None:
    client = RITClient()
    bus    = SignalBus()
    kill   = KillSwitch()
    pnl    = PnLTracker(config.TICKERS)
    orders = OrderManager(client, pnl)
    alphas = alpha_registry.get_all()
    exec_  = Executor(orders, bus, kill, pnl)

    # Wire up shared state
    alpha_names = [a.name for a in alphas]
    state.init_tickers(config.TICKERS, alpha_names)

    print("Waiting for case to become ACTIVE...")

    while not stop_event.is_set():
        snap = client.next_snapshot()
        if snap is None:
            print("Case ended or not active.")
            time.sleep(1.0)
            continue

        # Get z-score and beta from Kalman alpha for the snapshot
        kalman_alpha = next((a for a in alphas if isinstance(a, KalmanPairsAlpha)), None)
        if kalman_alpha:
            snap.zscore = kalman_alpha.last_zscore
            snap.beta   = kalman_alpha.last_beta

        # Regime from kill switch HMM
        snap.regime = kill.current_regime

        # Evaluate kill switch
        kill.check(snap, pnl)

        # Run all alphas → populate signal bus
        all_signals = {}
        for alpha in alphas:
            signals = alpha.update(snap)
            bus.set(alpha.name, signals)
            all_signals[alpha.name] = signals

        # Execute orders
        exec_.run_tick(snap)

        # Update PnL with latest prices
        pnl.update_prices(snap.mid, tick=snap.tick)

        # Push to dashboard
        state.push_tick(snap.tick, snap, all_signals, pnl, kill)

        print(
            f"[Tick {snap.tick:>3}] regime={snap.regime:<16} "
            f"z={snap.zscore:+.2f}  "
            f"PnL=${pnl.total_pnl():+.2f}"
            + ("  [HALTED]" if kill.is_halted() else "")
        )


def main() -> None:
    state      = DashboardState()
    stop_event = threading.Event()

    thread = threading.Thread(
        target=trading_loop,
        args=(state, stop_event),
        daemon=True,
        name="TradingLoop",
    )
    thread.start()

    app = create_app(state)
    print(f"\nDashboard: http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")
    print("Press Ctrl+C to stop.\n")

    try:
        app.run(
            host=config.DASHBOARD_HOST,
            port=config.DASHBOARD_PORT,
            debug=False,
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_event.set()


if __name__ == "__main__":
    main()
