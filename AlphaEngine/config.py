"""
Central configuration for AlphaEngine.
All constants live here — change feed, model, or thresholds in one place.
"""
import os

# ── API ───────────────────────────────────────────────────────────────────────
API_KEY       = "IRJW2AFJ"
BASE          = "http://localhost:9999/v1"

# ── Tickers ───────────────────────────────────────────────────────────────────
TICKERS       = ["CRZY", "TAME"]
PAIR          = ("CRZY", "TAME")     # y = CRZY, x = TAME

# ── Position limits ───────────────────────────────────────────────────────────
MAX_POSITION  = 10_000

# ── Trading ───────────────────────────────────────────────────────────────────
TRADE_SIZE        = 500
STOP_TRADING_TICK = 290
MIN_HISTORY       = 3
POLL_INTERVAL     = 0.25             # seconds between API polls
TRADE_COOLDOWN    = 5                # minimum ticks between orders on same ticker
SIGNAL_THRESHOLD  = 0.3             # |signal| must exceed this to trigger a trade

# ── Kalman pairs ──────────────────────────────────────────────────────────────
KF_DELTA          = 1e-3
PAIRS_ENTRY_Z     = 2.5
PAIRS_EXIT_Z      = 0.75
PAIRS_MIN_OBS     = 50
PAIRS_BASE_SIZE   = 500
PAIRS_MAX_SCALE   = 4

# ── Bollinger bands ───────────────────────────────────────────────────────────
BB_WINDOW     = 20
BB_WIDTH      = 1.5
BB_BASE_SIZE  = 500
BB_MAX_SCALE  = 3

# ── Momentum alpha ────────────────────────────────────────────────────────────
MOM_SHORT     = 5
MOM_LONG      = 20

# ── Backtesting ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
DEFAULT_CSV   = os.path.join(_HERE, "..", "Training_before_comp", "rit_full_session.csv")
SLIPPAGE_BPS  = 5.0                  # basis points of slippage per fill
BT_TRADE_SIZE = 500                  # shares per simulated fill

# ── Kill switch ───────────────────────────────────────────────────────────────
MAX_DRAWDOWN_USD  = 5000.0
VOL_SPIKE_FACTOR  = 3.0

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_HOST       = "127.0.0.1"
DASHBOARD_PORT       = 8050
DASHBOARD_UPDATE_MS  = 250
