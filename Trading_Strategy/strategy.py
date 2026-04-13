"""
Regime-Driven Trading Strategy
--------------------------------
Architecture
  1. HMM regime detector (primary driver)
       mean_reverting → Bollinger bands + Kalman pairs both active
       trending       → Kalman pairs only (direction-neutral spread)
       crisis         → flatten and halt

  2. Kalman filter (rolling beta pairs trading, CRZY vs TAME)
       Open pair when |z-score| > PAIRS_ENTRY_Z
       Close pair when |z-score| < PAIRS_EXIT_Z

  3. Bollinger bands (per-ticker mean reversion)
       Only active in mean_reverting regime

Transaction cost controls
  - State machine per ticker/pair: only trade on genuine signal transitions
  - TRADE_COOLDOWN: minimum ticks between successive orders on same ticker
  - All orders are LIMIT at mid price — never market orders
  - cancel_all() only when a signal reverses, not every tick

Compatible with Rotman Interactive Trader (RIT) REST API v1.
"""

import time
import requests
import numpy as np
from kalman_pairs import KalmanPairFilter
from hmm_regime import OnlineHMMRegime

# ── API config ────────────────────────────────────────────────────────────────
API_KEY = "IRJW2AFJ"
BASE    = "http://localhost:9999/v1"
HEADERS = {"X-API-Key": API_KEY}

# ── General ───────────────────────────────────────────────────────────────────
TICKERS           = ["CRZY", "TAME"]
PAIR              = ("CRZY", "TAME")   # y = CRZY, x = TAME
MAX_POSITION      = 10_000
TRADE_SIZE        = 500
STOP_TRADING_TICK = 290
MIN_HISTORY       = 3
POLL_INTERVAL     = 0.25
TRADE_COOLDOWN    = 5    # minimum ticks between orders on the same ticker

# ── Kalman pairs ──────────────────────────────────────────────────────────────
KF_DELTA          = 1e-3
PAIRS_ENTRY_Z     = 2.5    # only open pair when |z| exceeds this
PAIRS_EXIT_Z      = 0.75   # close pair when |z| falls below this
PAIRS_MIN_OBS     = 50     # minimum KF updates before trusting beta
PAIRS_BASE_SIZE   = 500    # base shares per leg at entry Z
PAIRS_MAX_SCALE   = 4      # maximum size multiplier (so up to 2000 shares)

# ── Bollinger bands ───────────────────────────────────────────────────────────
BB_WINDOW      = 20
BB_WIDTH       = 1.5
BB_BASE_SIZE   = 500
BB_MAX_SCALE   = 3         # max multiplier based on band penetration depth


# ── API helpers ───────────────────────────────────────────────────────────────

def get_case():
    return requests.get(f"{BASE}/case", headers=HEADERS).json()

def get_securities():
    return requests.get(f"{BASE}/securities", headers=HEADERS).json()

def get_book(ticker):
    return requests.get(
        f"{BASE}/securities/book", params={"ticker": ticker}, headers=HEADERS
    ).json()

def place_limit(ticker, action, qty, price):
    """Place a limit order at the given price."""
    return requests.post(f"{BASE}/orders", headers=HEADERS, params={
        "ticker":   ticker,
        "type":     "LIMIT",
        "quantity": qty,
        "action":   action,
        "price":    round(price, 2),
    }).json()

def cancel_all():
    try:
        requests.post(f"{BASE}/commands/cancel", params={"all": 1}, headers=HEADERS)
    except Exception as e:
        print(f"[WARN] cancel_all: {e}")

def get_positions():
    return {s["ticker"]: s["position"] for s in get_securities()
            if s["ticker"] in TICKERS}

def get_mid(ticker):
    """Return mid price, or None if book is empty."""
    book = get_book(ticker)
    bid  = book["bids"][0]["price"] if book["bids"] else None
    ask  = book["asks"][0]["price"] if book["asks"] else None
    if bid is None or ask is None:
        return None, None, None
    return (bid + ask) / 2, bid, ask

def flatten_ticker(ticker):
    pos = get_positions().get(ticker, 0)
    if pos == 0:
        return
    qty    = min(abs(int(pos)), 25_000)
    action = "SELL" if pos > 0 else "BUY"
    mid, bid, ask = get_mid(ticker)
    price  = bid if action == "SELL" else ask
    if price is None:
        return
    print(f"  [FLATTEN] {action} {qty} {ticker} @ {price:.2f}")
    place_limit(ticker, action, qty, price)


# ── Bollinger signal (returns desired state, not action) ──────────────────────

def bb_desired_state(price_history: list) -> str:
    """Returns 'long', 'short', or 'flat' based on Bollinger bands."""
    arr  = np.array(price_history[-BB_WINDOW:])
    mean = arr.mean()
    std  = arr.std()
    if std == 0:
        return "flat"
    p = price_history[-1]
    if p < mean - BB_WIDTH * std:
        return "long"
    elif p > mean + BB_WIDTH * std:
        return "short"
    return "flat"


# ── State machine helpers ─────────────────────────────────────────────────────

def transition(ticker, from_state, to_state, positions, tick, last_traded, price_histories):
    """
    Moves ticker from from_state → to_state using limit orders at mid.
    Only fires if TRADE_COOLDOWN has passed. Returns True if an order was sent.
    """
    if tick - last_traded.get(ticker, -999) < TRADE_COOLDOWN:
        return False

    pos = positions.get(ticker, 0)
    mid, bid, ask = get_mid(ticker)
    if mid is None:
        return False

    orders_sent = False

    # Scale size by how far price is outside the band
    arr   = np.array(price_histories[ticker][-BB_WINDOW:])
    mean, std = arr.mean(), arr.std()
    if std > 0:
        penetration = abs(price_histories[ticker][-1] - mean) / (BB_WIDTH * std)
        scale = min(max(penetration, 1.0), BB_MAX_SCALE)
    else:
        scale = 1.0
    sized = max(1, round(BB_BASE_SIZE * scale))

    # Close existing position first if changing sides
    if from_state == "long" and to_state != "long" and pos > 0:
        qty = min(abs(int(pos)), sized)
        print(f"  [CLOSE LONG]  {ticker} SELL {qty} @ {mid:.2f}")
        place_limit(ticker, "SELL", qty, mid)
        orders_sent = True

    elif from_state == "short" and to_state != "short" and pos < 0:
        qty = min(abs(int(pos)), sized)
        print(f"  [CLOSE SHORT] {ticker} BUY  {qty} @ {mid:.2f}")
        place_limit(ticker, "BUY", qty, mid)
        orders_sent = True

    # Open new position
    if to_state == "long" and pos < MAX_POSITION:
        qty = min(sized, MAX_POSITION - int(pos))
        print(f"  [OPEN  LONG]  {ticker} BUY  {qty} @ {mid:.2f}  (scale={scale:.2f})")
        place_limit(ticker, "BUY", qty, mid)
        orders_sent = True

    elif to_state == "short" and pos > -MAX_POSITION:
        qty = min(sized, MAX_POSITION + int(pos))
        print(f"  [OPEN  SHORT] {ticker} SELL {qty} @ {mid:.2f}  (scale={scale:.2f})")
        place_limit(ticker, "SELL", qty, mid)
        orders_sent = True

    if orders_sent:
        last_traded[ticker] = tick
    return orders_sent


# ── Pairs execution ───────────────────────────────────────────────────────────

def pairs_desired_state(zscore: float, current_pair_state: str) -> str:
    """
    Returns desired pair state: 'long_spread', 'short_spread', or 'flat'.
    Uses hysteresis to avoid churn near the thresholds.
    """
    if current_pair_state == "flat":
        if zscore > PAIRS_ENTRY_Z:
            return "short_spread"   # spread too wide: sell y, buy x
        elif zscore < -PAIRS_ENTRY_Z:
            return "long_spread"    # spread too narrow: buy y, sell x
        return "flat"
    elif current_pair_state == "short_spread":
        return "flat" if zscore < PAIRS_EXIT_Z else "short_spread"
    elif current_pair_state == "long_spread":
        return "flat" if zscore > -PAIRS_EXIT_Z else "long_spread"
    return "flat"


def execute_pair_transition(from_state, to_state, zscore, beta, positions, tick, last_traded):
    if from_state == to_state:
        return

    ticker_y, ticker_x = PAIR
    mid_y, bid_y, ask_y = get_mid(ticker_y)
    mid_x, bid_x, ask_x = get_mid(ticker_x)
    if mid_y is None or mid_x is None:
        return

    pos_y = positions.get(ticker_y, 0)
    pos_x = positions.get(ticker_x, 0)

    # Scale trade size by z-score conviction — bigger bet when more extreme
    scale = min(abs(zscore) / PAIRS_ENTRY_Z, PAIRS_MAX_SCALE)
    qty_y = max(1, round(PAIRS_BASE_SIZE * scale))
    qty_x = max(1, round(abs(beta) * PAIRS_BASE_SIZE * scale))

    # Cooldown check: use the y-leg ticker as the gating key
    if tick - last_traded.get("PAIR", -999) < TRADE_COOLDOWN:
        return

    cancel_all()

    if to_state == "flat":
        # Close both legs
        if pos_y > 0:
            place_limit(ticker_y, "SELL", min(abs(int(pos_y)), qty_y), mid_y)
        elif pos_y < 0:
            place_limit(ticker_y, "BUY",  min(abs(int(pos_y)), qty_y), mid_y)
        if pos_x > 0:
            place_limit(ticker_x, "SELL", min(abs(int(pos_x)), qty_x), mid_x)
        elif pos_x < 0:
            place_limit(ticker_x, "BUY",  min(abs(int(pos_x)), qty_x), mid_x)
        print(f"  [PAIRS CLOSE] flat  y={mid_y:.2f} x={mid_x:.2f}")

    elif to_state == "short_spread":
        # Spread too wide: SELL y, BUY x
        if pos_y > -MAX_POSITION:
            place_limit(ticker_y, "SELL", min(qty_y, MAX_POSITION + int(pos_y)), mid_y)
        if pos_x < MAX_POSITION:
            place_limit(ticker_x, "BUY",  min(qty_x, MAX_POSITION - int(pos_x)), mid_x)
        print(f"  [PAIRS OPEN]  short_spread  SELL {ticker_y} @ {mid_y:.2f}  BUY {ticker_x} @ {mid_x:.2f}  β={beta:.3f}")

    elif to_state == "long_spread":
        # Spread too narrow: BUY y, SELL x
        if pos_y < MAX_POSITION:
            place_limit(ticker_y, "BUY",  min(qty_y, MAX_POSITION - int(pos_y)), mid_y)
        if pos_x > -MAX_POSITION:
            place_limit(ticker_x, "SELL", min(qty_x, MAX_POSITION + int(pos_x)), mid_x)
        print(f"  [PAIRS OPEN]  long_spread   BUY  {ticker_y} @ {mid_y:.2f}  SELL {ticker_x} @ {mid_x:.2f}  β={beta:.3f}")

    last_traded["PAIR"] = tick


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    price_histories = {t: [] for t in TICKERS}
    kf              = KalmanPairFilter(delta=KF_DELTA)
    hmm             = OnlineHMMRegime()

    # Position state machines — only trade on transitions
    bb_state        = {t: "flat" for t in TICKERS}   # flat / long / short
    pair_state      = "flat"                           # flat / long_spread / short_spread
    last_traded     = {}                               # ticker/key → last tick traded
    last_tick       = -1

    print("Waiting for case to become ACTIVE...")

    while True:
        case = get_case()

        if case["status"] != "ACTIVE":
            if last_tick >= 0:
                print("Case ended. Exiting.")
                break
            time.sleep(0.5)
            continue

        tick = case["tick"]
        if tick == last_tick:
            time.sleep(POLL_INTERVAL)
            continue
        last_tick = tick

        # ── End-of-case flatten ───────────────────────────────────────────────
        if tick >= STOP_TRADING_TICK:
            print(f"\n[Tick {tick}] Flattening all positions...")
            cancel_all()
            for ticker in TICKERS:
                flatten_ticker(ticker)
            print("Done.")
            break

        # ── Snapshot ──────────────────────────────────────────────────────────
        securities = get_securities()
        prices = {s["ticker"]: s["last"] for s in securities if s["ticker"] in TICKERS}
        for ticker in TICKERS:
            if ticker in prices:
                price_histories[ticker].append(prices[ticker])

        min_history = min(len(price_histories[t]) for t in TICKERS)
        if min_history < MIN_HISTORY:
            print(f"[Tick {tick}] Collecting ({min_history}/{MIN_HISTORY})")
            time.sleep(POLL_INTERVAL)
            continue

        # ── Kalman update ─────────────────────────────────────────────────────
        ticker_y, ticker_x = PAIR
        kf_state = kf.update(
            y=price_histories[ticker_y][-1],
            x=price_histories[ticker_x][-1],
        )
        zscore = kf_state["zscore"]
        beta   = kf_state["beta"]

        # ── HMM regime update ─────────────────────────────────────────────────
        regime = hmm.update(price=price_histories[ticker_y][-1])

        print(f"\n[Tick {tick}/{case['ticks_per_period']}]  "
              f"regime={regime}  z={zscore:+.2f}  β={beta:.3f}")

        positions = get_positions()

        # ── Pairs signal ──────────────────────────────────────────────────────
        pairs_active = kf.n_obs >= PAIRS_MIN_OBS
        desired_pair = pairs_desired_state(zscore, pair_state) if pairs_active else "flat"
        if desired_pair != pair_state:
            execute_pair_transition(pair_state, desired_pair, zscore, beta, positions, tick, last_traded)
            pair_state = desired_pair
        if not pairs_active:
            print(f"  [PAIRS] warming up ({kf.n_obs}/{PAIRS_MIN_OBS} obs)")

        # ── Bollinger signals (only in mean_reverting regime) ─────────────────
        if regime == "mean_reverting":
            for ticker in TICKERS:
                desired = bb_desired_state(price_histories[ticker])
                current = bb_state[ticker]
                if desired != current:
                    # Signal changed — cancel stale orders and re-evaluate
                    cancel_all()
                    did_trade = transition(
                        ticker, current, desired, positions, tick, last_traded, price_histories
                    )
                    if did_trade:
                        bb_state[ticker] = desired
        else:
            # Trending regime: close any open BB positions to avoid fighting the trend
            for ticker in TICKERS:
                if bb_state[ticker] != "flat":
                    cancel_all()
                    transition(ticker, bb_state[ticker], "flat",
                               positions, tick, last_traded, price_histories)
                    bb_state[ticker] = "flat"

        time.sleep(POLL_INTERVAL)

    print("\n=== TRADING COMPLETE ===")
    final = get_positions()
    for ticker in TICKERS:
        print(f"Final position {ticker}: {final.get(ticker, 0)}")


if __name__ == "__main__":
    run()
