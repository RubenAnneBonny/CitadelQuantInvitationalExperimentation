"""
Combined Strategy: Bollinger Band Mean Reversion + Kalman Pairs Trading
------------------------------------------------------------------------
Two independent signal sources are blended per tick:

  1. Bollinger Bands (per-ticker):
       BUY  when price < lower band  (oversold)
       SELL when price > upper band  (overbought)
       CLOSE when price returns inside bands

  2. Kalman Pairs (CRZY vs TAME):
       Models  CRZY_t = alpha_t + beta_t * TAME_t + e_t
       beta_t  is estimated online — it drifts via a random-walk state model.
       The innovation e_t (spread) is normalised to a z-score.
       SELL CRZY / BUY  TAME when z-score >  PAIRS_ENTRY_Z  (spread too wide)
       BUY  CRZY / SELL TAME when z-score < -PAIRS_ENTRY_Z  (spread too narrow)
       Exit when |z-score| < PAIRS_EXIT_Z

Signal priority: pairs signal takes precedence when active; Bollinger runs
on whichever tickers are NOT currently committed to a pairs trade.

Compatible with Rotman Interactive Trader (RIT) REST API v1.
"""

import time
import requests
import numpy as np
from kalman_pairs import KalmanPairFilter

# ── API config ────────────────────────────────────────────────────────────────
API_KEY = "IRJW2AFJ"
BASE    = "http://localhost:9999/v1"
HEADERS = {"X-API-Key": API_KEY}

# ── General parameters ────────────────────────────────────────────────────────
TICKERS           = ["CRZY", "TAME"]
PAIR              = ("CRZY", "TAME")   # y, x  (CRZY = alpha + beta*TAME + e)
MAX_POSITION      = 10_000
TRADE_SIZE        = 500
STOP_TRADING_TICK = 290
MIN_HISTORY       = 3
POLL_INTERVAL     = 0.25

# ── Bollinger Band parameters ─────────────────────────────────────────────────
BB_WINDOW     = 20
BB_WIDTH      = 1.5

# ── Kalman pairs parameters ───────────────────────────────────────────────────
KF_DELTA      = 1e-3    # how fast beta is allowed to drift
PAIRS_ENTRY_Z = 2.0     # z-score to open a pairs trade
PAIRS_EXIT_Z  = 0.5     # z-score to close a pairs trade


# ── API helpers ───────────────────────────────────────────────────────────────

def get_case():
    return requests.get(f"{BASE}/case", headers=HEADERS).json()

def get_securities():
    return requests.get(f"{BASE}/securities", headers=HEADERS).json()

def get_book(ticker):
    return requests.get(
        f"{BASE}/securities/book", params={"ticker": ticker}, headers=HEADERS
    ).json()

def place_order(ticker, action, qty, price, order_type="LIMIT"):
    return requests.post(f"{BASE}/orders", headers=HEADERS, params={
        "ticker":   ticker,
        "type":     order_type,
        "quantity": qty,
        "action":   action,
        "price":    price,
    }).json()

def cancel_all():
    try:
        requests.post(f"{BASE}/commands/cancel", params={"all": 1}, headers=HEADERS)
    except Exception as e:
        print(f"[WARN] cancel_all: {e}")

def get_positions():
    return {s["ticker"]: s["position"] for s in get_securities()
            if s["ticker"] in TICKERS}

def get_best_bid_ask(ticker):
    book = get_book(ticker)
    bid  = book["bids"][0]["price"] if book["bids"] else None
    ask  = book["asks"][0]["price"] if book["asks"] else None
    return bid, ask

def flatten_position(ticker):
    positions = get_positions()
    pos = positions.get(ticker, 0)
    if pos == 0:
        return
    qty    = min(abs(int(pos)), 25_000)
    action = "SELL" if pos > 0 else "BUY"
    bid, ask = get_best_bid_ask(ticker)
    price  = bid if action == "SELL" else ask
    if price is None:
        return
    print(f"[FLATTEN] {action} {qty} {ticker} @ {price:.2f}")
    place_order(ticker, action, qty, price, order_type="MARKET")


# ── Bollinger Band signal ─────────────────────────────────────────────────────

def bb_signal(price_history: list) -> str:
    arr  = np.array(price_history[-BB_WINDOW:])
    mean = arr.mean()
    std  = arr.std()
    if std == 0:
        return "HOLD"
    current = price_history[-1]
    if current < mean - BB_WIDTH * std:
        return "BUY"
    elif current > mean + BB_WIDTH * std:
        return "SELL"
    return "CLOSE"


def execute_bb(ticker, signal, price_histories, positions):
    pos      = positions.get(ticker, 0)
    bid, ask = get_best_bid_ask(ticker)
    price    = price_histories[ticker][-1]

    arr  = np.array(price_histories[ticker][-BB_WINDOW:])
    mean, std = arr.mean(), arr.std()
    upper = mean + BB_WIDTH * std
    lower = mean - BB_WIDTH * std
    print(f"  [BB] {ticker}: {price:.2f}  [{lower:.2f}|{mean:.2f}|{upper:.2f}]  → {signal}")

    if signal == "BUY" and pos < MAX_POSITION and ask is not None:
        qty = min(TRADE_SIZE, MAX_POSITION - int(pos))
        place_order(ticker, "BUY", qty, ask)

    elif signal == "SELL" and pos > -MAX_POSITION and bid is not None:
        qty = min(TRADE_SIZE, MAX_POSITION + int(pos))
        place_order(ticker, "SELL", qty, bid)

    elif signal == "CLOSE":
        if pos > 0 and bid is not None:
            place_order(ticker, "SELL", min(int(pos), TRADE_SIZE), bid)
        elif pos < 0 and ask is not None:
            place_order(ticker, "BUY", min(abs(int(pos)), TRADE_SIZE), ask)


# ── Kalman pairs signal ───────────────────────────────────────────────────────

def execute_pairs(zscore: float, beta: float, positions: dict):
    """
    Spread = CRZY - alpha - beta*TAME.
    z > 0  → spread is too wide → SELL CRZY, BUY TAME (weighted by beta)
    z < 0  → spread is too narrow → BUY CRZY, SELL TAME
    """
    ticker_y, ticker_x = PAIR
    pos_y = positions.get(ticker_y, 0)
    pos_x = positions.get(ticker_x, 0)

    bid_y, ask_y = get_best_bid_ask(ticker_y)
    bid_x, ask_x = get_best_bid_ask(ticker_x)

    # Hedge qty for x leg scaled by beta (round to nearest lot)
    qty_y = TRADE_SIZE
    qty_x = max(1, round(abs(beta) * TRADE_SIZE))

    if abs(zscore) < PAIRS_EXIT_Z:
        # Close pairs positions if we have any
        if pos_y != 0 and bid_y and ask_y:
            action = "SELL" if pos_y > 0 else "BUY"
            price  = bid_y if action == "SELL" else ask_y
            print(f"  [PAIRS EXIT] {action} {min(abs(int(pos_y)), TRADE_SIZE)} {ticker_y}")
            place_order(ticker_y, action, min(abs(int(pos_y)), TRADE_SIZE), price)
        if pos_x != 0 and bid_x and ask_x:
            action = "SELL" if pos_x > 0 else "BUY"
            price  = bid_x if action == "SELL" else ask_x
            print(f"  [PAIRS EXIT] {action} {min(abs(int(pos_x)), TRADE_SIZE)} {ticker_x}")
            place_order(ticker_x, action, min(abs(int(pos_x)), TRADE_SIZE), price)

    elif zscore > PAIRS_ENTRY_Z:
        # Spread too wide: SELL y, BUY x
        if pos_y > -MAX_POSITION and bid_y:
            qty = min(qty_y, MAX_POSITION + int(pos_y))
            print(f"  [PAIRS] SELL {qty} {ticker_y} @ {bid_y:.2f}  z={zscore:.2f}  β={beta:.3f}")
            place_order(ticker_y, "SELL", qty, bid_y)
        if pos_x < MAX_POSITION and ask_x:
            qty = min(qty_x, MAX_POSITION - int(pos_x))
            print(f"  [PAIRS] BUY  {qty} {ticker_x} @ {ask_x:.2f}")
            place_order(ticker_x, "BUY", qty, ask_x)

    elif zscore < -PAIRS_ENTRY_Z:
        # Spread too narrow: BUY y, SELL x
        if pos_y < MAX_POSITION and ask_y:
            qty = min(qty_y, MAX_POSITION - int(pos_y))
            print(f"  [PAIRS] BUY  {qty} {ticker_y} @ {ask_y:.2f}  z={zscore:.2f}  β={beta:.3f}")
            place_order(ticker_y, "BUY", qty, ask_y)
        if pos_x > -MAX_POSITION and bid_x:
            qty = min(qty_x, MAX_POSITION + int(pos_x))
            print(f"  [PAIRS] SELL {qty} {ticker_x} @ {bid_x:.2f}")
            place_order(ticker_x, "SELL", qty, bid_x)

    else:
        print(f"  [PAIRS] HOLD  z={zscore:.2f}  β={beta:.3f}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    price_histories = {t: [] for t in TICKERS}
    kf              = KalmanPairFilter(delta=KF_DELTA)
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

        # ── End-of-case flatten ──────────────────────────────────────────────
        if tick >= STOP_TRADING_TICK:
            print(f"\n[Tick {tick}] Flattening all positions...")
            cancel_all()
            for ticker in TICKERS:
                flatten_position(ticker)
            print("Done.")
            break

        # ── Snapshot ─────────────────────────────────────────────────────────
        securities = get_securities()
        prices = {s["ticker"]: s["last"] for s in securities if s["ticker"] in TICKERS}
        for ticker in TICKERS:
            if ticker in prices:
                price_histories[ticker].append(prices[ticker])

        min_history = min(len(price_histories[t]) for t in TICKERS)
        print(f"\n[Tick {tick}/{case['ticks_per_period']}]", end="  ")

        if min_history < MIN_HISTORY:
            print(f"Collecting ({min_history}/{MIN_HISTORY})")
            time.sleep(POLL_INTERVAL)
            continue

        # ── Kalman filter update ─────────────────────────────────────────────
        ticker_y, ticker_x = PAIR
        kf_state = kf.update(
            y=price_histories[ticker_y][-1],
            x=price_histories[ticker_x][-1],
        )
        zscore = kf_state["zscore"]
        beta   = kf_state["beta"]
        print(f"spread={kf_state['spread']:+.3f}  z={zscore:+.2f}  β={beta:.3f}")

        # ── Execute signals ──────────────────────────────────────────────────
        cancel_all()
        positions = get_positions()

        # Pairs signal (both legs)
        execute_pairs(zscore, beta, positions)

        # Bollinger on each ticker independently
        for ticker in TICKERS:
            signal = bb_signal(price_histories[ticker])
            execute_bb(ticker, signal, price_histories, positions)

        time.sleep(POLL_INTERVAL)

    print("\n=== TRADING COMPLETE ===")
    positions = get_positions()
    for ticker in TICKERS:
        print(f"Final position {ticker}: {positions.get(ticker, 0)}")


if __name__ == "__main__":
    run()
