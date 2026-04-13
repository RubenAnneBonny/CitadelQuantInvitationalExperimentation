"""
Alpha Signal Dashboard for Rotman Interactive Trader
------------------------------------------------------
Connects to RIT REST API, calculates alpha signals in real time,
and displays a live dashboard to help you decide BUY / SELL / HOLD.

Alphas calculated:
  1. Volume-Price Divergence  → short-term mean reversion signal
  2. Short-term Momentum      → trend continuation signal
  3. Bollinger Band Z-Score   → mean reversion entry timing
  4. Pair Z-Score (CRZY/TAME) → spread trading signal
  5. Composite Signal         → weighted combination of all above

HOW TO USE:
  1. Make sure RIT is running and your case is ACTIVE
  2. Change API_KEY below to match your RIT session
  3. Change TICKERS to match your case's assets
  4. Run:  python alpha_dashboard.py
  5. A live window will open — use it alongside RIT to decide when to trade
"""

import time
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from collections import deque
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams

# ── CONFIG — change these to match your RIT session ──────────────────────────
API_KEY  = "IRJW2AFJ"        # your RIT API key
BASE     = "http://localhost:9999/v1"
HEADERS  = {"X-API-Key": API_KEY}

TICKERS  = ["CRZY", "TAME"]  # assets to track
PAIR     = ("CRZY", "TAME")  # y, x for pair trading

# ── ALPHA PARAMETERS ─────────────────────────────────────────────────────────
VOL_PRICE_WINDOW  = 6    # lookback for volume-price divergence
MOMENTUM_WINDOW   = 10   # lookback for momentum
BB_WINDOW         = 20   # Bollinger Band window
BB_WIDTH          = 2.0  # standard deviations for BB
PAIR_WINDOW       = 30   # lookback for pair z-score
MAX_HISTORY       = 200  # max data points to keep

# ── STYLE ─────────────────────────────────────────────────────────────────────
rcParams['font.family']      = 'monospace'
rcParams['figure.facecolor'] = '#0a0e1a'
rcParams['text.color']       = '#e0e0e0'

BG        = '#0a0e1a'
BG2       = '#111827'
GREEN     = '#00ff88'
RED       = '#ff4466'
YELLOW    = '#ffd700'
BLUE      = '#4488ff'
PURPLE    = '#aa66ff'
GRAY      = '#444466'
WHITE     = '#e0e0e0'


# ── RIT API helpers ───────────────────────────────────────────────────────────

def get_case():
    try:
        return requests.get(f"{BASE}/case", headers=HEADERS, timeout=2).json()
    except:
        return {"status": "OFFLINE", "tick": 0, "ticks_per_period": 300}

def get_securities():
    try:
        return requests.get(f"{BASE}/securities", headers=HEADERS, timeout=2).json()
    except:
        return []

def get_book(ticker):
    try:
        return requests.get(
            f"{BASE}/securities/book",
            params={"ticker": ticker},
            headers=HEADERS,
            timeout=2
        ).json()
    except:
        return {"bids": [], "asks": []}

def get_mid(ticker):
    book = get_book(ticker)
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    if not bids or not asks:
        return None, None, None
    bid = bids[0]["price"]
    ask = asks[0]["price"]
    return (bid + ask) / 2.0, bid, ask

def get_positions():
    secs = get_securities()
    return {s["ticker"]: s.get("position", 0) for s in secs if s["ticker"] in TICKERS}


# ── ALPHA CALCULATIONS ────────────────────────────────────────────────────────

def alpha_vol_price_divergence(prices, volumes, window=VOL_PRICE_WINDOW):
    """
    -1 * correlation(rank(delta(log(volume), 2)), rank((close-open)/open), window)
    Simplified: correlation between recent volume change and price change.
    Positive = volume and price diverging = potential BUY signal.
    """
    if len(prices) < window + 2 or len(volumes) < window + 2:
        return 0.0
    p = np.array(prices[-(window + 2):])
    v = np.array(volumes[-(window + 2):])
    price_changes  = np.diff(p) / (p[:-1] + 1e-9)
    volume_changes = np.diff(np.log(v + 1))
    n = min(len(price_changes), len(volume_changes), window)
    if n < 3:
        return 0.0
    pc = price_changes[-n:]
    vc = volume_changes[-n:]
    if pc.std() == 0 or vc.std() == 0:
        return 0.0
    corr = np.corrcoef(pc, vc)[0, 1]
    return float(-corr) if not np.isnan(corr) else 0.0


def alpha_momentum(prices, window=MOMENTUM_WINDOW):
    """
    Simple cross-period momentum: current price vs price N periods ago.
    Positive = upward momentum = BUY signal.
    """
    if len(prices) < window + 1:
        return 0.0
    ret = (prices[-1] - prices[-window - 1]) / (prices[-window - 1] + 1e-9)
    # Normalize to roughly [-1, 1]
    return float(np.tanh(ret * 20))


def alpha_bollinger(prices, window=BB_WINDOW, width=BB_WIDTH):
    """
    Bollinger Band Z-score.
    Negative (price below band) = oversold = BUY signal.
    Positive (price above band) = overbought = SELL signal.
    Returns z-score (negative = buy, positive = sell).
    """
    if len(prices) < window:
        return 0.0
    arr  = np.array(prices[-window:])
    mean = arr.mean()
    std  = arr.std()
    if std == 0:
        return 0.0
    z = (prices[-1] - mean) / std
    return float(-np.tanh(z))  # flip: negative z means oversold = positive alpha


def alpha_pair_zscore(prices_y, prices_x, window=PAIR_WINDOW):
    """
    Simple OLS spread z-score between two assets.
    Positive = spread unusually wide = BUY y, SELL x.
    Negative = spread unusually narrow = SELL y, BUY x.
    """
    n = min(len(prices_y), len(prices_x), window)
    if n < 10:
        return 0.0
    y = np.array(prices_y[-n:])
    x = np.array(prices_x[-n:])
    # OLS beta
    beta = np.cov(y, x)[0, 1] / (np.var(x) + 1e-9)
    spread = y - beta * x
    mean   = spread.mean()
    std    = spread.std()
    if std == 0:
        return 0.0
    z = (spread[-1] - mean) / std
    return float(-np.tanh(z))  # negative z = spread too low = buy spread


def composite_signal(vp, mom, bb, pair):
    """
    Weighted combination of all alphas.
    Returns value in [-1, 1]:
      > 0.3  → BUY
      < -0.3 → SELL
      else   → HOLD
    """
    weights = [0.35, 0.25, 0.25, 0.15]
    signals = [vp, mom, bb, pair]
    return float(np.tanh(sum(w * s for w, s in zip(weights, signals))))


def signal_label(value, thresholds=(0.25, 0.5)):
    lo, hi = thresholds
    if value > hi:
        return "STRONG BUY",  GREEN,  "▲▲"
    elif value > lo:
        return "BUY",         GREEN,  "▲"
    elif value < -hi:
        return "STRONG SELL", RED,    "▼▼"
    elif value < -lo:
        return "SELL",        RED,    "▼"
    else:
        return "HOLD",        YELLOW, "●"


# ── STATE ─────────────────────────────────────────────────────────────────────

histories = {t: {"price": deque(maxlen=MAX_HISTORY),
                  "volume": deque(maxlen=MAX_HISTORY)} for t in TICKERS}

alpha_histories = {
    t: {
        "vol_price": deque(maxlen=MAX_HISTORY),
        "momentum":  deque(maxlen=MAX_HISTORY),
        "bollinger": deque(maxlen=MAX_HISTORY),
        "composite": deque(maxlen=MAX_HISTORY),
    } for t in TICKERS
}

pair_zscore_history = deque(maxlen=MAX_HISTORY)
tick_history        = deque(maxlen=MAX_HISTORY)
status_text         = {"tick": 0, "total": 300, "status": "WAITING", "pnl": {}}


# ── FIGURE SETUP ──────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 11), facecolor=BG)
fig.canvas.manager.set_window_title("Alpha Signal Dashboard — RIT")

gs = gridspec.GridSpec(
    4, 4,
    figure=fig,
    hspace=0.55,
    wspace=0.35,
    left=0.05, right=0.97,
    top=0.91,  bottom=0.06
)

# Row 0: header signal boxes (one per ticker + pair + composite)
ax_sig  = [fig.add_subplot(gs[0, i]) for i in range(4)]

# Row 1-2: price + alpha chart per ticker (2 cols each)
ax_price = [fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[1, 2:])]
ax_alpha = [fig.add_subplot(gs[2, :2]), fig.add_subplot(gs[2, 2:])]

# Row 3: pair z-score full width + info box
ax_pair  = fig.add_subplot(gs[3, :3])
ax_info  = fig.add_subplot(gs[3, 3])

all_axes = ax_sig + ax_price + ax_alpha + [ax_pair, ax_info]
for ax in all_axes:
    ax.set_facecolor(BG2)
    for spine in ax.spines.values():
        spine.set_color(GRAY)


def style_ax(ax, title, ylabel="", xlabel=""):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=WHITE, labelsize=7)
    ax.set_title(title, color=WHITE, fontsize=8, pad=4, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, color=GRAY, fontsize=7)
    if xlabel:
        ax.set_xlabel(xlabel, color=GRAY, fontsize=7)
    ax.grid(True, color=GRAY, alpha=0.2, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(GRAY)


# ── DRAW SIGNAL BOX ───────────────────────────────────────────────────────────

def draw_signal_box(ax, title, value, extra_lines=None):
    ax.clear()
    ax.set_facecolor(BG2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    label, color, icon = signal_label(value)

    # Background glow rectangle
    rect = FancyBboxPatch(
        (0.05, 0.05), 0.9, 0.9,
        boxstyle="round,pad=0.02",
        linewidth=2,
        edgecolor=color,
        facecolor=BG2,
        alpha=0.9
    )
    ax.add_patch(rect)

    ax.text(0.5, 0.88, title,   ha='center', va='top',    fontsize=9,  color=GRAY,  fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.62, icon,    ha='center', va='center', fontsize=20, color=color, transform=ax.transAxes)
    ax.text(0.5, 0.38, label,   ha='center', va='center', fontsize=10, color=color, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.18, f"{value:+.3f}", ha='center', va='center', fontsize=8, color=WHITE, transform=ax.transAxes)

    if extra_lines:
        for i, line in enumerate(extra_lines):
            ax.text(0.5, 0.08 - i * 0.06, line, ha='center', va='center', fontsize=6, color=GRAY, transform=ax.transAxes)


# ── UPDATE FUNCTION ───────────────────────────────────────────────────────────

def update(_frame):
    case  = get_case()
    secs  = get_securities()
    tick  = case.get("tick", 0)
    total = case.get("ticks_per_period", 300)
    cstat = case.get("status", "UNKNOWN")

    # Pull prices and volumes
    prices_now = {s["ticker"]: s.get("last", 0) for s in secs if s["ticker"] in TICKERS}
    volumes_now = {s["ticker"]: s.get("volume", 0) for s in secs if s["ticker"] in TICKERS}
    positions   = {s["ticker"]: s.get("position", 0) for s in secs if s["ticker"] in TICKERS}
    pnl_now     = {s["ticker"]: s.get("unrealized_pl", 0) for s in secs if s["ticker"] in TICKERS}

    for t in TICKERS:
        if t in prices_now and prices_now[t] > 0:
            histories[t]["price"].append(prices_now[t])
            histories[t]["volume"].append(max(volumes_now.get(t, 1), 1))

    tick_history.append(tick)

    # ── Calculate alphas ──────────────────────────────────────────────────────
    alphas = {}
    for t in TICKERS:
        p = list(histories[t]["price"])
        v = list(histories[t]["volume"])

        vp   = alpha_vol_price_divergence(p, v)
        mom  = alpha_momentum(p)
        bb   = alpha_bollinger(p)
        comp = composite_signal(vp, mom, bb, 0)  # pair added below

        alpha_histories[t]["vol_price"].append(vp)
        alpha_histories[t]["momentum"].append(mom)
        alpha_histories[t]["bollinger"].append(bb)
        alpha_histories[t]["composite"].append(comp)
        alphas[t] = {"vol_price": vp, "momentum": mom, "bollinger": bb, "composite": comp}

    # Pair z-score (uses first ticker as y)
    py = list(histories[PAIR[0]]["price"])
    px = list(histories[PAIR[1]]["price"])
    pz = alpha_pair_zscore(py, px)
    pair_zscore_history.append(pz)

    # Recompute composite with pair signal
    for t in TICKERS:
        a = alphas[t]
        comp_full = composite_signal(a["vol_price"], a["momentum"], a["bollinger"], pz)
        alpha_histories[t]["composite"][-1] = comp_full
        alphas[t]["composite"] = comp_full

    # ── Draw signal boxes (row 0) ─────────────────────────────────────────────
    for i, t in enumerate(TICKERS[:2]):
        a = alphas[t]
        pos = positions.get(t, 0)
        pnl = pnl_now.get(t, 0)
        draw_signal_box(
            ax_sig[i], t,
            a["composite"],
            extra_lines=[f"pos={int(pos):+d}  PnL={pnl:+.0f}"]
        )

    # Pair signal box
    draw_signal_box(ax_sig[2], f"PAIR {PAIR[0]}/{PAIR[1]}", pz,
                    extra_lines=[f"z-score={pz:+.3f}"])

    # Overall composite (average of both tickers)
    overall = np.mean([alphas[t]["composite"] for t in TICKERS[:2]])
    draw_signal_box(ax_sig[3], "COMPOSITE", overall,
                    extra_lines=[f"tick {tick}/{total}  {cstat}"])

    # ── Price charts (row 1) ──────────────────────────────────────────────────
    for i, t in enumerate(TICKERS[:2]):
        ax = ax_price[i]
        ax.clear()
        style_ax(ax, f"{t} — Price", ylabel="$")

        p = list(histories[t]["price"])
        if len(p) < 2:
            continue

        ticks = list(range(len(p)))
        ax.plot(ticks, p, color=BLUE, linewidth=1.2, label="price")

        # Bollinger bands
        if len(p) >= BB_WINDOW:
            arr  = np.array(p)
            mean = [arr[max(0, j - BB_WINDOW):j].mean() for j in range(BB_WINDOW, len(p) + 1)]
            std  = [arr[max(0, j - BB_WINDOW):j].std()  for j in range(BB_WINDOW, len(p) + 1)]
            xs   = list(range(BB_WINDOW - 1, len(p)))
            upper = [m + BB_WIDTH * s for m, s in zip(mean, std)]
            lower = [m - BB_WIDTH * s for m, s in zip(mean, std)]
            ax.plot(xs, mean,  color=YELLOW, linewidth=0.8, linestyle='--', alpha=0.7, label="BB mid")
            ax.plot(xs, upper, color=RED,    linewidth=0.6, linestyle=':',  alpha=0.7, label="BB upper")
            ax.plot(xs, lower, color=GREEN,  linewidth=0.6, linestyle=':',  alpha=0.7, label="BB lower")
            ax.fill_between(xs, lower, upper, color=BLUE, alpha=0.04)

        # Mark current signal
        label, color, icon = signal_label(alphas[t]["composite"])
        ax.axvline(x=len(p) - 1, color=color, linewidth=1.5, alpha=0.8)
        ax.text(0.98, 0.95, f"{icon} {label}", transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color=color, fontweight='bold')
        ax.legend(fontsize=6, loc='upper left',
                  facecolor=BG2, edgecolor=GRAY, labelcolor=WHITE)

    # ── Alpha charts (row 2) ──────────────────────────────────────────────────
    alpha_colors = {
        "vol_price": PURPLE,
        "momentum":  BLUE,
        "bollinger": YELLOW,
        "composite": GREEN,
    }
    alpha_labels = {
        "vol_price": "Vol/Price Div",
        "momentum":  "Momentum",
        "bollinger": "Bollinger",
        "composite": "Composite",
    }

    for i, t in enumerate(TICKERS[:2]):
        ax = ax_alpha[i]
        ax.clear()
        style_ax(ax, f"{t} — Alpha Signals", ylabel="signal")
        ax.axhline(y=0,     color=GRAY,  linewidth=0.8, alpha=0.5)
        ax.axhline(y=0.25,  color=GREEN, linewidth=0.6, linestyle='--', alpha=0.5)
        ax.axhline(y=-0.25, color=RED,   linewidth=0.6, linestyle='--', alpha=0.5)
        ax.axhline(y=0.5,   color=GREEN, linewidth=0.6, linestyle=':',  alpha=0.4)
        ax.axhline(y=-0.5,  color=RED,   linewidth=0.6, linestyle=':',  alpha=0.4)

        # Shade BUY/SELL regions
        ax.axhspan(0.25,  1.5, color=GREEN, alpha=0.04)
        ax.axhspan(-1.5, -0.25, color=RED,   alpha=0.04)

        for key, color in alpha_colors.items():
            vals = list(alpha_histories[t][key])
            if not vals:
                continue
            lw = 2.0 if key == "composite" else 0.9
            ax.plot(vals, color=color, linewidth=lw,
                    alpha=1.0 if key == "composite" else 0.7,
                    label=alpha_labels[key])

        ax.set_ylim(-1.2, 1.2)
        ax.legend(fontsize=6, loc='upper left',
                  facecolor=BG2, edgecolor=GRAY, labelcolor=WHITE, ncol=2)

        # Latest values annotation
        latest = {k: list(v)[-1] if v else 0 for k, v in alpha_histories[t].items()}
        ann = "  ".join(f"{alpha_labels[k][:3]}={v:+.2f}" for k, v in latest.items())
        ax.text(0.5, -0.14, ann, transform=ax.transAxes,
                ha='center', fontsize=6.5, color=GRAY)

    # ── Pair z-score chart (row 3 left) ──────────────────────────────────────
    ax_pair.clear()
    style_ax(ax_pair, f"Pair Z-Score: {PAIR[0]} vs {PAIR[1]}", ylabel="z-score")
    pzh = list(pair_zscore_history)
    if len(pzh) > 1:
        ax_pair.plot(pzh, color=PURPLE, linewidth=1.5)
        ax_pair.axhline(y=0,     color=GRAY,  linewidth=0.8, alpha=0.6)
        ax_pair.axhline(y=0.5,   color=GREEN, linewidth=0.8, linestyle='--', alpha=0.7,
                        label="buy spread threshold")
        ax_pair.axhline(y=-0.5,  color=RED,   linewidth=0.8, linestyle='--', alpha=0.7,
                        label="sell spread threshold")
        ax_pair.fill_between(range(len(pzh)), pzh, 0,
                             where=[z > 0.5  for z in pzh], color=GREEN, alpha=0.15)
        ax_pair.fill_between(range(len(pzh)), pzh, 0,
                             where=[z < -0.5 for z in pzh], color=RED,   alpha=0.15)

        # Current z annotation
        cur_z = pzh[-1]
        lbl, col, ico = signal_label(cur_z)
        ax_pair.text(0.99, 0.95, f"z = {cur_z:+.3f}  {ico} {lbl}",
                     transform=ax_pair.transAxes,
                     ha='right', va='top', fontsize=9, color=col, fontweight='bold')
    ax_pair.set_ylim(-1.3, 1.3)
    ax_pair.legend(fontsize=6, loc='upper left',
                   facecolor=BG2, edgecolor=GRAY, labelcolor=WHITE)

    # ── Info panel (row 3 right) ──────────────────────────────────────────────
    ax_info.clear()
    ax_info.set_facecolor(BG2)
    ax_info.axis('off')
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)

    lines = [
        ("LIVE INFO", WHITE, 10, True),
        (f"Tick:   {tick} / {total}", WHITE, 8, False),
        (f"Status: {cstat}", GREEN if cstat == 'ACTIVE' else RED, 8, False),
        ("", WHITE, 6, False),
        ("POSITIONS", GRAY, 8, True),
    ]
    for t in TICKERS:
        pos = positions.get(t, 0)
        pnl = pnl_now.get(t, 0)
        col = GREEN if pnl >= 0 else RED
        lines.append((f"{t}: {int(pos):+d}  PnL={pnl:+.1f}", col, 8, False))

    lines += [
        ("", WHITE, 6, False),
        ("THRESHOLDS", GRAY, 8, True),
        ("> +0.50  STRONG BUY",  GREEN,  7, False),
        ("> +0.25  BUY",         GREEN,  7, False),
        ("  ±0.25  HOLD",        YELLOW, 7, False),
        ("< -0.25  SELL",        RED,    7, False),
        ("< -0.50  STRONG SELL", RED,    7, False),
    ]

    y = 0.97
    for text, color, size, bold in lines:
        if not text:
            y -= 0.04
            continue
        fw = 'bold' if bold else 'normal'
        ax_info.text(0.08, y, text, color=color, fontsize=size,
                     fontweight=fw, va='top', transform=ax_info.transAxes)
        y -= 0.07 if size >= 8 else 0.06

    # ── Main title ────────────────────────────────────────────────────────────
    fig.suptitle(
        "⚡  ALPHA SIGNAL DASHBOARD  —  ROTMAN INTERACTIVE TRADER",
        color=WHITE, fontsize=13, fontweight='bold', y=0.975
    )


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Alpha Dashboard...")
    print(f"Connecting to RIT at {BASE}")
    print("Make sure your RIT case is ACTIVE.\n")
    print("Alphas being calculated:")
    print("  1. Volume-Price Divergence  (window=6)")
    print("  2. Short-term Momentum      (window=10)")
    print("  3. Bollinger Band Z-Score   (window=20)")
    print("  4. Pair Z-Score CRZY/TAME   (window=30)")
    print("  5. Composite Signal         (weighted average)\n")

    ani = animation.FuncAnimation(
        fig, update,
        interval=500,   # refresh every 500ms
        cache_frame_data=False
    )

    plt.show()