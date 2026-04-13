"""
Dash callbacks: read DashboardState and update all chart components.
All callbacks are triggered by a single dcc.Interval (250ms).
"""
import plotly.graph_objects as go
from dash import Input, Output
from .. import config


REGIME_COLORS = {
    "mean_reverting": "#16a34a",
    "trending":       "#d97706",
    "crisis":         "#dc2626",
    "unknown":        "#6b7280",
}

# Shared layout defaults for all charts
CHART_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font={"color": "#111827", "family": "Arial, sans-serif", "size": 13},
    xaxis=dict(
        showgrid=True, gridcolor="#e5e7eb", gridwidth=1,
        showline=True, linecolor="#d1d5db", linewidth=1,
        zeroline=False, tickfont={"size": 12, "color": "#374151"},
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#e5e7eb", gridwidth=1,
        showline=True, linecolor="#d1d5db", linewidth=1,
        zeroline=False, tickfont={"size": 12, "color": "#374151"},
    ),
    title_font={"size": 15, "color": "#111827"},
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)", bordercolor="#d1d5db", borderwidth=1,
        font={"size": 12, "color": "#374151"},
    ),
    margin={"l": 55, "r": 60, "t": 45, "b": 45},
)

SIGNAL_COLORS = ["#7c3aed", "#0891b2", "#b45309", "#be185d", "#065f46"]


def register_callbacks(app, state):

    @app.callback(
        Output("status-tick",   "children"),
        Output("status-regime", "children"),
        Output("status-regime", "style"),
        Output("status-kill",   "children"),
        Output("status-kill",   "style"),
        Output("status-pnl",    "children"),
        Input("interval",       "n_intervals"),
    )
    def update_status(n):
        s = state.snapshot()
        tick_text  = f"Tick {s['tick']} / {s['ticks_per_period']}"
        regime_txt = s["regime"].replace("_", " ").upper()
        regime_style = {
            "backgroundColor": REGIME_COLORS.get(s["regime"], "#6b7280"),
            "color": "white", "padding": "5px 12px", "borderRadius": "5px",
            "fontWeight": "700", "fontSize": "0.85rem", "letterSpacing": "0.05em",
        }
        if s["kill_switch_on"]:
            kill_text  = f"⛔ HALTED: {s['halt_reason']}"
            kill_style = {"backgroundColor": "#dc2626", "color": "white",
                          "padding": "5px 12px", "borderRadius": "5px", "fontWeight": "700"}
        else:
            kill_text  = "● LIVE"
            kill_style = {"backgroundColor": "#16a34a", "color": "white",
                          "padding": "5px 12px", "borderRadius": "5px", "fontWeight": "700"}

        real   = s["realized_pnl"]
        unreal = s["unrealized_pnl"]
        pnl_color = "#16a34a" if real >= 0 else "#dc2626"
        pnl_text  = f"Realized: ${real:+,.2f}  |  Unrealized: ${unreal:+,.2f}"

        return tick_text, regime_txt, regime_style, kill_text, kill_style, pnl_text

    @app.callback(
        Output("price-chart-crzy", "figure"),
        Output("price-chart-tame", "figure"),
        Input("interval", "n_intervals"),
    )
    def update_price_charts(n):
        s    = state.snapshot()
        figs = []
        for ticker in config.TICKERS:
            ticks  = s["tick_history"]
            prices = s["price_history"].get(ticker, [])
            bids   = s["bid_history"].get(ticker, [])
            asks   = s["ask_history"].get(ticker, [])

            length = min(len(ticks), len(prices))
            ticks  = ticks[-length:]
            prices = prices[-length:]
            bids   = bids[-length:]
            asks   = asks[-length:]

            fig = go.Figure()

            # Bid/ask fill band
            if bids and asks:
                fig.add_trace(go.Scatter(
                    x=ticks + ticks[::-1],
                    y=asks + bids[::-1],
                    fill="toself",
                    fillcolor="rgba(59,130,246,0.08)",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                ))

            fig.add_trace(go.Scatter(
                x=ticks, y=bids, name="Bid",
                line={"color": "#10b981", "width": 1.5, "dash": "dot"},
                opacity=0.8,
            ))
            fig.add_trace(go.Scatter(
                x=ticks, y=asks, name="Ask",
                line={"color": "#ef4444", "width": 1.5, "dash": "dot"},
                opacity=0.8,
            ))
            fig.add_trace(go.Scatter(
                x=ticks, y=prices, name="Last",
                line={"color": "#1d4ed8", "width": 2.5},
            ))

            # Alpha signals on secondary y-axis
            for i, (alpha_name, ticker_sigs) in enumerate(s["signal_history"].items()):
                sigs    = ticker_sigs.get(ticker, [])
                sig_len = min(len(ticks), len(sigs))
                if sig_len > 0:
                    color = SIGNAL_COLORS[i % len(SIGNAL_COLORS)]
                    fig.add_trace(go.Scatter(
                        x=ticks[-sig_len:], y=sigs[-sig_len:],
                        name=f"{alpha_name}",
                        yaxis="y2",
                        line={"width": 2, "color": color},
                        opacity=0.85,
                    ))

            layout = {**CHART_LAYOUT,
                "title": f"<b>{ticker}</b> — Price & Alpha Signals",
                "xaxis_title": "Tick",
                "yaxis_title": "Price ($)",
                "yaxis2": {
                    "title": "Signal [-1 → 1]",
                    "overlaying": "y", "side": "right",
                    "range": [-1.3, 1.3],
                    "showgrid": False,
                    "zeroline": True, "zerolinecolor": "#9ca3af", "zerolinewidth": 1,
                    "tickfont": {"size": 11, "color": "#6b7280"},
                    "titlefont": {"size": 12, "color": "#6b7280"},
                },
                "height": 340,
                "legend": {**CHART_LAYOUT["legend"], "orientation": "h",
                           "x": 0, "y": -0.18},
            }
            fig.update_layout(**layout)
            figs.append(fig)

        return figs[0], figs[1]

    @app.callback(
        Output("zscore-chart", "figure"),
        Input("interval", "n_intervals"),
    )
    def update_zscore(n):
        s      = state.snapshot()
        ticks  = s["tick_history"]
        zs     = s["zscore_history"]
        length = min(len(ticks), len(zs))
        ticks  = ticks[-length:]
        zs     = zs[-length:]

        fig = go.Figure()

        # Shaded entry/exit bands
        fig.add_hrect(y0=config.PAIRS_EXIT_Z,  y1=config.PAIRS_ENTRY_Z,
                      fillcolor="rgba(239,68,68,0.07)", line_width=0)
        fig.add_hrect(y0=-config.PAIRS_ENTRY_Z, y1=-config.PAIRS_EXIT_Z,
                      fillcolor="rgba(239,68,68,0.07)", line_width=0)

        # Zero line
        fig.add_hline(y=0, line={"color": "#9ca3af", "width": 1})

        # Entry/exit threshold lines
        for level, color, dash, label in [
            ( config.PAIRS_ENTRY_Z,  "#dc2626", "dash",  f"+{config.PAIRS_ENTRY_Z} entry"),
            (-config.PAIRS_ENTRY_Z,  "#dc2626", "dash",  f"-{config.PAIRS_ENTRY_Z} entry"),
            ( config.PAIRS_EXIT_Z,   "#d97706", "dot",   f"+{config.PAIRS_EXIT_Z} exit"),
            (-config.PAIRS_EXIT_Z,   "#d97706", "dot",   f"-{config.PAIRS_EXIT_Z} exit"),
        ]:
            fig.add_hline(y=level, line={"color": color, "dash": dash, "width": 1.5},
                          annotation_text=label,
                          annotation_position="right",
                          annotation_font={"size": 11, "color": color})

        fig.add_trace(go.Scatter(
            x=ticks, y=zs, name="Z-Score",
            line={"color": "#7c3aed", "width": 2.5},
            fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
        ))

        layout = {**CHART_LAYOUT,
            "title": "<b>Kalman Pairs Z-Score</b>",
            "xaxis_title": "Tick",
            "yaxis_title": "Z-Score",
            "height": 270,
            "showlegend": False,
        }
        fig.update_layout(**layout)
        return fig

    @app.callback(
        Output("pnl-chart", "figure"),
        Input("interval", "n_intervals"),
    )
    def update_pnl(n):
        s    = state.snapshot()
        hist = s["pnl_history"]
        if not hist:
            fig = go.Figure()
            fig.update_layout(**{**CHART_LAYOUT, "title": "<b>Algo PnL</b>", "height": 300})
            return fig

        ticks    = [h[0] for h in hist]
        realized = [h[1] for h in hist]
        total    = [h[2] for h in hist]

        fig = go.Figure()
        fig.add_hline(y=0, line={"color": "#9ca3af", "width": 1})

        fig.add_trace(go.Scatter(
            x=ticks, y=total, name="Total PnL",
            line={"color": "#0891b2", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(8,145,178,0.1)",
        ))
        fig.add_trace(go.Scatter(
            x=ticks, y=realized, name="Realized PnL",
            line={"color": "#16a34a", "width": 2.5},
        ))

        layout = {**CHART_LAYOUT,
            "title": "<b>Algo PnL</b> (excludes manual trades)",
            "xaxis_title": "Tick",
            "yaxis_title": "PnL ($)",
            "height": 300,
        }
        fig.update_layout(**layout)
        return fig

    @app.callback(
        Output("positions-table", "data"),
        Input("interval", "n_intervals"),
    )
    def update_positions(n):
        s   = state.snapshot()
        pos = s["positions"]
        rows = []
        for ticker, info in pos.items():
            rows.append({
                "Ticker":     ticker,
                "Position":   info.get("position", 0),
                "Avg Cost":   f"${info.get('avg_cost', 0):.3f}",
                "Realized":   f"${info.get('realized', 0):+,.2f}",
                "Unrealized": f"${info.get('unrealized', 0):+,.2f}",
            })
        return rows
