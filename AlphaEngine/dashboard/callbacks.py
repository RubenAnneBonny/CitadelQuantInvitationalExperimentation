"""
Dash callbacks: read DashboardState and update all chart components.
All callbacks are triggered by a single dcc.Interval (250ms).
"""
import plotly.graph_objects as go
from dash import Input, Output, callback_context
from .. import config


REGIME_COLORS = {
    "mean_reverting": "#27ae60",
    "trending":       "#e67e22",
    "crisis":         "#e74c3c",
    "unknown":        "#95a5a6",
}


def register_callbacks(app, state):

    @app.callback(
        Output("status-tick",    "children"),
        Output("status-regime",  "children"),
        Output("status-regime",  "style"),
        Output("status-kill",    "children"),
        Output("status-kill",    "style"),
        Output("status-pnl",     "children"),
        Input("interval",        "n_intervals"),
    )
    def update_status(n):
        s = state.snapshot()
        tick_text   = f"Tick {s['tick']} / {s['ticks_per_period']}"
        regime_text = s["regime"].upper()
        regime_style = {
            "backgroundColor": REGIME_COLORS.get(s["regime"], "#95a5a6"),
            "color": "white", "padding": "4px 10px", "borderRadius": "4px",
            "fontWeight": "bold",
        }
        if s["kill_switch_on"]:
            kill_text  = f"HALTED: {s['halt_reason']}"
            kill_style = {"backgroundColor": "#e74c3c", "color": "white",
                          "padding": "4px 10px", "borderRadius": "4px", "fontWeight": "bold"}
        else:
            kill_text  = "LIVE"
            kill_style = {"backgroundColor": "#27ae60", "color": "white",
                          "padding": "4px 10px", "borderRadius": "4px", "fontWeight": "bold"}

        real  = s["realized_pnl"]
        unreal = s["unrealized_pnl"]
        pnl_color = "#27ae60" if real >= 0 else "#e74c3c"
        pnl_text  = f"Realized: ${real:+.2f}  |  Unrealized: ${unreal:+.2f}"

        return tick_text, regime_text, regime_style, kill_text, kill_style, pnl_text

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

            # Trim to same length
            length = min(len(ticks), len(prices))
            ticks  = ticks[-length:]
            prices = prices[-length:]
            bids   = bids[-length:]
            asks   = asks[-length:]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ticks, y=prices, name="Last", line={"color": "#3498db", "width": 2}
            ))
            fig.add_trace(go.Scatter(
                x=ticks, y=bids, name="Bid",
                line={"color": "#27ae60", "width": 1, "dash": "dot"}, opacity=0.6
            ))
            fig.add_trace(go.Scatter(
                x=ticks, y=asks, name="Ask",
                line={"color": "#e74c3c", "width": 1, "dash": "dot"}, opacity=0.6
            ))

            # Overlay alpha signals on secondary y-axis
            for alpha_name, ticker_sigs in s["signal_history"].items():
                sigs = ticker_sigs.get(ticker, [])
                sig_len = min(len(ticks), len(sigs))
                if sig_len > 0:
                    fig.add_trace(go.Scatter(
                        x=ticks[-sig_len:], y=sigs[-sig_len:],
                        name=f"{alpha_name} signal",
                        yaxis="y2", line={"width": 1}, opacity=0.7,
                    ))

            fig.update_layout(
                title=f"{ticker} Price & Signals",
                xaxis_title="Tick",
                yaxis_title="Price",
                yaxis2={"title": "Signal", "overlaying": "y", "side": "right",
                        "range": [-1.1, 1.1]},
                height=300,
                margin={"l": 40, "r": 60, "t": 40, "b": 30},
                legend={"orientation": "h", "y": -0.2},
                paper_bgcolor="#1e1e2e",
                plot_bgcolor="#2a2a3e",
                font={"color": "#cdd6f4"},
            )
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
        fig.add_trace(go.Scatter(
            x=ticks, y=zs, name="Z-Score",
            line={"color": "#cba6f7", "width": 2}, fill="tozeroy",
            fillcolor="rgba(203,166,247,0.15)",
        ))
        for level, color, dash in [
            (config.PAIRS_ENTRY_Z,  "#e74c3c", "dash"),
            (-config.PAIRS_ENTRY_Z, "#e74c3c", "dash"),
            (config.PAIRS_EXIT_Z,   "#f1c40f", "dot"),
            (-config.PAIRS_EXIT_Z,  "#f1c40f", "dot"),
        ]:
            fig.add_hline(y=level, line={"color": color, "dash": dash, "width": 1})

        fig.update_layout(
            title="Kalman Pairs Z-Score",
            xaxis_title="Tick", yaxis_title="Z-Score",
            height=250, margin={"l": 40, "r": 40, "t": 40, "b": 30},
            paper_bgcolor="#1e1e2e", plot_bgcolor="#2a2a3e", font={"color": "#cdd6f4"},
        )
        return fig

    @app.callback(
        Output("pnl-chart", "figure"),
        Input("interval", "n_intervals"),
    )
    def update_pnl(n):
        s    = state.snapshot()
        hist = s["pnl_history"]
        if not hist:
            return go.Figure()

        ticks    = [h[0] for h in hist]
        realized = [h[1] for h in hist]
        total    = [h[2] for h in hist]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ticks, y=realized, name="Realized PnL",
            line={"color": "#a6e3a1", "width": 2},
        ))
        fig.add_trace(go.Scatter(
            x=ticks, y=total, name="Total PnL",
            line={"color": "#89dceb", "width": 2, "dash": "dash"},
            fill="tonexty",
            fillcolor="rgba(137,220,235,0.1)",
        ))
        fig.add_hline(y=0, line={"color": "#6c7086", "width": 1})

        fig.update_layout(
            title="Algo PnL (excludes manual trades)",
            xaxis_title="Tick", yaxis_title="PnL ($)",
            height=280, margin={"l": 40, "r": 40, "t": 40, "b": 30},
            paper_bgcolor="#1e1e2e", plot_bgcolor="#2a2a3e", font={"color": "#cdd6f4"},
        )
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
                "Ticker":    ticker,
                "Position":  info.get("position", 0),
                "Avg Cost":  f"{info.get('avg_cost', 0):.3f}",
                "Realized":  f"${info.get('realized', 0):+.2f}",
                "Unrealized": f"${info.get('unrealized', 0):+.2f}",
            })
        return rows
