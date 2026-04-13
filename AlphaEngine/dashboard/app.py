"""
Dashboard app: Dash layout + server factory.

Panels:
  1. Status bar   — tick, regime, kill switch, PnL summary
  2. Price charts — per-ticker price + BB bands + alpha signal overlays
  3. Z-score      — Kalman pairs z-score with entry/exit thresholds
  4. PnL chart    — realized (solid) vs total (dashed), algo-only
  5. Positions    — per-ticker position, avg cost, PnL breakdown
"""
import dash
from dash import dcc, html, dash_table
from .callbacks import register_callbacks
from .. import config


def create_app(state) -> dash.Dash:
    app = dash.Dash(
        __name__,
        title="AlphaEngine",
        update_title=None,
    )

    # ── Dark theme base styles ────────────────────────────────────────────────
    DARK_BG    = "#1e1e2e"
    CARD_BG    = "#313244"
    TEXT_COLOR = "#cdd6f4"
    BORDER     = "1px solid #45475a"

    card = {"backgroundColor": CARD_BG, "padding": "12px", "borderRadius": "8px",
            "marginBottom": "12px", "border": BORDER}

    app.layout = html.Div(
        style={"backgroundColor": DARK_BG, "color": TEXT_COLOR,
               "fontFamily": "monospace", "padding": "16px", "minHeight": "100vh"},
        children=[
            # ── Header ───────────────────────────────────────────────────────
            html.H2("AlphaEngine Dashboard",
                    style={"color": "#cba6f7", "marginBottom": "16px"}),

            # ── Status bar ───────────────────────────────────────────────────
            html.Div(style={**card, "display": "flex", "gap": "20px", "alignItems": "center"},
                     children=[
                         html.Span(id="status-tick",   style={"fontSize": "1.1rem", "fontWeight": "bold"}),
                         html.Span(id="status-regime", style={"padding": "4px 10px"}),
                         html.Span(id="status-kill",   style={"padding": "4px 10px"}),
                         html.Span(id="status-pnl",    style={"marginLeft": "auto", "fontSize": "1rem"}),
                     ]),

            # ── Price charts (2 columns) ──────────────────────────────────────
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
                     children=[
                         html.Div(style=card, children=[dcc.Graph(id="price-chart-crzy")]),
                         html.Div(style=card, children=[dcc.Graph(id="price-chart-tame")]),
                     ]),

            # ── Z-score chart ─────────────────────────────────────────────────
            html.Div(style=card, children=[dcc.Graph(id="zscore-chart")]),

            # ── PnL chart ─────────────────────────────────────────────────────
            html.Div(style=card, children=[dcc.Graph(id="pnl-chart")]),

            # ── Positions table ───────────────────────────────────────────────
            html.Div(style=card, children=[
                html.H4("Positions (algo trades only)",
                        style={"color": "#89dceb", "marginBottom": "8px"}),
                dash_table.DataTable(
                    id="positions-table",
                    columns=[
                        {"name": c, "id": c}
                        for c in ["Ticker", "Position", "Avg Cost", "Realized", "Unrealized"]
                    ],
                    data=[],
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#45475a", "color": TEXT_COLOR,
                        "fontWeight": "bold", "border": BORDER,
                    },
                    style_data={
                        "backgroundColor": CARD_BG, "color": TEXT_COLOR, "border": BORDER,
                    },
                    style_data_conditional=[
                        {
                            "if": {"filter_query": "{Position} > 0"},
                            "backgroundColor": "#1e3a2a",
                        },
                        {
                            "if": {"filter_query": "{Position} < 0"},
                            "backgroundColor": "#3a1e1e",
                        },
                    ],
                ),
            ]),

            # ── Update interval ───────────────────────────────────────────────
            dcc.Interval(
                id="interval",
                interval=config.DASHBOARD_UPDATE_MS,
                n_intervals=0,
            ),
        ],
    )

    register_callbacks(app, state)
    return app
