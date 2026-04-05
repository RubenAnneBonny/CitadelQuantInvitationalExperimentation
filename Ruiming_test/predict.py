import pandas as pd
import numpy as np
from collect import collect_data

def predict(df, ticker="CRZY"):
    # Filter to one ticker
    data = df[df["ticker"] == ticker].copy()
    data = data.sort_values("tick").reset_index(drop=True)

    prices = data["last"].values

    # --- Feature Engineering ---
    data["ma_5"]       = data["last"].rolling(5).mean()
    data["ma_20"]      = data["last"].rolling(20).mean()
    data["spread"]     = data["ask"] - data["bid"]
    data["mid"]        = (data["ask"] + data["bid"]) / 2
    data["momentum"]   = data["last"].diff(5)          # price change over 5 ticks
    data["volatility"] = data["last"].rolling(10).std() # rolling std dev

    # Drop rows with NaN from rolling calculations
    data = data.dropna().reset_index(drop=True)

    # --- Simple Linear Regression Forecast ---
    # Use last 20 ticks to project next price
    recent = data.tail(20)
    x = np.arange(len(recent))
    y = recent["last"].values

    slope, intercept = np.polyfit(x, y, 1)
    next_price = intercept + slope * len(recent)

    # --- Signal ---
    current_price = data["last"].iloc[-1]
    ma_5          = data["ma_5"].iloc[-1]
    ma_20         = data["ma_20"].iloc[-1]
    momentum      = data["momentum"].iloc[-1]

    print(f"\n=== {ticker} PREDICTION ===")
    print(f"Current price:    {current_price:.2f}")
    print(f"Predicted next:   {next_price:.2f}")
    print(f"MA5:              {ma_5:.2f}")
    print(f"MA20:             {ma_20:.2f}")
    print(f"Momentum:         {momentum:.4f}")
    print(f"Volatility:       {data['volatility'].iloc[-1]:.4f}")

    if next_price > current_price and ma_5 > ma_20:
        signal = "BUY"
    elif next_price < current_price and ma_5 < ma_20:
        signal = "SELL"
    else:
        signal = "HOLD"

    print(f"Signal:           {signal}")

    return {
        "ticker":        ticker,
        "current_price": current_price,
        "predicted":     next_price,
        "signal":        signal,
        "volatility":    data["volatility"].iloc[-1],
        "momentum":      momentum,
    }


if __name__ == "__main__":
    # Collect data first, then predict
    df = collect_data(api_key="IRJW2AFJ", collect_until_tick=100)

    crzy_prediction = predict(df, ticker="CRZY")
    tame_prediction = predict(df, ticker="TAME")

    print("\n=== FINAL SIGNALS ===")
    print(f"CRZY → {crzy_prediction['signal']} @ predicted {crzy_prediction['predicted']:.2f}")
    print(f"TAME → {tame_prediction['signal']} @ predicted {tame_prediction['predicted']:.2f}")