import numpy as np
import pandas as pd
import matplotlib as plt

class Data_transformer:
    def __init__(self, df):
        self.df = df.copy()

    def transform(self):
        def features_per_ticker(g):
            g = g.sort_values('Date').copy()
            O = g['Open']
            H = g['High']
            L = g['Low']
            C = g['Close']
            V = g['Volume']

            # ── Returns ──
            g['rt']    = np.log(C / C.shift(1))
            g['rt_3']  = np.log(C / C.shift(3))
            g['rt_5']  = np.log(C / C.shift(5))
            g['rt_10'] = np.log(C / C.shift(10))
            g['rt_20'] = np.log(C / C.shift(20))

            # ── Volatility ──
            g['sigma_5']  = g['rt'].rolling(5).std()
            g['sigma_10'] = g['rt'].rolling(10).std()
            g['sigma_20'] = g['rt'].rolling(20).std()
            g['RV']       = np.log(H / L) ** 2
            g['rho']      = g['sigma_5'] / g['sigma_20']

            # ── Volume ──
            V_mean20        = V.rolling(20).mean()
            V_std20         = V.rolling(20).std()
            g['vz']         = (V - V_mean20) / V_std20
            g['vwap']       = (C * V).rolling(20).sum() / V.rolling(20).sum()
            g['delta_vwap'] = (C - g['vwap']) / C

            # ── Microstructure ──
            g['gt'] = np.log(O / C.shift(1))
            g['it'] = np.log(C / O)
            g['ht'] = np.log(H / L)

            # ── Target ──
            g['Y'] = g['rt'].shift(-2)

            return g

        self.df = self.df.groupby('Ticker', group_keys=False).apply(features_per_ticker)
        self.df = self.df.dropna()
        return self.df

    def get_X(self):
        feature_cols = ['rt','rt_3','rt_5','rt_10','rt_20',
                        'sigma_5','sigma_10','sigma_20','RV','rho',
                        'vz','delta_vwap',
                        'gt','it','ht']
        return self.df[feature_cols]

    def get_Y(self):
        return self.df['Y']
    
    def get_time_since_last_market_day(self):
        dates = pd.to_datetime(self.df.index.get_level_values('Date').unique()).sort_values()
        gaps = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        gaps.insert(0, 1)  # first day has no previous, default to 1
        return gaps