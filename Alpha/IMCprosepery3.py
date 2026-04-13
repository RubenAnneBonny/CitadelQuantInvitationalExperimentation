from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import numpy as np
import jsonpickle


class Trader:

    POSITION_LIMIT = 20

    def run(self, state: TradingState):
        result = {}

        # Load memory
        if state.traderData:
            memory = jsonpickle.decode(state.traderData)
        else:
            memory = {
                "prices": {},
                "spreads": []
            }

        # Store midprices
        for product in state.order_depths:
            od = state.order_depths[product]

            if len(od.buy_orders) == 0 or len(od.sell_orders) == 0:
                continue

            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            mid = (best_bid + best_ask) / 2

            if product not in memory["prices"]:
                memory["prices"][product] = []

            memory["prices"][product].append(mid)

            if len(memory["prices"][product]) > 50:
                memory["prices"][product].pop(0)

        ########################################
        # ALPHA STRATEGY
        ########################################
        for product in state.order_depths:

            od = state.order_depths[product]

            if len(od.buy_orders) == 0 or len(od.sell_orders) == 0:
                continue

            prices = memory["prices"][product]

            if len(prices) < 6:
                continue

            orders = []

            close = prices[-1]
            delta = prices[-1] - prices[-2]

            last5_delta = np.diff(prices[-6:])

            ts_min = min(last5_delta)
            ts_max = max(last5_delta)

            ### ALPHA #9
            if 0 < ts_min:
                alpha9 = delta
            elif ts_max < 0:
                alpha9 = delta
            else:
                alpha9 = -delta

            ### ALPHA #10
            alpha10 = alpha9  # simplified rank later

            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())

            pos = state.position.get(product, 0)

            ########################################
            # SIGNAL EXECUTION
            ########################################

            if alpha9 > 0 and alpha10 > 0:
                vol = min(5, self.POSITION_LIMIT - pos)
                if vol > 0:
                    orders.append(Order(product, best_ask, vol))

            elif alpha9 < 0 and alpha10 < 0:
                vol = min(5, self.POSITION_LIMIT + pos)
                if vol > 0:
                    orders.append(Order(product, best_bid, -vol))

            ########################################
            # MEAN REVERSION LAYER
            ########################################

            mean_price = np.mean(prices[-10:])

            if close < mean_price - 2:
                vol = min(3, self.POSITION_LIMIT - pos)
                if vol > 0:
                    orders.append(Order(product, best_ask, vol))

            elif close > mean_price + 2:
                vol = min(3, self.POSITION_LIMIT + pos)
                if vol > 0:
                    orders.append(Order(product, best_bid, -vol))

            result[product] = orders

        ########################################
        # PAIR TRADING EXAMPLE
        ########################################

        pairA = "AMETHYSTS"
        pairB = "STARFRUIT"

        if pairA in memory["prices"] and pairB in memory["prices"]:
            if len(memory["prices"][pairA]) > 10 and len(memory["prices"][pairB]) > 10:

                spread = memory["prices"][pairA][-1] - memory["prices"][pairB][-1]
                memory["spreads"].append(spread)

                if len(memory["spreads"]) > 50:
                    memory["spreads"].pop(0)

                spread_mean = np.mean(memory["spreads"])

                if spread > spread_mean + 3:

                    if pairA in state.order_depths:
                        best_bid = max(state.order_depths[pairA].buy_orders.keys())
                        result.setdefault(pairA, []).append(Order(pairA, best_bid, -3))

                    if pairB in state.order_depths:
                        best_ask = min(state.order_depths[pairB].sell_orders.keys())
                        result.setdefault(pairB, []).append(Order(pairB, best_ask, 3))

                elif spread < spread_mean - 3:

                    if pairA in state.order_depths:
                        best_ask = min(state.order_depths[pairA].sell_orders.keys())
                        result.setdefault(pairA, []).append(Order(pairA, best_ask, 3))

                    if pairB in state.order_depths:
                        best_bid = max(state.order_depths[pairB].buy_orders.keys())
                        result.setdefault(pairB, []).append(Order(pairB, best_bid, -3))

        traderData = jsonpickle.encode(memory)

        conversions = 0

        return result, conversions, traderData