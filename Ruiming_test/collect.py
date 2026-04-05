import time
import pandas as pd
from ritc import RIT

def collect_data(api_key, collect_until_tick=100):
    rit = RIT(api_key)
    records = []
    last_tick_recorded = -1
    case_started = False

    print("Waiting for case to become ACTIVE...")

    while True:
        case = rit.get_case()

        if case.status != "ACTIVE":
            if case_started:
                print("Case ended before target tick. Stopping.")
                break
            time.sleep(0.5)
            continue

        case_started = True

        if case.tick == last_tick_recorded:
            time.sleep(0.2)
            continue

        last_tick_recorded = case.tick

        if case.tick > collect_until_tick:
            print(f"Reached tick {collect_until_tick}. Collection done.")
            break

        for s in rit.get_securities():
            records.append({
                "tick":       case.tick,
                "ticker":     s.ticker,
                "last":       s.last,
                "bid":        s.bid,
                "ask":        s.ask,
                "bid_size":   s.bid_size,
                "ask_size":   s.ask_size,
                "volume":     s.volume,
                "position":   s.position,
                "unrealized": s.unrealized,
            })

        print(f"Tick {case.tick}/{collect_until_tick} captured")

    df = pd.DataFrame(records)
    df.to_csv("rit_first100.csv", index=False)
    print(f"Saved {len(df)} rows to rit_first100.csv")
    
    return df  # <-- hands data to prediction file


if __name__ == "__main__":
    df = collect_data(api_key="IRJW2AFJ")
    print(df)