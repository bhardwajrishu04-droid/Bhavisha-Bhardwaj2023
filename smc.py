
def detect_market_structure(df):
    highs = df["High"]
    lows = df["Low"]

    if highs.iloc[-1] > highs.iloc[-5] and lows.iloc[-1] > lows.iloc[-5]:
        return {"trend":"UPTREND"}

    if highs.iloc[-1] < highs.iloc[-5] and lows.iloc[-1] < lows.iloc[-5]:
        return {"trend":"DOWNTREND"}

    return {"trend":"RANGE"}
