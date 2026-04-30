from kiteconnect import KiteConnect
import yfinance as yf
import pandas as pd
import numpy as np
import time
from config import API_KEY, ACCESS_TOKEN

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

stock = "RELIANCE.NS"

capital = 10000
risk_pct = 1.5

in_trade = False
entry_price = 0
qty = 0
trail_sl = 0

while True:
    try:
        df = yf.Ticker(stock).history(period="5d", interval="5m")

        if df.empty:
            time.sleep(60)
            continue

        price = df["Close"].iloc[-1]

        # INDICATORS
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()

        ema20 = df["EMA20"].iloc[-1]
        ema50 = df["EMA50"].iloc[-1]

        trend = price > ema20 > ema50

        # ENTRY
        if not in_trade and trend:

            risk_amt = capital * (risk_pct / 100)
            qty = max(1, int(risk_amt / (price * 0.02)))

            kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NSE,
                tradingsymbol=stock.replace(".NS",""),
                transaction_type=kite.TRANSACTION_TYPE_BUY,
                quantity=qty,
                order_type=kite.ORDER_TYPE_MARKET,
                product=kite.PRODUCT_CNC
            )

            entry_price = price
            trail_sl = price * 0.98
            in_trade = True

            print(f"BUY @ {price}")

        # EXIT
        if in_trade:

            if price > entry_price:
                trail_sl = max(trail_sl, price * 0.98)

            if price <= trail_sl:

                kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=kite.EXCHANGE_NSE,
                    tradingsymbol=stock.replace(".NS",""),
                    transaction_type=kite.TRANSACTION_TYPE_SELL,
                    quantity=qty,
                    order_type=kite.ORDER_TYPE_MARKET,
                    product=kite.PRODUCT_CNC
                )

                print(f"EXIT @ {price}")
                in_trade = False

        time.sleep(60)

    except Exception as e:
        print("Error:", e)
        time.sleep(60)