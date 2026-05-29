# =============================================================
# bot.py — AI Trading PRO+ Auto Bot
# Uses config.py for keys — safe to upload to GitHub
# =============================================================

from kiteconnect import KiteConnect
import yfinance as yf
import time

try:
    from config import API_KEY, ACCESS_TOKEN
except ImportError:
    print("Error: config.py not found. Copy config.py template and fill your keys.")
    exit(1)

if not ACCESS_TOKEN:
    print("Error: ACCESS_TOKEN empty. Run kite_token.py first.")
    exit(1)

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# Settings
STOCK     = "RELIANCE.NS"
CAPITAL   = 10000
RISK_PCT  = 1.5

in_trade  = False
entry_px  = 0
qty       = 0
trail_sl  = 0

print(f"Bot started | Stock: {STOCK} | Capital: Rs.{CAPITAL}")

while True:
    try:
        df = yf.Ticker(STOCK).history(period="5d", interval="5m")
        if df.empty:
            time.sleep(60); continue

        price = float(df["Close"].iloc[-1])
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()
        ema20 = float(df["EMA20"].iloc[-1])
        ema50 = float(df["EMA50"].iloc[-1])

        # EMA crossover
        trend = price > ema20 > ema50

        if not in_trade and trend:
            risk_amt = CAPITAL * (RISK_PCT / 100)
            qty      = max(1, int(risk_amt / (price * 0.02)))
            kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NSE,
                tradingsymbol=STOCK.replace(".NS",""),
                transaction_type=kite.TRANSACTION_TYPE_BUY,
                quantity=qty,
                order_type=kite.ORDER_TYPE_MARKET,
                product=kite.PRODUCT_CNC
            )
            entry_px = price
            trail_sl = price * 0.98
            in_trade = True
            print(f"BUY @ Rs.{price:.2f} | Qty: {qty}")

        if in_trade:
            if price > entry_px:
                trail_sl = max(trail_sl, price * 0.98)
            if price <= trail_sl:
                kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=kite.EXCHANGE_NSE,
                    tradingsymbol=STOCK.replace(".NS",""),
                    transaction_type=kite.TRANSACTION_TYPE_SELL,
                    quantity=qty,
                    order_type=kite.ORDER_TYPE_MARKET,
                    product=kite.PRODUCT_CNC
                )
                pnl = (price - entry_px) * qty
                print(f"EXIT @ Rs.{price:.2f} | P&L: Rs.{pnl:+.2f}")
                in_trade = False

        time.sleep(60)

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(60)
