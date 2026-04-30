# =========================
# AI Trading PRO+ — FIXED v1.1
# Patches applied:
#   FIX 1 — Price chart + RSI chart + MACD chart (Plotly)
#   FIX 2 — Paper trade log records both BUY and SELL correctly
#   FIX 3 — Position sizing (Qty, SL, Target, Risk ₹) shown below signal
# =========================

from kiteconnect import KiteConnect
import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import json
import os
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

# ── Plotly for FIX 1 ─────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

from config import API_KEY, API_SECRET

kite = KiteConnect(api_key=API_KEY)

st.set_page_config(page_title="AI Trading PRO+", layout="wide")

# =========================
# 🔐 KITE LOGIN
# =========================
st.sidebar.subheader("🔐 Kite Login")
st.sidebar.markdown(f"[👉 Login to Kite]({kite.login_url()})")

if "access_token" not in st.session_state:
    token = st.sidebar.text_input("Paste Request Token")
    if st.sidebar.button("Connect Kite"):
        try:
            data = kite.generate_session(token, api_secret=API_SECRET)
            st.session_state.access_token = data["access_token"]
            kite.set_access_token(data["access_token"])
            st.success("Kite Connected ✅")
        except Exception as e:
            st.error(e)
else:
    kite.set_access_token(st.session_state.access_token)

if "access_token" in st.session_state:
    st.sidebar.success("Kite Connected ✅")
else:
    st.sidebar.error("Kite Not Connected ❌")

# =========================
# ⚙️ AUTO TRADING
# =========================
st.sidebar.subheader("⚙️ Auto Trading")
auto_trade = st.sidebar.toggle("Enable Auto Trading", value=False)
interval = st.sidebar.number_input("Run every sec", 10, 120, 15)

if "last_trade" not in st.session_state:
    st.session_state.last_trade = None

if "trade_log" not in st.session_state:
    st.session_state.trade_log = []

# ── FIX 2: paper state ──────────────────────────────────────
if "paper_position" not in st.session_state:
    st.session_state.paper_position = None

if "paper_balance" not in st.session_state:
    st.session_state.paper_balance = 100000.0

if "pnl_history" not in st.session_state:
    st.session_state.pnl_history = []

def can_trade():
    if st.session_state.last_trade is None:
        return True
    return (datetime.datetime.now() - st.session_state.last_trade).seconds > interval

def kite_ok():
    try:
        kite.profile()
        return True
    except:
        return False

# =========================
# LOGIN / SIGNUP SYSTEM
# =========================
DB = "users.json"
if not os.path.exists(DB):
    json.dump({}, open(DB, "w"))
try:
    users = json.load(open(DB))
except:
    users = {}

if "admin" not in users:
    users["admin"] = {
        "password": "admin123",
        "role": "admin",
        "status": "active",
        "expiry": "2099-12-31"
    }
    json.dump(users, open(DB, "w"))

if "user" not in st.session_state:
    st.session_state.user = None

if not st.session_state.user:
    st.title("🔐 Login / Signup")
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u in users and users[u]["password"] == p:
                if users[u].get("status") != "active":
                    st.error("❌ Not Approved by Admin")
                    st.stop()
                exp = users[u].get("expiry", "2000-01-01")
                if datetime.date.today() > datetime.datetime.strptime(exp, "%Y-%m-%d").date():
                    st.error("❌ Subscription Expired")
                    st.stop()
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid Login")

    with tab2:
        new_u = st.text_input("New Username")
        new_p = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            if new_u in users:
                st.warning("User already exists")
            else:
                users[new_u] = {
                    "password": new_p,
                    "role": "user",
                    "status": "pending",
                    "expiry": "2000-01-01"
                }
                json.dump(users, open(DB, "w"))
                st.success("✅ Account Created (Wait for Admin Approval)")
    st.stop()

user = st.session_state.user

st.sidebar.success(f"👤 {user}")
if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.session_state.admin = False
    st.rerun()

# =========================
# ADMIN PANEL
# =========================
role = users[user].get("role", "user")

if role == "admin":
    st.sidebar.success("👑 Admin")
    if st.sidebar.button("Open Admin Panel"):
        st.session_state.admin = True

if st.session_state.get("admin"):
    st.title("🛠 ADMIN PANEL")
    pending = [u for u in users if users[u].get("status") == "pending"]
    active  = [u for u in users if users[u].get("status") == "active"]

    st.subheader("⏳ Pending Users")
    for u in pending:
        col1, col2 = st.columns(2)
        if col1.button(f"Approve {u}", key=f"approve_{u}"):
            users[u]["status"] = "active"
            users[u]["expiry"] = str(datetime.date.today() + datetime.timedelta(days=30))
            json.dump(users, open(DB, "w"))
            st.success(f"{u} approved")
            st.rerun()
        if col2.button(f"Reject {u}", key=f"reject_{u}"):
            del users[u]
            json.dump(users, open(DB, "w"))
            st.warning(f"{u} removed")
            st.rerun()

    st.subheader("✅ Active Users")
    for u in active:
        if u == "admin":
            continue
        st.write(f"{u} | Expiry: {users[u]['expiry']}")
        col1, col2 = st.columns(2)
        if col1.button(f"Extend {u}", key=f"extend_{u}"):
            # FIX: extend from max(expiry, today) so we don't lose remaining days
            cur = datetime.datetime.strptime(users[u]["expiry"], "%Y-%m-%d").date()
            new_exp = max(cur, datetime.date.today()) + datetime.timedelta(days=30)
            users[u]["expiry"] = str(new_exp)
            json.dump(users, open(DB, "w"))
            st.success("Extended")
            st.rerun()
        if col2.button(f"Delete {u}", key=f"delete_{u}"):
            del users[u]
            json.dump(users, open(DB, "w"))
            st.error("Deleted")
            st.rerun()
    st.stop()

# =========================
# ACCESS CONTROL
# =========================
if users[user]["status"] != "active":
    st.error("❌ Access Denied (Wait for Admin Approval)")
    st.stop()

# =========================
# DASHBOARD
# =========================
st.title("📊 AI Trading PRO Dashboard")

mode    = st.radio("Mode", ["Paper", "Live"])
capital = st.number_input("Capital (₹)", 10000)
risk    = st.number_input("Risk %", 1.5)

if mode == "Live":
    st.warning("⚠️ LIVE TRADING ENABLED (REAL MONEY)")

# ── FIX 2: Show paper balance when in paper mode ─────────────
if mode == "Paper":
    pnl_total = sum(x["pnl"] for x in st.session_state.pnl_history)
    wins = sum(1 for x in st.session_state.pnl_history if x["pnl"] > 0)
    n = len(st.session_state.pnl_history)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Paper Balance", f"₹{st.session_state.paper_balance:,.0f}")
    c2.metric("📈 Total P&L",     f"₹{pnl_total:+,.2f}")
    c3.metric("🏆 Win Rate",      f"{wins/n*100:.1f}%" if n > 0 else "—")
    c4.metric("📊 Closed Trades", n)
    st.markdown("---")

# =========================
# STOCK LIST
# =========================
stocks = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS"]

# =========================
# SCANNER
# =========================
scan = []
for s in stocks:
    try:
        d = yf.Ticker(s).history(period="5d", interval="15m")
        if d.empty:
            continue
        d["EMA20"] = d["Close"].ewm(span=20).mean()
        d["EMA50"] = d["Close"].ewm(span=50).mean()
        last = d.iloc[-1]
        score = 2 if last["Close"] > last["EMA20"] > last["EMA50"] else 0
        scan.append({"Stock": s, "Score": score, "Price": round(last["Close"], 2)})
    except:
        pass

df_scan = pd.DataFrame(scan)
st.dataframe(df_scan)

best = stocks[0]
if not df_scan.empty:
    best = df_scan.sort_values("Score", ascending=False).iloc[0]["Stock"]

stock = st.selectbox("Stock", stocks, index=stocks.index(best))

# =========================
# DATA + INDICATORS
# =========================
df = yf.Ticker(stock).history(period="5d", interval="5m")

if df is None or df.empty:
    st.error("No Market Data")
    st.stop()

price = float(df["Close"].iloc[-1])

df["EMA20"] = df["Close"].ewm(span=20).mean()
df["EMA50"] = df["Close"].ewm(span=50).mean()

delta = df["Close"].diff()
gain  = (delta.where(delta > 0, 0)).rolling(14).mean()
loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs    = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

df["EMA12"]      = df["Close"].ewm(span=12).mean()
df["EMA26"]      = df["Close"].ewm(span=26).mean()
df["MACD"]       = df["EMA12"] - df["EMA26"]
df["MACD_Signal"]= df["MACD"].ewm(span=9).mean()
df["MACD_Hist"]  = df["MACD"] - df["MACD_Signal"]

df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()

df["BB_Mid"]   = df["Close"].rolling(20).mean()
bb_std         = df["Close"].rolling(20).std()
df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std

# =========================
# AI MODEL
# =========================
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

features = df[["EMA20","EMA50","RSI","MACD"]].dropna()
target   = df["Target"].loc[features.index]

model = RandomForestClassifier(n_estimators=200)
model.fit(features, target)
pred = model.predict(features.iloc[-1:].values)[0]

# =========================
# SIGNAL ENGINE
# =========================
force_trade = st.checkbox("🔥 FORCE TRADE (TEST MODE)", value=False)

trend   = price > df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]
rsi_ok  = df["RSI"].iloc[-1] > 50
macd_ok = df["MACD"].iloc[-1] > 0
score   = sum([trend, rsi_ok, macd_ok, pred])
signal  = True if force_trade else score >= 3

if signal:
    st.success(f"🟢 BUY SIGNAL | Score {score}/4")
else:
    st.warning(f"🟡 NO TRADE | Score {score}/4")

# =========================
# ── FIX 3: POSITION SIZING (shown immediately below signal) ──
# =========================
atr          = float(df["ATR"].iloc[-1])
risk_amount  = capital * (risk / 100)
sl_dist      = atr * 1.5
qty          = max(1, int(risk_amount / max(sl_dist, 0.01)))
stop_loss    = round(price - sl_dist, 2)
target_price = round(price + sl_dist * 2, 2)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("📦 Qty",       f"{qty} shares")
col_b.metric("🛡 Stop Loss",  f"₹{stop_loss:,.2f}")
col_c.metric("🎯 Target",     f"₹{target_price:,.2f}")
col_d.metric("💸 Risk ₹",     f"₹{risk_amount:,.0f}")

st.caption(f"Entry: ₹{price:.2f}  |  ATR: ₹{atr:.2f}  |  R:R = 2.0 : 1  |  SL dist: ₹{sl_dist:.2f}")

# =========================
# ── FIX 2: TRADE ENGINE (paper trade log fixed) ──
# =========================
def log_trade(action, stk, px, q, md, pnl=None):
    st.session_state.trade_log.append({
        "time":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stock":      stk,
        "action":     action,
        "price":      round(px, 2),
        "qty":        q,
        "mode":       md,
        "stop_loss":  stop_loss,
        "target":     target_price,
        # FIX: was `if pnl` which silently drops pnl=0.0 — use is not None
        "pnl":        round(pnl, 2) if pnl is not None else "—",
    })

def order(txn):
    sym = stock.replace(".NS", "")

    # ── PAPER MODE ──────────────────────────────────────────
    if mode == "Paper":
        live_px = float(yf.Ticker(stock).history(period="1d", interval="1m")["Close"].iloc[-1])

        if txn == "BUY":
            if st.session_state.paper_position:
                st.warning(f"⚠️ Already holding {st.session_state.paper_position['stock']} — sell first")
                return
            st.session_state.paper_position = {
                "stock": stock, "price": live_px, "qty": qty,
                "stop_loss": stop_loss, "target": target_price,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.paper_balance -= live_px * qty
            # ✅ FIX: was missing log call entirely for paper BUY
            log_trade("BUY", stock, live_px, qty, "Paper", pnl=None)
            st.success(f"📄 Paper BUY | {sym} | ₹{live_px:.2f} × {qty} | SL ₹{stop_loss} | TGT ₹{target_price}")
            return

        if txn == "SELL":
            if not st.session_state.paper_position:
                st.warning("⚠️ No open paper position to sell")
                return
            pos = st.session_state.paper_position
            pnl = (live_px - pos["price"]) * pos["qty"]
            st.session_state.paper_balance += live_px * pos["qty"]
            # ✅ FIX: pnl now recorded in both pnl_history and trade_log
            st.session_state.pnl_history.append({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stock": stock,
                "pnl": round(pnl, 2)
            })
            log_trade("SELL", stock, live_px, pos["qty"], "Paper", pnl=pnl)
            st.session_state.paper_position = None
            emoji = "🟢" if pnl >= 0 else "🔴"
            st.success(f"📄 Paper SELL | {sym} | ₹{live_px:.2f} | P&L: {emoji} ₹{pnl:+.2f}")
            return

    # ── LIVE MODE ────────────────────────────────────────────
    if not kite_ok():
        st.error("❌ Kite Not Connected")
        return
    try:
        kite.place_order(
            variety="regular",
            exchange="NSE",
            tradingsymbol=sym,
            transaction_type=txn,
            quantity=qty,
            order_type="MARKET",
            product="CNC"
        )
        log_trade(txn, stock, price, qty, "Live", pnl=None)
        st.success(f"✅ Live {txn} | {sym} × {qty}")
    except Exception as e:
        st.error(e)

# =========================
# 🔘 MANUAL BUTTONS
# =========================
col1, col2 = st.columns(2)
if col1.button("🚀 BUY NOW"):
    order("BUY")
    st.session_state.last_trade = datetime.datetime.now()

if col2.button("🛑 SELL NOW"):
    order("SELL")
    st.session_state.last_trade = datetime.datetime.now()

# ── Open paper position tracker ───────────────────────────────
if mode == "Paper" and st.session_state.paper_position:
    pos = st.session_state.paper_position
    open_pnl = (price - pos["price"]) * pos["qty"]
    color = "#2d8a4e" if open_pnl >= 0 else "#c0392b"
    st.markdown(f"""
    <div style="background:#f0fff4;border:1px solid #2d8a4e;border-radius:6px;
                padding:10px 16px;margin:8px 0;">
        <b>📦 Open Paper Position</b><br>
        {pos['stock']} &nbsp;|&nbsp; Entry: ₹{pos['price']:.2f}
        &nbsp;|&nbsp; Qty: {pos['qty']}
        &nbsp;|&nbsp; SL: ₹{pos['stop_loss']}
        &nbsp;|&nbsp; TGT: ₹{pos['target']}<br>
        Unrealised P&amp;L: <b style="color:{color}">₹{open_pnl:+.2f}</b>
    </div>
    """, unsafe_allow_html=True)

# =========================
# AUTO LOOP
# =========================
if auto_trade:
    if mode == "Live" and kite_ok() and signal and can_trade():
        order("BUY")
        st.session_state.last_trade = datetime.datetime.now()
    time.sleep(interval)
    st.rerun()

# =========================
# ── FIX 1: CHARTS ────────
# =========================
st.markdown("---")
chart_df = df.tail(100).copy()
chart_df.index = pd.to_datetime(chart_df.index)

if PLOTLY_OK:

    # ── Chart 1: Candlestick + EMA20 + EMA50 + Bollinger Bands ──
    st.subheader("📉 Price Chart")
    fig1 = go.Figure()

    fig1.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"],   close=chart_df["Close"],
        name="Price",
        increasing_line_color="#27ae60",
        decreasing_line_color="#e74c3c",
        increasing_fillcolor="#d5f5e3",
        decreasing_fillcolor="#fadbd8",
    ))
    fig1.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["EMA20"],
        line=dict(color="#2980b9", width=1.5), name="EMA 20"
    ))
    fig1.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["EMA50"],
        line=dict(color="#8e44ad", width=1.5, dash="dot"), name="EMA 50"
    ))
    fig1.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["BB_Upper"],
        line=dict(color="#95a5a6", width=1, dash="dash"), name="BB Upper"
    ))
    fig1.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["BB_Lower"],
        line=dict(color="#95a5a6", width=1, dash="dash"), name="BB Lower",
        fill="tonexty", fillcolor="rgba(149,165,166,0.07)"
    ))

    # Stop loss & target reference lines
    fig1.add_hline(
        y=stop_loss,
        line_color="#e74c3c", line_dash="dot", line_width=1.5,
        annotation_text=f"SL  ₹{stop_loss}",
        annotation_position="bottom right",
        annotation_font_color="#e74c3c"
    )
    fig1.add_hline(
        y=target_price,
        line_color="#27ae60", line_dash="dot", line_width=1.5,
        annotation_text=f"TGT  ₹{target_price}",
        annotation_position="top right",
        annotation_font_color="#27ae60"
    )
    fig1.add_hline(
        y=price,
        line_color="#f39c12", line_dash="solid", line_width=1,
        annotation_text=f"LTP  ₹{price:.2f}",
        annotation_position="top left",
        annotation_font_color="#f39c12"
    )

    fig1.update_layout(
        height=380,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)", font_size=11),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showgrid=True, gridcolor="#ecf0f1"),
        yaxis=dict(showgrid=True, gridcolor="#ecf0f1"),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: RSI with overbought / oversold bands ──────────
    st.subheader("📊 RSI (14)")
    fig2 = go.Figure()

    fig2.add_hrect(y0=70, y1=100, fillcolor="rgba(231,76,60,0.08)",  line_width=0)
    fig2.add_hrect(y0=0,  y1=30,  fillcolor="rgba(39,174,96,0.08)",  line_width=0)
    fig2.add_hline(y=70, line_color="#e74c3c", line_dash="dot", line_width=1,
                   annotation_text="Overbought 70", annotation_position="bottom right")
    fig2.add_hline(y=30, line_color="#27ae60", line_dash="dot", line_width=1,
                   annotation_text="Oversold 30", annotation_position="top right")
    fig2.add_hline(y=50, line_color="#95a5a6", line_dash="dot", line_width=1)

    rsi_series = chart_df["RSI"]
    rsi_colors = [
        "#e74c3c" if v > 70 else ("#27ae60" if v < 30 else "#2980b9")
        for v in rsi_series
    ]
    fig2.add_trace(go.Scatter(
        x=chart_df.index, y=rsi_series,
        line=dict(color="#2980b9", width=2), name="RSI",
        fill="tozeroy", fillcolor="rgba(41,128,185,0.07)"
    ))
    # Colour dots at overbought / oversold
    fig2.add_trace(go.Scatter(
        x=chart_df.index, y=rsi_series,
        mode="markers",
        marker=dict(color=rsi_colors, size=3),
        name="RSI level", showlegend=False
    ))
    # Current RSI annotation
    cur_rsi = rsi_series.iloc[-1]
    fig2.add_annotation(
        x=chart_df.index[-1], y=cur_rsi,
        text=f"  {cur_rsi:.1f}",
        showarrow=False, font=dict(color="#2980b9", size=12, family="monospace")
    )

    fig2.update_layout(
        height=220,
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor="#ecf0f1"),
        xaxis=dict(showgrid=True, gridcolor="#ecf0f1"),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: MACD histogram + lines ───────────────────────
    st.subheader("📈 MACD")
    fig3 = go.Figure()

    hist_colors = [
        "#27ae60" if v >= 0 else "#e74c3c"
        for v in chart_df["MACD_Hist"]
    ]
    fig3.add_trace(go.Bar(
        x=chart_df.index, y=chart_df["MACD_Hist"],
        marker_color=hist_colors, name="Histogram", opacity=0.65
    ))
    fig3.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["MACD"],
        line=dict(color="#2980b9", width=1.5), name="MACD"
    ))
    fig3.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["MACD_Signal"],
        line=dict(color="#f39c12", width=1.5, dash="dot"), name="Signal"
    ))
    fig3.add_hline(y=0, line_color="#bdc3c7", line_width=1)

    fig3.update_layout(
        height=220,
        xaxis=dict(showgrid=True, gridcolor="#ecf0f1"),
        yaxis=dict(showgrid=True, gridcolor="#ecf0f1"),
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", font_size=11),
        margin=dict(l=0, r=0, t=10, b=0),
        barmode="relative",
    )
    st.plotly_chart(fig3, use_container_width=True)

else:
    # ── Fallback: plain Streamlit charts if plotly not installed ──
    st.warning("💡 Install plotly for rich charts:  pip install plotly")
    st.subheader("📉 Price + EMA + Bollinger Bands")
    st.line_chart(chart_df[["Close","EMA20","EMA50","BB_Upper","BB_Lower"]])
    st.subheader("📊 RSI")
    st.line_chart(chart_df[["RSI"]])
    st.subheader("📈 MACD")
    st.line_chart(chart_df[["MACD","MACD_Signal"]])

# =========================
# TRADE LOGS
# =========================
st.markdown("---")
st.subheader("📋 Trade Logs")

if st.session_state.trade_log:
    log_df = pd.DataFrame(st.session_state.trade_log)
    st.dataframe(log_df, hide_index=True, use_container_width=True)

    # ── Cumulative P&L chart (paper mode) ────────────────────
    closed = [x for x in st.session_state.pnl_history if x.get("pnl") is not None]
    if len(closed) > 1:
        pnl_df = pd.DataFrame(closed)
        pnl_df["Cumulative P&L"] = pnl_df["pnl"].cumsum()
        st.subheader("📈 Cumulative P&L")
        st.line_chart(pnl_df.set_index("time")["Cumulative P&L"])

    if st.button("🗑 Clear Logs"):
        st.session_state.trade_log  = []
        st.session_state.pnl_history = []
        st.rerun()
else:
    st.info("No trades yet — place your first trade above")
