# =============================================================
# AI Trading PRO+ v1.3 — COMPLETE MERGED VERSION
# Built on your v1.2 — all existing features preserved +
#   NEW: Top 20 NSE stocks across 6 sector universes
#   NEW: Trading Mode — Intraday / Swing / Futures / Options
#   NEW: Mode-specific timeframe, signal, position sizing
#   NEW: F&O lot size + margin + premium estimator
#   NEW: Intraday market hours enforcement
#   NEW: Options CE/PE strike calculator
#   NEW: 5-factor scanner with colour-coded signals
#   NEW: EMA9 + Stochastic + Volume ratio indicators
# =============================================================

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

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

from config import (
    API_KEY, API_SECRET,
    ALERT_ON_SIGNAL, ALERT_ON_EXECUTION,
    ALERT_MIN_SCORE, ALERT_COOLDOWN_MIN,
)

try:
    from alerts import send_alert
    ALERTS_AVAILABLE = True
except Exception as _ae:
    ALERTS_AVAILABLE = False
    _ALERT_IMPORT_ERROR = str(_ae)

kite = KiteConnect(api_key=API_KEY)
st.set_page_config(page_title="AI Trading PRO+ v1.3", layout="wide", page_icon="📈")

# =============================================================
# STOCK UNIVERSES — 6 SECTORS, 60+ STOCKS
# =============================================================
STOCK_UNIVERSE = {
    "⭐ Nifty 50 Top 20": [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS",
        "HINDUNILVR.NS","SBIN.NS","BAJFINANCE.NS","BHARTIARTL.NS","KOTAKBANK.NS",
        "WIPRO.NS","AXISBANK.NS","LTIM.NS","HCLTECH.NS","ASIANPAINT.NS",
        "MARUTI.NS","TITAN.NS","SUNPHARMA.NS","ULTRACEMCO.NS","NESTLEIND.NS"
    ],
    "🏦 Bank Nifty": [
        "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS",
        "BANKBARODA.NS","CANBK.NS","INDUSINDBK.NS","FEDERALBNK.NS","IDFCFIRSTB.NS"
    ],
    "💻 IT Sector": [
        "TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","LTIM.NS",
        "TECHM.NS","MPHASIS.NS","COFORGE.NS","PERSISTENT.NS","OFSS.NS"
    ],
    "🚗 Auto Sector": [
        "MARUTI.NS","TATAMOTORS.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","EICHERMOT.NS",
        "ASHOKLEY.NS","TVSMOTOR.NS","BALKRISIND.NS","BOSCHLTD.NS","MOTHERSON.NS"
    ],
    "🛒 FMCG Sector": [
        "HINDUNILVR.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS","MARICO.NS",
        "COLPAL.NS","GODREJCP.NS","ITC.NS","TATACONSUM.NS","EMAMILTD.NS"
    ],
    "💊 Pharma Sector": [
        "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","BIOCON.NS",
        "AUROPHARMA.NS","TORNTPHARM.NS","LUPIN.NS","IPCALAB.NS","ALKEM.NS"
    ],
}

# F&O Lot Sizes (NSE official)
FO_LOTS = {
    "RELIANCE.NS":250,"TCS.NS":150,"HDFCBANK.NS":550,"ICICIBANK.NS":700,
    "INFY.NS":300,"SBIN.NS":1500,"BAJFINANCE.NS":125,"BHARTIARTL.NS":950,
    "KOTAKBANK.NS":400,"WIPRO.NS":1500,"AXISBANK.NS":1200,"HCLTECH.NS":350,
    "HINDUNILVR.NS":300,"MARUTI.NS":100,"TITAN.NS":375,"SUNPHARMA.NS":700,
    "TATAMOTORS.NS":2850,"BANKBARODA.NS":5850,"LTIM.NS":75,"ASIANPAINT.NS":200,
    "NESTLEIND.NS":40,"ULTRACEMCO.NS":100,"INDUSINDBK.NS":900,"TECHM.NS":600,
    "DRREDDY.NS":125,"CIPLA.NS":650,"BAJAJ-AUTO.NS":75,"EICHERMOT.NS":200,
}

# Trading mode config
MODES = {
    "📈 Intraday": {
        "period":"1d","interval":"5m","sl_mult":1.0,"rr":1.5,
        "product":"MIS","desc":"Same-day exit before 3:15 PM","color":"#4e8fff",
    },
    "🌊 Swing": {
        "period":"1mo","interval":"1d","sl_mult":2.0,"rr":3.0,
        "product":"CNC","desc":"Hold 3–15 trading days","color":"#a78bfa",
    },
    "📊 Futures": {
        "period":"5d","interval":"15m","sl_mult":1.5,"rr":2.0,
        "product":"NRML","desc":"Monthly/weekly futures contract","color":"#ffa94d",
    },
    "🎯 Options": {
        "period":"5d","interval":"15m","sl_mult":1.5,"rr":2.5,
        "product":"NRML","desc":"CE/PE options — limited risk","color":"#00e5a0",
    },
}

# =============================================================
# SESSION STATE
# =============================================================
for _k, _v in {
    "access_token": None, "last_trade": None,
    "trade_log": [], "paper_position": None,
    "paper_balance": 100000.0, "pnl_history": [],
    "user": None, "admin": False,
    "last_alert_time": {}, "alert_log": [],
    "scan_results": [],
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# =============================================================
# ALERT HELPER — unchanged from your v1.2
# =============================================================
def fire_alert(action, stk, px, q, sl, tgt, sc, md, pnl=None):
    if not ALERTS_AVAILABLE: return
    if not (ALERT_ON_SIGNAL or ALERT_ON_EXECUTION): return
    if sc < ALERT_MIN_SCORE and pnl is None: return
    cooldown_key = f"{stk}_{action}"
    last_sent = st.session_state.last_alert_time.get(cooldown_key)
    if last_sent:
        if (datetime.datetime.now() - last_sent).seconds / 60 < ALERT_COOLDOWN_MIN:
            return
    try:
        results = send_alert(action=action, stock=stk, price=px, qty=q,
                             stop_loss=sl, target=tgt, score=sc, mode=md, pnl=pnl)
        st.session_state.last_alert_time[cooldown_key] = datetime.datetime.now()
        for r in results:
            st.session_state.alert_log.insert(0, {
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "stock": stk, "action": action, "result": r
            })
        st.session_state.alert_log = st.session_state.alert_log[:10]
        for r in results:
            st.toast(r, icon="✅" if "✅" in r else "⚠️")
    except Exception as e:
        st.toast(f"Alert error: {e}", icon="⚠️")

# =============================================================
# KITE SIDEBAR — unchanged from your v1.2
# =============================================================
st.sidebar.subheader("🔐 Kite Login")
st.sidebar.markdown(f"[👉 Login to Kite]({kite.login_url()})")

if not st.session_state.access_token:
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
    st.sidebar.success("Kite Connected ✅")

if not st.session_state.access_token:
    st.sidebar.error("Kite Not Connected ❌")

st.sidebar.subheader("⚙️ Auto Trading")
auto_trade = st.sidebar.toggle("Enable Auto Trading", value=False)
interval   = st.sidebar.number_input("Run every sec", 10, 120, 15)

# Alert Settings Panel — unchanged from your v1.2
st.sidebar.markdown("---")
st.sidebar.subheader("🔔 Alert Settings")

if not ALERTS_AVAILABLE:
    st.sidebar.error(f"alerts.py import failed:\n{_ALERT_IMPORT_ERROR}")
else:
    from config import (
        EMAIL_ALERTS_ON, CALLMEBOT_ALERTS_ON, TWILIO_ALERTS_ON,
        ALERT_EMAIL_TO, CALLMEBOT_PHONE,
    )
    if EMAIL_ALERTS_ON:
        st.sidebar.success(f"📧 Email → {ALERT_EMAIL_TO}")
    else:
        st.sidebar.info("📧 Email: OFF  (set EMAIL_ALERTS_ON=True in config.py)")
    if CALLMEBOT_ALERTS_ON:
        st.sidebar.success(f"📱 WhatsApp → {CALLMEBOT_PHONE}")
    else:
        st.sidebar.info("📱 WhatsApp: OFF  (set CALLMEBOT_ALERTS_ON=True in config.py)")
    if TWILIO_ALERTS_ON:
        st.sidebar.success("📱 Twilio WhatsApp: ON")
    if not any([EMAIL_ALERTS_ON, CALLMEBOT_ALERTS_ON, TWILIO_ALERTS_ON]):
        st.sidebar.warning("All alerts OFF — edit config.py to enable")
    if st.sidebar.button("🧪 Send Test Alert"):
        fire_alert("TEST SIGNAL","RELIANCE.NS",1427.50,32,1422.95,1437.05,4,"Paper")
    if st.session_state.alert_log:
        st.sidebar.markdown("**Recent alerts:**")
        for a in st.session_state.alert_log[:5]:
            icon = "✅" if "✅" in a["result"] else "❌"
            st.sidebar.caption(f"{icon} {a['time']} · {a['stock']} · {a['action']}")

# =============================================================
# LOGIN / SIGNUP — unchanged from your v1.2
# =============================================================
DB = "users.json"
if not os.path.exists(DB):
    json.dump({}, open(DB, "w"))
try:
    users = json.load(open(DB))
except Exception:
    users = {}

if "admin" not in users:
    users["admin"] = {"password":"admin123","role":"admin","status":"active","expiry":"2099-12-31"}
    json.dump(users, open(DB, "w"))

if not st.session_state.user:
    st.title("🔐 Login / Signup")
    tab1, tab2 = st.tabs(["Login", "Signup"])
    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u in users and users[u]["password"] == p:
                if users[u].get("status") != "active":
                    st.error("❌ Not Approved by Admin"); st.stop()
                exp = users[u].get("expiry", "2000-01-01")
                if datetime.date.today() > datetime.datetime.strptime(exp, "%Y-%m-%d").date():
                    st.error("❌ Subscription Expired"); st.stop()
                st.session_state.user = u; st.rerun()
            else:
                st.error("Invalid Login")
    with tab2:
        nu = st.text_input("New Username")
        np_ = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            if nu in users:
                st.warning("User already exists")
            else:
                users[nu] = {"password":np_,"role":"user","status":"pending","expiry":"2000-01-01"}
                json.dump(users, open(DB, "w"))
                st.success("✅ Account Created (Wait for Admin Approval)")
    st.stop()

user = st.session_state.user
st.sidebar.success(f"👤 {user}")
if st.sidebar.button("Logout"):
    st.session_state.user = None; st.session_state.admin = False; st.rerun()

# =============================================================
# ADMIN PANEL — unchanged from your v1.2
# =============================================================
role = users[user].get("role", "user")
if role == "admin":
    st.sidebar.success("👑 Admin")
    if st.sidebar.button("Open Admin Panel"):
        st.session_state.admin = True

if st.session_state.admin:
    st.title("🛠 ADMIN PANEL")
    pending = [u for u in users if users[u].get("status") == "pending"]
    active  = [u for u in users if users[u].get("status") == "active"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Users", len(users)-1)
    c2.metric("Active", len([u for u in active if u != "admin"]))
    c3.metric("Pending", len(pending))
    st.subheader("⏳ Pending Users")
    for u in pending:
        c1, c2 = st.columns(2)
        if c1.button(f"Approve {u}", key=f"ap_{u}"):
            users[u]["status"] = "active"
            users[u]["expiry"] = str(datetime.date.today() + datetime.timedelta(days=30))
            json.dump(users, open(DB, "w")); st.rerun()
        if c2.button(f"Reject {u}", key=f"rj_{u}"):
            del users[u]; json.dump(users, open(DB, "w")); st.rerun()
    st.subheader("✅ Active Users")
    for u in active:
        if u == "admin": continue
        st.write(f"{u} | Expiry: {users[u]['expiry']}")
        c1, c2 = st.columns(2)
        if c1.button(f"Extend {u}", key=f"ex_{u}"):
            cur = datetime.datetime.strptime(users[u]["expiry"], "%Y-%m-%d").date()
            users[u]["expiry"] = str(max(cur, datetime.date.today()) + datetime.timedelta(days=30))
            json.dump(users, open(DB, "w")); st.rerun()
        if c2.button(f"Delete {u}", key=f"dl_{u}"):
            del users[u]; json.dump(users, open(DB, "w")); st.rerun()
    st.stop()

if users[user]["status"] != "active":
    st.error("❌ Access Denied"); st.stop()

# =============================================================
# HELPER FUNCTIONS
# =============================================================
def can_trade():
    if st.session_state.last_trade is None: return True
    return (datetime.datetime.now() - st.session_state.last_trade).seconds > interval

def kite_ok():
    try: kite.profile(); return True
    except: return False

def compute_indicators(df):
    """Extended indicators — EMA9/20/50/200, RSI, MACD, ATR, BB, Stochastic, Volume"""
    df = df.copy()
    for span in [9, 20, 50, 200]:
        df[f"EMA{span}"] = df["Close"].ewm(span=span).mean()
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"]         = 100 - (100 / (1 + gain / loss))
    df["EMA12"]       = df["Close"].ewm(span=12).mean()
    df["EMA26"]       = df["Close"].ewm(span=26).mean()
    df["MACD"]        = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    df["ATR"]         = (df["High"] - df["Low"]).rolling(14).mean()
    df["BB_Mid"]      = df["Close"].rolling(20).mean()
    bb_std            = df["Close"].rolling(20).std()
    df["BB_Upper"]    = df["BB_Mid"] + 2 * bb_std
    df["BB_Lower"]    = df["BB_Mid"] - 2 * bb_std
    low14             = df["Low"].rolling(14).min()
    high14            = df["High"].rolling(14).max()
    df["Stoch_K"]     = 100 * (df["Close"] - low14) / (high14 - low14 + 1e-9)
    df["Vol_Ratio"]   = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    return df

# =============================================================
# MAIN DASHBOARD
# =============================================================
st.title("📊 AI Trading PRO+ v1.3")
st.caption(f"👤 {user}  |  {datetime.datetime.now().strftime('%d %b %Y  %H:%M:%S')}")

# ── TRADING MODE SELECTOR (NEW) ───────────────────────────────
st.markdown("### 🎯 Select Trading Mode")
selected_mode = st.radio(
    "Trading Mode", list(MODES.keys()),
    horizontal=True, label_visibility="collapsed"
)
mcfg = MODES[selected_mode]

# Mode banner
mode_icons = {"📈 Intraday":"🔵","🌊 Swing":"🟣","📊 Futures":"🟠","🎯 Options":"🟢"}
st.info(f"{mode_icons.get(selected_mode,'🔵')} **{selected_mode}** — {mcfg['desc']}  |  Kite Product: `{mcfg['product']}`")

# Intraday market hours enforcement (NEW)
if selected_mode == "📈 Intraday":
    now_t = datetime.datetime.now().time()
    if now_t < datetime.time(9, 15):
        st.warning("⏰ Market opens at 9:15 AM IST — come back then")
    elif now_t > datetime.time(14, 45):
        st.error("🔴 Intraday cutoff passed (2:45 PM) — NO new entries! Square off open positions before 3:15 PM.")
    else:
        dt_close  = datetime.datetime.combine(datetime.date.today(), datetime.time(15, 15))
        mins_left = int((dt_close - datetime.datetime.now()).seconds / 60)
        st.success(f"✅ Market open — {mins_left} minutes left for intraday trading")

st.markdown("---")

# ── SETTINGS PANEL ───────────────────────────────────────────
col_main, col_set = st.columns([3, 1])

with col_set:
    st.markdown("#### ⚙️ Settings")
    mode = st.radio("Order Mode", ["Paper", "Live"], horizontal=True)
    if mode == "Live":
        st.warning("⚠️ REAL MONEY")
    capital  = st.number_input("Capital (₹)", 10000, 10000000, 100000, step=5000)
    risk     = st.number_input("Risk %", 0.5, 5.0, 1.5, step=0.1)
    force_trade = st.checkbox("🔥 FORCE TRADE (TEST MODE)", value=False)

with col_main:
    # Paper balance metrics
    if mode == "Paper":
        pnl_total = sum(x["pnl"] for x in st.session_state.pnl_history)
        wins = sum(1 for x in st.session_state.pnl_history if x["pnl"] > 0)
        n    = len(st.session_state.pnl_history)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Paper Balance", f"₹{st.session_state.paper_balance:,.0f}")
        c2.metric("📈 Total P&L",     f"₹{pnl_total:+,.2f}")
        c3.metric("🏆 Win Rate",      f"{wins/n*100:.1f}%" if n > 0 else "—")
        c4.metric("📊 Closed Trades", n)
        st.markdown("---")

    # ── UNIVERSE SELECTOR (NEW) ───────────────────────────────
    st.markdown("#### 📋 Stock Universe")
    universe_name = st.selectbox("Select Universe", list(STOCK_UNIVERSE.keys()))
    stocks = STOCK_UNIVERSE[universe_name]

    # ── SCANNER (UPGRADED — 5 factors, colour signals) ────────
    st.markdown(f"#### 🔍 Scanner — {universe_name} ({len(stocks)} stocks)")
    col_scan1, col_scan2 = st.columns([2,3])
    scan_btn = col_scan1.button(f"🔍 Scan All {len(stocks)} Stocks", type="primary")
    col_scan2.caption("Click to scan all stocks with AI signals, RSI, MACD and Volume")

    # Auto-run on first load or when universe changes
    universe_key = f"last_universe_{universe_name}"
    if universe_key not in st.session_state:
        st.session_state[universe_key] = False
    auto_scan = not st.session_state[universe_key]

    if scan_btn or auto_scan:
        st.session_state[universe_key] = True
        with st.spinner(f"Scanning {len(stocks)} stocks for {selected_mode} signals..."):
            scan = []
            for s in stocks:
                try:
                    d = yf.Ticker(s).history(period=mcfg["period"], interval=mcfg["interval"])
                    if d is None or d.empty or len(d) < 20: continue
                    d = compute_indicators(d)
                    last = d.iloc[-1]
                    sc = 0
                    if last["Close"] > last["EMA20"] > last["EMA50"]: sc += 2
                    if 45 < last["RSI"] < 75: sc += 1
                    if last["MACD"] > last["MACD_Signal"]: sc += 1
                    if last["Vol_Ratio"] > 1.2: sc += 1
                    chg = (last["Close"] - d["Close"].iloc[-2]) / d["Close"].iloc[-2] * 100
                    scan.append({
                        "Stock":   s.replace(".NS",""),
                        "Price":   round(float(last["Close"]), 2),
                        "Chg %":   round(chg, 2),
                        "RSI":     round(float(last["RSI"]), 1),
                        "MACD":    round(float(last["MACD"]), 2),
                        "Vol":     round(float(last["Vol_Ratio"]), 2),
                        "Score":   f"{sc}/5",
                        "Signal":  "🟢 BUY" if sc >= 3 else ("🔴 SELL" if sc <= 1 else "🟡 HOLD"),
                        "_score":  sc,
                        "_sym":    s,
                    })
                except: pass
            st.session_state.scan_results = sorted(scan, key=lambda x: -x["_score"])

    if st.session_state.scan_results:
        df_sc = pd.DataFrame(st.session_state.scan_results)
        display_df = df_sc[["Stock","Price","Chg %","RSI","MACD","Vol","Score","Signal"]].copy()
        st.dataframe(display_df, hide_index=True, use_container_width=True, height=270)
        best_sym  = df_sc.iloc[0]["_sym"]
        best_idx  = stocks.index(best_sym) if best_sym in stocks else 0
    else:
        st.info("👆 Click **Scan All Stocks** to get AI-scored signals for all stocks in this universe")
        best_idx = 0

    stock = st.selectbox(
        "📌 Select Stock to Trade", stocks,
        index=best_idx,
        format_func=lambda x: x.replace(".NS", "")
    )

    # ── DATA + INDICATORS — smart fallback if market closed ──
    def load_data(sym, period, interval):
        """Try primary period/interval, fallback if market closed."""
        fallbacks = [
            (period, interval),
            ("5d",  "15m"),
            ("5d",  "30m"),
            ("1mo", "1d"),
            ("3mo", "1d"),
        ]
        seen = set()
        for fp, fi in fallbacks:
            key = f"{fp}_{fi}"
            if key in seen: continue
            seen.add(key)
            try:
                d = yf.Ticker(sym).history(period=fp, interval=fi)
                if d is not None and not d.empty and len(d) >= 20:
                    if (fp, fi) != (period, interval):
                        st.caption(f"ℹ️ Using {fp}/{fi} data (market closed for primary timeframe)")
                    return d
            except Exception:
                continue
        return None

    with st.spinner(f"Loading {stock.replace('.NS','')} [{selected_mode}]..."):
        df = load_data(stock, mcfg["period"], mcfg["interval"])

    if df is None or df.empty or len(df) < 5:
        st.error(f"⚠️ No data for {stock.replace('.NS','')} — market may be closed or check internet")
        st.info("💡 Tip: Switch to **🌊 Swing** mode — uses daily data which works anytime")
        st.stop()

    df    = compute_indicators(df)
    last  = df.iloc[-1]
    price = float(last["Close"])
    prev  = float(df["Close"].iloc[-2])
    chg_v = price - prev
    chg_p = chg_v / prev * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💲 Price",     f"₹{price:.2f}",     f"{chg_v:+.2f} ({chg_p:+.2f}%)")
    c2.metric("📊 RSI",       f"{last['RSI']:.1f}", "OB" if last['RSI']>70 else ("OS" if last['RSI']<30 else "OK"))
    c3.metric("📉 MACD",      f"{last['MACD']:.2f}",f"Hist: {last['MACD_Hist']:+.2f}")
    c4.metric("📈 ATR",       f"₹{last['ATR']:.2f}")
    c5.metric("🔊 Vol Ratio", f"{last['Vol_Ratio']:.2f}x")

    # ── AI MODEL (UPGRADED — 6 features) ─────────────────────
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    feat_cols    = ["EMA20","EMA50","RSI","MACD","Stoch_K","Vol_Ratio"]
    fd = df[feat_cols].dropna()
    td = df["Target"].loc[fd.index]
    if len(fd) >= 30:
        mdl = RandomForestClassifier(n_estimators=200, random_state=42)
        mdl.fit(fd, td)
        pred    = mdl.predict(fd.iloc[-1:].values)[0]
        ai_prob = mdl.predict_proba(fd.iloc[-1:].values)[0][1]
    else:
        pred = 0; ai_prob = 0.5

    # ── SIGNAL ENGINE (UPGRADED — 5 checks) ──────────────────
    st.markdown("---")
    col_sig, col_pos = st.columns(2)

    with col_sig:
        st.markdown(f"#### 🎯 {selected_mode} Signal")

        c_trend = last["Close"] > last["EMA20"] > last["EMA50"]
        c_rsi   = 45 < last["RSI"] < 75
        c_macd  = last["MACD"] > last["MACD_Signal"]
        c_vol   = last["Vol_Ratio"] > 1.1
        c_ai    = ai_prob > 0.55
        score   = sum([c_trend, c_rsi, c_macd, c_vol, c_ai])

        # SAFETY RULE: Block BUY if AI < 40% (strongly bearish)
        ai_blocked = ai_prob < 0.40
        signal = (score >= 3 and not ai_blocked) or force_trade

        # Determine direction
        if signal:
            direction = "BUY"
        elif ai_prob < 0.40 and score >= 2:
            direction = "SELL"
        else:
            direction = "WAIT"

        checks = {
            "Trend (Price > EMA20 > EMA50)":   c_trend,
            "RSI Bullish (45–75)":              c_rsi,
            "MACD > Signal Line":               c_macd,
            "Volume Surge (> 1.1×)":            c_vol,
            f"AI Bullish ({ai_prob*100:.0f}%)": c_ai,
        }

        if direction == "BUY":
            st.success(f"🟢 BUY SIGNAL | Score {score}/5 | AI {ai_prob*100:.0f}%")
            if ALERT_ON_SIGNAL:
                atr_now = float(last["ATR"])
                fire_alert(
                    f"BUY SIGNAL [{selected_mode}]", stock, price,
                    max(1, int((capital*risk/100)/max(atr_now*1.5,0.01))),
                    round(price-atr_now*1.5,2), round(price+atr_now*3,2),
                    score, mode
                )
        elif direction == "SELL":
            st.error(f"🔴 SELL / SHORT SIGNAL | Score {score}/5 | AI {ai_prob*100:.0f}% (Bearish)")
            if ALERT_ON_SIGNAL:
                atr_now = float(last["ATR"])
                fire_alert(
                    f"SELL SIGNAL [{selected_mode}]", stock, price,
                    max(1, int((capital*risk/100)/max(atr_now*1.5,0.01))),
                    round(price+atr_now*1.5,2), round(price-atr_now*3,2),
                    score, mode
                )
        else:
            if ai_blocked:
                st.warning(f"🟡 WAIT — AI Bearish ({ai_prob*100:.0f}%) | Score {score}/5 | Signal blocked for safety")
            else:
                st.warning(f"🟡 WAIT / NO TRADE | Score {score}/5 | AI {ai_prob*100:.0f}%")

        # AI confidence bar
        ai_color = "#27ae60" if ai_prob > 0.55 else ("#e74c3c" if ai_prob < 0.40 else "#f39c12")
        ai_pct = int(ai_prob * 100)
        st.markdown(f"""
        <div style="background:#f8f9fa;border-radius:6px;padding:8px 12px;margin:6px 0;border:1px solid #e0e0e0;">
        <div style="font-size:11px;color:#666;margin-bottom:4px;">🤖 AI Confidence</div>
        <div style="background:#e0e0e0;border-radius:99px;height:8px;">
          <div style="width:{ai_pct}%;background:{ai_color};border-radius:99px;height:8px;"></div>
        </div>
        <div style="font-size:12px;font-weight:600;color:{ai_color};margin-top:3px;">{ai_pct}% Bullish {'✅ Strong' if ai_pct>65 else ('⚠️ Weak' if ai_pct>40 else '🔴 Bearish')}</div>
        </div>""", unsafe_allow_html=True)

        chk_df = pd.DataFrame([{"Check": k, "Result": "✅" if v else "❌"} for k, v in checks.items()])
        st.dataframe(chk_df, hide_index=True, height=205)

    # ── POSITION SIZING (MODE-SPECIFIC — NEW) ─────────────────
    with col_pos:
        st.markdown(f"#### ⚖️ Position Sizing — {selected_mode}")
        atr       = max(float(last["ATR"]), 0.01)
        risk_amount = capital * (risk / 100)
        sl_dist   = atr * mcfg["sl_mult"]
        tgt_dist  = sl_dist * mcfg["rr"]
        stop_loss = round(price - sl_dist, 2)
        target_price = round(price + tgt_dist, 2)
        rr_ratio  = round(tgt_dist / sl_dist, 1)
        lot_size  = FO_LOTS.get(stock, 500)

        if selected_mode == "📈 Intraday":
            qty = max(1, int(risk_amount / sl_dist))
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("📦 Qty",      f"{qty} sh")
            p2.metric("💸 Risk ₹",   f"₹{risk_amount:,.0f}")
            p3.metric("🛡 Stop Loss", f"₹{stop_loss:,.2f}")
            p4.metric("🎯 Target",   f"₹{target_price:,.2f}")
            st.caption(f"Entry ₹{price:.2f}  |  ATR ₹{atr:.2f}  |  SL dist ₹{sl_dist:.2f}  |  R:R = {rr_ratio}:1  |  MIS")
            st.info("⏰ **Intraday rule:** Enter after 9:30 AM. Square off ALL positions before 3:15 PM.")

        elif selected_mode == "🌊 Swing":
            qty = max(1, int(risk_amount / sl_dist))
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("📦 Qty",      f"{qty} sh")
            p2.metric("💸 Risk ₹",   f"₹{risk_amount:,.0f}")
            p3.metric("🛡 Stop Loss", f"₹{stop_loss:,.2f}")
            p4.metric("🎯 Target",   f"₹{target_price:,.2f}")
            st.caption(f"Entry ₹{price:.2f}  |  ATR ₹{atr:.2f}  |  SL dist ₹{sl_dist:.2f}  |  R:R = {rr_ratio}:1  |  CNC")
            st.info(f"📅 **Swing rule:** Hold 3–15 days. Trail SL upward after each 1× ATR gain. Target R:R = {rr_ratio}:1")

        elif selected_mode == "📊 Futures":
            margin_est  = price * lot_size * 0.15
            profit_pot  = tgt_dist * lot_size
            loss_pot    = sl_dist  * lot_size
            qty = 1
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("📦 Lots",     "1 lot")
            p2.metric("📋 Lot Size", f"{lot_size} sh")
            p3.metric("🛡 Stop Loss",f"₹{stop_loss:,.2f}")
            p4.metric("🎯 Target",   f"₹{target_price:,.2f}")
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("💰 Margin",   f"₹{margin_est:,.0f}")
            m2.metric("📈 Profit",   f"₹{profit_pot:,.0f}")
            m3.metric("📉 Max Loss", f"₹{loss_pot:,.0f}")
            m4.metric("🔢 R:R",      f"{rr_ratio}:1")
            st.caption(f"Exposure = ₹{price*lot_size:,.0f}  |  NRML product  |  High leverage")
            st.warning(f"⚠️ Futures: Full exposure ₹{price*lot_size:,.0f}. Always use stop loss.")

        elif selected_mode == "🎯 Options":
            si  = 100 if price > 2000 else (50 if price > 500 else (20 if price > 100 else 10))
            atm = round(price / si) * si
            premium_est  = round(atr * 2.5, 2)
            cost_per_lot = premium_est * lot_size
            qty = 1
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("📦 Lots",     "1 lot")
            p2.metric("📋 Lot Size", f"{lot_size}")
            p3.metric("💸 Max Loss", f"₹{cost_per_lot:,.0f}")
            p4.metric("🎯 R:R",      f"{rr_ratio}:1")
            st.markdown("---")
            oc1, oc2 = st.columns(2)
            with oc1:
                st.markdown(f"""
**📞 CALL — Bullish View**
- Strike: **₹{atm} CE**
- Premium est: **₹{premium_est}/share**
- 1 lot cost: **₹{cost_per_lot:,.0f}**
- Target premium: **₹{round(premium_est*2.5,2)}**
- Max loss: ₹{cost_per_lot:,.0f}
                """)
            with oc2:
                st.markdown(f"""
**📉 PUT — Bearish View**
- Strike: **₹{atm} PE**
- Premium est: **₹{premium_est}/share**
- 1 lot cost: **₹{cost_per_lot:,.0f}**
- Target premium: **₹{round(premium_est*2.5,2)}**
- Max loss: ₹{cost_per_lot:,.0f}
                """)
            st.caption(f"Spot ₹{price:.2f}  |  ATM ₹{atm}  |  ATR ₹{atr:.2f}  |  NRML product")
            st.info("💡 Buy CE if BUY signal. Buy PE if bearish. Exit when premium doubles or halves.")

    # ── TRADE ENGINE (UPGRADED — mode-aware) ─────────────────
    st.markdown("---")

    def log_trade(action, stk, px, q, md, pnl=None):
        st.session_state.trade_log.append({
            "time":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strategy": selected_mode,
            "stock":    stk.replace(".NS",""),
            "action":   action,
            "price":    round(px, 2),
            "qty":      q,
            "mode":     md,
            "SL":       stop_loss,
            "Target":   target_price,
            "pnl":      round(pnl, 2) if pnl is not None else "—",
        })

    def order(txn):
        sym = stock.replace(".NS", "")

        # ── PAPER MODE ────────────────────────────────────────
        if mode == "Paper":
            try:
                live_px = float(yf.Ticker(stock).history(
                    period="1d", interval="1m")["Close"].iloc[-1])
            except:
                live_px = price

            if txn == "BUY":
                if st.session_state.paper_position:
                    st.warning(f"⚠️ Already holding {st.session_state.paper_position['stock']} — sell first"); return
                st.session_state.paper_position = {
                    "stock": stock, "price": live_px, "qty": qty,
                    "stop_loss": stop_loss, "target": target_price,
                    "strategy": selected_mode,
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.paper_balance -= live_px * qty
                log_trade("BUY", stock, live_px, qty, "Paper")
                st.success(f"📄 Paper BUY | {sym} | ₹{live_px:.2f}×{qty} | SL ₹{stop_loss} | TGT ₹{target_price} [{selected_mode}]")
                if ALERT_ON_EXECUTION:
                    fire_alert(f"BUY EXECUTED [{selected_mode}]", stock, live_px, qty,
                               stop_loss, target_price, score, "Paper")
                return

            if txn == "SELL":
                if not st.session_state.paper_position:
                    st.warning("⚠️ No open paper position to sell"); return
                pos  = st.session_state.paper_position
                pnl  = (live_px - pos["price"]) * pos["qty"]
                st.session_state.paper_balance += live_px * pos["qty"]
                st.session_state.pnl_history.append({
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "stock": stock, "pnl": round(pnl, 2), "strategy": selected_mode
                })
                log_trade("SELL", stock, live_px, pos["qty"], "Paper", pnl=pnl)
                st.session_state.paper_position = None
                emoji = "🟢" if pnl >= 0 else "🔴"
                st.success(f"📄 Paper SELL | {sym} | ₹{live_px:.2f} | P&L: {emoji} ₹{pnl:+.2f} [{selected_mode}]")
                if ALERT_ON_EXECUTION:
                    fire_alert(f"SELL EXECUTED [{selected_mode}]", stock, live_px, pos["qty"],
                               pos["stop_loss"], pos["target"], score, "Paper", pnl=pnl)
                return

        # ── LIVE MODE ─────────────────────────────────────────
        if not kite_ok():
            st.error("❌ Kite Not Connected"); return
        try:
            fo_qty = lot_size if selected_mode in ["📊 Futures","🎯 Options"] else qty
            kite.place_order(
                variety="regular", exchange="NSE",
                tradingsymbol=sym, transaction_type=txn,
                quantity=fo_qty, order_type="MARKET",
                product=mcfg["product"]
            )
            log_trade(txn, stock, price, fo_qty, "Live")
            st.success(f"✅ Live {txn} | {sym}×{fo_qty} | {mcfg['product']} [{selected_mode}]")
            if ALERT_ON_EXECUTION:
                fire_alert(f"{txn} EXECUTED LIVE [{selected_mode}]", stock, price, fo_qty,
                           stop_loss, target_price, score, "Live")
        except Exception as e:
            st.error(e)

    # ── BUY / SELL / REFRESH BUTTONS ─────────────────────────
    col1, col2, col3 = st.columns(3)
    if col1.button("🚀 BUY NOW",  use_container_width=True, type="primary"):
        order("BUY"); st.session_state.last_trade = datetime.datetime.now()
    if col2.button("🛑 SELL NOW", use_container_width=True):
        order("SELL"); st.session_state.last_trade = datetime.datetime.now()
    if col3.button("🔁 Refresh",  use_container_width=True):
        st.rerun()

    # Open paper position tracker
    if mode == "Paper" and st.session_state.paper_position:
        pos      = st.session_state.paper_position
        open_pnl = (price - pos["price"]) * pos["qty"]
        color    = "#2d8a4e" if open_pnl >= 0 else "#c0392b"
        st.markdown(f"""
        <div style="background:#f0fff4;border:1px solid #2d8a4e;border-radius:6px;
                    padding:10px 16px;margin:8px 0;">
            <b>📦 Open Paper Position [{pos.get('strategy','—')}]</b><br>
            {pos['stock'].replace('.NS','')} &nbsp;|&nbsp; Entry: ₹{pos['price']:.2f}
            &nbsp;|&nbsp; Qty: {pos['qty']}
            &nbsp;|&nbsp; SL: ₹{pos['stop_loss']}
            &nbsp;|&nbsp; TGT: ₹{pos['target']}<br>
            Unrealised P&amp;L: <b style="color:{color}">₹{open_pnl:+.2f}</b>
        </div>
        """, unsafe_allow_html=True)

    # ── AUTO LOOP ─────────────────────────────────────────────
    if auto_trade:
        if mode == "Live" and kite_ok() and signal and can_trade():
            order("BUY")
            st.session_state.last_trade = datetime.datetime.now()
        time.sleep(interval)
        st.rerun()

    # ── CHARTS (UPGRADED — EMA9 added, mode colour) ──────────
    st.markdown("---")
    chart_df = df.tail(100).copy()
    chart_df.index = pd.to_datetime(chart_df.index)
    lc = mcfg["color"]  # line colour per mode

    if PLOTLY_OK:
        st.subheader(f"📉 Price Chart — {stock.replace('.NS','')} [{selected_mode}]")
        fig1 = go.Figure()
        fig1.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"], high=chart_df["High"],
            low=chart_df["Low"],   close=chart_df["Close"],
            name="Price",
            increasing_line_color="#27ae60", decreasing_line_color="#e74c3c",
            increasing_fillcolor="#d5f5e3",  decreasing_fillcolor="#fadbd8",
        ))
        fig1.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA9"],
            line=dict(color="#00e5a0", width=1), name="EMA 9"))
        fig1.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA20"],
            line=dict(color="#2980b9", width=1.5), name="EMA 20"))
        fig1.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA50"],
            line=dict(color="#8e44ad", width=1.5, dash="dot"), name="EMA 50"))
        fig1.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_Upper"],
            line=dict(color="#95a5a6", width=1, dash="dash"), name="BB Upper"))
        fig1.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_Lower"],
            line=dict(color="#95a5a6", width=1, dash="dash"), name="BB Lower",
            fill="tonexty", fillcolor="rgba(149,165,166,0.07)"))
        fig1.add_hline(y=stop_loss,    line_color="#e74c3c", line_dash="dot", line_width=1.5,
            annotation_text=f"SL ₹{stop_loss}", annotation_font_color="#e74c3c")
        fig1.add_hline(y=target_price, line_color="#27ae60", line_dash="dot", line_width=1.5,
            annotation_text=f"TGT ₹{target_price}", annotation_font_color="#27ae60")
        fig1.add_hline(y=price,        line_color="#f39c12", line_dash="solid", line_width=1,
            annotation_text=f"LTP ₹{price:.2f}", annotation_font_color="#f39c12")
        fig1.update_layout(height=380, xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=0,t=30,b=0),
            xaxis=dict(showgrid=True, gridcolor="#ecf0f1"),
            yaxis=dict(showgrid=True, gridcolor="#ecf0f1"))
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("📊 RSI (14)")
        fig2 = go.Figure()
        fig2.add_hrect(y0=70, y1=100, fillcolor="rgba(231,76,60,0.08)",  line_width=0)
        fig2.add_hrect(y0=0,  y1=30,  fillcolor="rgba(39,174,96,0.08)",  line_width=0)
        fig2.add_hline(y=70, line_color="#e74c3c", line_dash="dot", line_width=1,
            annotation_text="Overbought 70", annotation_position="bottom right")
        fig2.add_hline(y=30, line_color="#27ae60", line_dash="dot", line_width=1,
            annotation_text="Oversold 30", annotation_position="top right")
        fig2.add_hline(y=50, line_color="#95a5a6", line_dash="dot", line_width=1)
        rsi_s = chart_df["RSI"]
        rsi_c = ["#e74c3c" if v>70 else ("#27ae60" if v<30 else lc) for v in rsi_s]
        fig2.add_trace(go.Scatter(x=chart_df.index, y=rsi_s,
            line=dict(color=lc, width=2), fill="tozeroy",
            fillcolor="rgba(41,128,185,0.07)", name="RSI"))
        fig2.add_trace(go.Scatter(x=chart_df.index, y=rsi_s, mode="markers",
            marker=dict(color=rsi_c, size=3), showlegend=False))
        fig2.add_annotation(x=chart_df.index[-1], y=rsi_s.iloc[-1],
            text=f"  {rsi_s.iloc[-1]:.1f}", showarrow=False,
            font=dict(color=lc, size=12, family="monospace"))
        fig2.update_layout(height=220,
            yaxis=dict(range=[0,100], showgrid=True, gridcolor="#ecf0f1"),
            xaxis=dict(showgrid=True, gridcolor="#ecf0f1"),
            margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📈 MACD")
        fig3 = go.Figure()
        hist_c = ["#27ae60" if v>=0 else "#e74c3c" for v in chart_df["MACD_Hist"]]
        fig3.add_trace(go.Bar(x=chart_df.index, y=chart_df["MACD_Hist"],
            marker_color=hist_c, name="Histogram", opacity=0.65))
        fig3.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD"],
            line=dict(color=lc, width=1.5), name="MACD"))
        fig3.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD_Signal"],
            line=dict(color="#f39c12", width=1.5, dash="dot"), name="Signal"))
        fig3.add_hline(y=0, line_color="#bdc3c7", line_width=1)
        fig3.update_layout(height=220,
            xaxis=dict(showgrid=True, gridcolor="#ecf0f1"),
            yaxis=dict(showgrid=True, gridcolor="#ecf0f1"),
            legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=0,t=10,b=0), barmode="relative")
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.warning("💡 Install plotly for rich charts:  pip install plotly")
        st.line_chart(chart_df[["Close","EMA20","EMA50","BB_Upper","BB_Lower"]])
        st.line_chart(chart_df[["RSI"]])
        st.line_chart(chart_df[["MACD","MACD_Signal"]])

# =============================================================
# TRADE LOGS — upgraded with strategy column
# =============================================================
st.markdown("---")
st.subheader("📋 Trade Logs")

if st.session_state.trade_log:
    log_df = pd.DataFrame(st.session_state.trade_log)
    st.dataframe(log_df, hide_index=True, use_container_width=True)
    closed = [x for x in st.session_state.pnl_history if x.get("pnl") is not None]
    if len(closed) > 1:
        pnl_df = pd.DataFrame(closed)
        pnl_df["Cumulative P&L"] = pnl_df["pnl"].cumsum()
        st.subheader("📈 Cumulative P&L")
        st.line_chart(pnl_df.set_index("time")["Cumulative P&L"])
    if st.button("🗑 Clear Logs"):
        st.session_state.trade_log   = []
        st.session_state.pnl_history = []
        st.rerun()
else:
    st.info("No trades yet — place your first trade above")
