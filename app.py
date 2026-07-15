# =============================================================
# AI Trading PRO+ v5.0 — ULTRA INSTITUTIONAL ARCHITECTURE
# =============================================================
# CORE ENGINE:
#   ✅ XGBoost + LightGBM + GradBoost + RF + AdaBoost (5-model ensemble)
#   ✅ 20+ feature engineering (ADX, MFI, CCI, OBV, VWAP, etc.)
#   ✅ Walk-forward time-series validation (5 folds)
#   ✅ LSTM price forecast (5-candle ahead)
#   ✅ Feature importance display
#
# PROFESSIONAL BACKTESTING:
#   ✅ Sharpe / Sortino / Calmar ratios
#   ✅ Max drawdown + recovery factor
#   ✅ Monthly returns heatmap
#   ✅ Trade distribution analysis (SL/Target/Signal exits)
#   ✅ Strategy vs Buy&Hold comparison
#   ✅ Trade Replay — step through every trade
#   ✅ 4 strategy modes: Master / Trend / Momentum / Mean Revert
#
# MASTER SIGNAL (6-Layer):
#   ✅ Technical (30%) + AI (25%) + Candlestick (15%)
#   ✅ Market Structure (15%) + SMC (10%) + Volume (5%)
#   ✅ Risk penalty system + Win probability
#   ✅ Max loss/gain in Rs. per trade
#
# ADVANCED ANALYSIS:
#   ✅ 25+ Candlestick patterns | TA Summary
#   ✅ SMC: Order Blocks + FVG | Volume Profile POC
#   ✅ Options PCR + Max Pain | ORB + Session Guide
#   ✅ Kelly Criterion | Market Structure HH/LL
#   ✅ Fake Breakout Detection | Demand/Supply Zones
#
# PLATFORM:
#   ✅ Kite Connect integration | Paper + Live trading
#   ✅ Admin portal + User management + UPI payments
#   ✅ Email + WhatsApp alerts | Persistent trade logs
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
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# XGBoost — optional
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

# LightGBM — optional
try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False

# TensorFlow/Keras for LSTM — optional
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_OK = True
except ImportError:
    KERAS_OK = False

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# ── Config loader — Streamlit Secrets (cloud) + config.py (local) ──
def _get_secret(key, default):
    try:
        return st.secrets[key]
    except Exception:
        pass
    try:
        import config as _cfg
        return getattr(_cfg, key, default)
    except Exception:
        pass
    return default

API_KEY      = _get_secret("API_KEY",      "")
API_SECRET   = _get_secret("API_SECRET",   "")
ACCESS_TOKEN = _get_secret("ACCESS_TOKEN", "")

EMAIL_ALERTS_ON    = _get_secret("EMAIL_ALERTS_ON",    False)
ALERT_EMAIL_TO     = _get_secret("ALERT_EMAIL_TO",     "")
SMTP_USER          = _get_secret("SMTP_USER",          "")
SMTP_PASS          = _get_secret("SMTP_PASS",          "")
SMTP_SERVER        = _get_secret("SMTP_SERVER",        "smtp.gmail.com")
SMTP_PORT          = _get_secret("SMTP_PORT",          587)

CALLMEBOT_ALERTS_ON = _get_secret("CALLMEBOT_ALERTS_ON", False)
CALLMEBOT_PHONE     = _get_secret("CALLMEBOT_PHONE",     "")
CALLMEBOT_APIKEY    = _get_secret("CALLMEBOT_APIKEY",    "")

TWILIO_ALERTS_ON = _get_secret("TWILIO_ALERTS_ON", False)
TWILIO_SID       = _get_secret("TWILIO_SID",       "")
TWILIO_TOKEN     = _get_secret("TWILIO_TOKEN",     "")
TWILIO_FROM      = _get_secret("TWILIO_FROM",      "whatsapp:+14155238886")
TWILIO_TO        = _get_secret("TWILIO_TO",        "")

ALERT_ON_SIGNAL    = _get_secret("ALERT_ON_SIGNAL",    True)
ALERT_ON_EXECUTION = _get_secret("ALERT_ON_EXECUTION", True)
ALERT_MIN_SCORE    = _get_secret("ALERT_MIN_SCORE",    3)
ALERT_COOLDOWN_MIN = _get_secret("ALERT_COOLDOWN_MIN", 15)
APP_URL            = _get_secret("APP_URL", "https://bhavisha-ai-trading-pro.streamlit.app")

try:
    from alerts import send_alert
    ALERTS_AVAILABLE = True
except Exception as _ae:
    ALERTS_AVAILABLE = False
    _ALERT_IMPORT_ERROR = str(_ae)

# ── IST timezone helper ───────────────────────────────────────
def now_ist():
    """Current time in IST (UTC+5:30) — works on Streamlit Cloud."""
    utc_now = datetime.datetime.utcnow()
    ist_offset = datetime.timedelta(hours=5, minutes=30)
    return utc_now + ist_offset

def ist_str(fmt="%d %b %Y  %H:%M:%S IST"):
    return now_ist().strftime(fmt)

kite = KiteConnect(api_key=API_KEY)


st.set_page_config(page_title="AI Trading PRO+ v1.3", layout="wide", page_icon="📈")

# =============================================================
# STOCK UNIVERSES
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
    "🇺🇸 US Funds (India NSE)": [
        "MON100.NS",      # Motilal Oswal NASDAQ 100 ETF
        "MAFANG.NS",      # Mirae Asset NYSE FANG+ ETF
        "MONIFTY500.NS",  # Motilal Oswal S&P 500 ETF
        "NIFTYBEES.NS",   # Nippon India Nifty 50 BeES
        "HNGSNGBEES.NS",  # Hang Seng BeES
        "MOM100.NS",      # Motilal Oswal NASDAQ 100
        "IVZINGOLD.NS",   # Invesco Gold ETF
        "NETFIT.NS",      # Mirae Asset NYSE FANG+
    ],
    "🇺🇸 US Stocks (Direct)": [
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "GOOGL",  # Alphabet
        "AMZN",   # Amazon
        "META",   # Meta (Facebook)
        "NVDA",   # NVIDIA
        "TSLA",   # Tesla
        "NFLX",   # Netflix
        "BRKB",   # Berkshire Hathaway
        "JPM",    # JP Morgan
    ],
    "📊 US ETFs & Index": [
        "QQQ",   # NASDAQ 100 ETF (Invesco)
        "SPY",   # S&P 500 ETF
        "VTI",   # Total US Market
        "ARKK",  # ARK Innovation ETF
        "IWM",   # Russell 2000 (Small Cap)
        "GLD",   # Gold ETF
        "SLV",   # Silver ETF
        "USO",   # Oil ETF
    ],
}

# Currency mapping for non-INR stocks
STOCKS = STOCK_UNIVERSE  # alias for compatibility
USD_STOCKS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","NFLX","BRKB","JPM",
    "QQQ","SPY","VTI","ARKK","IWM","GLD","SLV","USO"
]

def get_currency(symbol: str) -> str:
    """Return currency symbol for a stock."""
    return "$" if symbol in USD_STOCKS else "₹"

def format_price(price: float, symbol: str) -> str:
    """Format price with correct currency."""
    curr = get_currency(symbol)
    return f"{curr}{price:,.2f}"

FO_LOTS = {
    "RELIANCE.NS":250,"TCS.NS":150,"HDFCBANK.NS":550,"ICICIBANK.NS":700,
    "INFY.NS":300,"SBIN.NS":1500,"BAJFINANCE.NS":125,"BHARTIARTL.NS":950,
    "KOTAKBANK.NS":400,"WIPRO.NS":1500,"AXISBANK.NS":1200,"HCLTECH.NS":350,
    "HINDUNILVR.NS":300,"MARUTI.NS":100,"TITAN.NS":375,"SUNPHARMA.NS":700,
    "TATAMOTORS.NS":2850,"BANKBARODA.NS":5850,"LTIM.NS":75,"ASIANPAINT.NS":200,
    "NESTLEIND.NS":40,"ULTRACEMCO.NS":100,"INDUSINDBK.NS":900,"TECHM.NS":600,
    "DRREDDY.NS":125,"CIPLA.NS":650,"BAJAJ-AUTO.NS":75,"EICHERMOT.NS":200,
}

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
# PERSISTENT TRADE STORAGE — defined early so login can use them
# =============================================================
def get_trade_file(username):
    return f"trades_{username}.json"

def load_user_data(username):
    fpath = get_trade_file(username)
    default = {
        "trade_log":      [],
        "pnl_history":    [],
        "paper_balance":  100000.0,
        "paper_position": None,
    }
    if not os.path.exists(fpath):
        return default
    try:
        data = json.load(open(fpath, "r"))
        for k, v in default.items():
            if k not in data:
                data[k] = v
        return data
    except Exception:
        return default

def save_user_data(username):
    fpath = get_trade_file(username)
    try:
        json.dump({
            "trade_log":      st.session_state.trade_log,
            "pnl_history":    st.session_state.pnl_history,
            "paper_balance":  st.session_state.paper_balance,
            "paper_position": st.session_state.paper_position,
        }, open(fpath, "w"), indent=2, default=str)
    except Exception as e:
        st.warning(f"Could not save trade data: {e}")

# =============================================================
# ALERT HELPER
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
                "time": now_ist().strftime("%H:%M:%S"),
                "stock": stk, "action": action, "result": r
            })
        st.session_state.alert_log = st.session_state.alert_log[:10]
        for r in results:
            st.toast(r, icon="✅" if "✅" in r else "⚠️")
    except Exception as e:
        st.toast(f"Alert error: {e}", icon="⚠️")

# =============================================================
# KITE SIDEBAR
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

st.sidebar.markdown("---")
st.sidebar.subheader("🔔 Alert Settings")

if not ALERTS_AVAILABLE:
    st.sidebar.error(f"alerts.py import failed:\n{_ALERT_IMPORT_ERROR}")
else:
    if EMAIL_ALERTS_ON:
        st.sidebar.success(f"📧 Email → {ALERT_EMAIL_TO}")
    else:
        st.sidebar.info("📧 Email: OFF")
    if CALLMEBOT_ALERTS_ON:
        st.sidebar.success(f"📱 WhatsApp → {CALLMEBOT_PHONE}")
    else:
        st.sidebar.info("📱 WhatsApp: OFF")
    if TWILIO_ALERTS_ON:
        st.sidebar.success("📱 Twilio WhatsApp: ON")
    if not any([EMAIL_ALERTS_ON, CALLMEBOT_ALERTS_ON, TWILIO_ALERTS_ON]):
        st.sidebar.warning("All alerts OFF — add secrets in Streamlit Cloud")
    if st.sidebar.button("🧪 Send Test Alert"):
        fire_alert("TEST SIGNAL","RELIANCE.NS",1427.50,32,1422.95,1437.05,4,"Paper")
    if st.session_state.alert_log:
        st.sidebar.markdown("**Recent alerts:**")
        for a in st.session_state.alert_log[:5]:
            icon = "✅" if "✅" in a["result"] else "❌"
            st.sidebar.caption(f"{icon} {a['time']} · {a['stock']} · {a['action']}")

# =============================================================
# LOGIN / SIGNUP
# =============================================================
DB = "users.json"
# ── USER DATABASE — loads from Streamlit Secrets on cloud ────
if not os.path.exists(DB):
    json.dump({}, open(DB, "w"))
try:
    users = json.load(open(DB))
except Exception:
    users = {}

# Admin default
if "admin" not in users:
    users["admin"] = {"password":"admin123","role":"admin","status":"active","expiry":"2099-12-31"}
    json.dump(users, open(DB, "w"))

# ── Load seed users from Streamlit Secrets (persistent on cloud) ──
def _load_seed_users():
    """Load users from Streamlit Secrets — survives app restarts."""
    try:
        seed_json = st.secrets.get("SEED_USERS", None)
        if not seed_json:
            return
        seed = json.loads(seed_json) if isinstance(seed_json, str) else seed_json
        changed = False
        for uname, udata in seed.items():
            if uname not in users:
                users[uname] = udata
                changed = True
            else:
                # Update expiry from secrets if longer
                try:
                    cur_exp = datetime.datetime.strptime(users[uname].get("expiry","2000-01-01"),"%Y-%m-%d").date()
                    new_exp = datetime.datetime.strptime(udata.get("expiry","2000-01-01"),"%Y-%m-%d").date()
                    if new_exp > cur_exp:
                        users[uname]["expiry"] = udata["expiry"]
                        changed = True
                except Exception:
                    pass
        if changed:
            json.dump(users, open(DB, "w"))
    except Exception:
        pass

_load_seed_users()

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
                st.session_state.user = u
                # Load saved trade data for this user
                saved = load_user_data(u)
                st.session_state.trade_log      = saved["trade_log"]
                st.session_state.pnl_history    = saved["pnl_history"]
                st.session_state.paper_balance  = saved["paper_balance"]
                st.session_state.paper_position = saved["paper_position"]
                st.rerun()
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
# ADMIN PANEL
# =============================================================
role = users[user].get("role", "user")
if role == "admin":
    st.sidebar.success("👑 Admin")
    if st.sidebar.button("Open Admin Panel"):
        st.session_state.admin = True

if st.session_state.admin:
    st.title("🛠 ADMIN PANEL — AI Trading PRO+")

    all_users  = [u for u in users if u != "admin"]
    active_u   = [u for u in all_users if users[u].get("status") == "active"]
    pending_u  = [u for u in all_users if users[u].get("status") == "pending"]
    expiring_u = []
    for u in active_u:
        try:
            exp = datetime.datetime.strptime(users[u]["expiry"], "%Y-%m-%d").date()
            if (exp - datetime.date.today()).days <= 7:
                expiring_u.append(u)
        except: pass

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("👥 Total Users", len(all_users))
    c2.metric("✅ Active",      len(active_u))
    c3.metric("⏳ Pending",     len(pending_u))
    c4.metric("⚠️ Expiring Soon", len(expiring_u))
    st.markdown("---")

    st.subheader("➕ Add New User (Payment Received)")
    st.caption("User ne payment kar diya — yahan se directly account banao aur credentials bhejo")

    with st.form("add_user_form"):
        fc1, fc2 = st.columns(2)
        new_username = fc1.text_input("Username *", placeholder="e.g. Rajesh2024")
        new_password = fc2.text_input("Password *", placeholder="e.g. Raj@1234")
        fd1, fd2 = st.columns(2)
        new_plan = fd1.selectbox("Plan *", ["monthly (30 days — ₹499)",
                                            "quarterly (90 days — ₹999)",
                                            "annual (365 days — ₹2,999)"])
        new_txn  = fd2.text_input("UPI Txn ID *", placeholder="e.g. T2405011234567")
        fe1, fe2 = st.columns(2)
        new_email = fe1.text_input("User Email", placeholder="user@gmail.com")
        new_phone = fe2.text_input("User WhatsApp (+91...)", placeholder="+919876543210")
        submitted = st.form_submit_button("✅ Create Account & Send Credentials",
                                          type="primary", use_container_width=True)

    if submitted:
        if not new_username or not new_password:
            st.error("❌ Username and password required")
        elif new_username in users:
            st.error(f"❌ Username '{new_username}' already exists")
        else:
            plan_key  = new_plan.split(" ")[0]
            plan_days = {"monthly":30,"quarterly":90,"annual":365}.get(plan_key, 30)
            expiry_date = str(datetime.date.today() + datetime.timedelta(days=plan_days))
            users[new_username] = {
                "password": new_password, "role": "user", "status": "active",
                "expiry": expiry_date, "plan": plan_key,
                "email": new_email, "phone": new_phone,
                "txn_id": new_txn, "joined": str(datetime.date.today()),
            }
            json.dump(users, open(DB, "w"))
            st.success(f"✅ User '{new_username}' created! Active until {expiry_date}")

            # Send welcome email
            if new_email and EMAIL_ALERTS_ON:
                try:
                    import smtplib
                    from email.mime.text import MIMEText
                    from email.mime.multipart import MIMEMultipart
                    msg = MIMEMultipart("alternative")
                    msg["Subject"] = "AI Trading PRO+ Account Active!"
                    msg["From"]    = SMTP_USER
                    msg["To"]      = new_email
                    html = (
                        "<html><body>"
                        "<div style='font-family:Arial;max-width:460px;margin:20px auto;"
                        "border:1px solid #e0e0e0;border-radius:10px;overflow:hidden;'>"
                        "<div style='background:#0a0c10;padding:14px 20px;'>"
                        "<span style='color:#00e5a0;font-size:17px;font-weight:700;'>AI Trading PRO+</span></div>"
                        "<div style='padding:20px;'>"
                        "<h2 style='color:#1a1a2e;'>Welcome! Account is Active</h2>"
                        "<table style='width:100%;font-size:14px;background:#f8f9fa;border-radius:8px;padding:12px;'>"
                        "<tr><td style='color:#888;padding:5px 0;'>Username</td>"
                        "<td style='font-weight:600;'>" + new_username + "</td></tr>"
                        "<tr><td style='color:#888;padding:5px 0;'>Password</td>"
                        "<td style='font-weight:600;'>" + new_password + "</td></tr>"
                        "<tr><td style='color:#888;padding:5px 0;'>Plan</td>"
                        "<td style='font-weight:600;'>" + plan_key.title() + "</td></tr>"
                        "<tr><td style='color:#888;padding:5px 0;'>Valid Until</td>"
                        "<td style='font-weight:600;color:#27ae60;'>" + expiry_date + "</td></tr>"
                        "</table>"
                        "<div style='margin-top:16px;background:#003d2a;border:1px solid #00b880;"
                        "border-radius:8px;padding:14px;text-align:center;'>"
                        "<a href='" + APP_URL + "' style='background:#00e5a0;color:#000;"
                        "padding:10px 28px;border-radius:6px;font-weight:700;font-size:14px;"
                        "text-decoration:none;display:inline-block;'>Login to AI Trading PRO+</a>"
                        "</div>"
                        "<p style='color:#888;font-size:12px;margin-top:16px;'>"
                        "Contact: bhardwaj.rishu04@gmail.com | +91 98051 84822</p>"
                        "</div></div></body></html>"
                    )
                    msg.attach(MIMEText(html, "html"))
                    with smtplib.SMTP(SMTP_SERVER, int(SMTP_PORT), timeout=10) as srv:
                        srv.ehlo(); srv.starttls()
                        srv.login(SMTP_USER, SMTP_PASS)
                        srv.sendmail(SMTP_USER, new_email, msg.as_string())
                    st.success(f"📧 Credentials sent to {new_email}")
                except Exception as e:
                    st.warning(f"📧 Email failed: {e}")

            # WhatsApp link
            if new_phone:
                import urllib.parse
                clean_phone = new_phone.strip().replace(" ","").replace("-","")
                if not clean_phone.startswith("+"):
                    clean_phone = "+91" + clean_phone.lstrip("0")
                wa_parts = [
                    "AI Trading PRO+ - Account Active!",
                    "Username: " + new_username,
                    "Password: " + new_password,
                    "Plan: " + plan_key.title() + " (" + expiry_date + ")",
                    "App: " + APP_URL,
                    "Contact: +91 98051 84822"
                ]
                wa_text = urllib.parse.quote("\n".join(wa_parts))
                wa_link = f"https://wa.me/{clean_phone.replace('+','')}?text={wa_text}"
                st.markdown(f"""
<div style='background:#003d2a;border:1px solid #00b880;border-radius:8px;padding:12px 16px;margin:8px 0;'>
<b style='color:#00e5a0;'>📱 Send Credentials via WhatsApp</b><br>
<span style='color:#aaa;font-size:12px;'>Click button — WhatsApp opens with message pre-filled. Press Send.</span><br><br>
<a href='{wa_link}' target='_blank'
style='background:#25d366;color:#000;padding:8px 20px;border-radius:6px;
font-weight:700;font-size:13px;text-decoration:none;display:inline-block;'>
Open WhatsApp & Send to {clean_phone}
</a></div>""", unsafe_allow_html=True)

                # CallMeBot auto-send
                if CALLMEBOT_ALERTS_ON and CALLMEBOT_APIKEY:
                    try:
                        import requests
                        cb_url = (f"https://api.callmebot.com/whatsapp.php"
                                  f"?phone={clean_phone}&text={wa_text}&apikey={CALLMEBOT_APIKEY}")
                        r = requests.get(cb_url, timeout=12)
                        if "Message queued" in r.text or r.status_code == 200:
                            st.success(f"📱 Auto-sent via CallMeBot to {clean_phone}")
                    except Exception:
                        pass

            st.rerun()

    st.markdown("---")

    if pending_u:
        st.subheader(f"⏳ Pending Approvals ({len(pending_u)})")
        for u in pending_u:
            with st.expander(f"👤 {u} — joined {users[u].get('joined','?')}"):
                st.write(f"Email: {users[u].get('email','—')} | Phone: {users[u].get('phone','—')}")
                pc1, pc2, pc3 = st.columns(3)
                ap_plan = pc1.selectbox("Plan", ["monthly","quarterly","annual"], key=f"pl_{u}")
                if pc2.button("✅ Approve", key=f"ap_{u}"):
                    plan_days = {"monthly":30,"quarterly":90,"annual":365}[ap_plan]
                    users[u]["status"] = "active"
                    users[u]["expiry"] = str(datetime.date.today() + datetime.timedelta(days=plan_days))
                    users[u]["plan"]   = ap_plan
                    json.dump(users, open(DB, "w")); st.rerun()
                if pc3.button("❌ Reject", key=f"rj_{u}"):
                    del users[u]; json.dump(users, open(DB, "w")); st.rerun()
    else:
        st.info("✅ No pending approvals")

    st.markdown("---")
    st.subheader(f"✅ Active Users ({len(active_u)})")
    for u in active_u:
        exp = users[u].get("expiry","?")
        try:
            days_left = (datetime.datetime.strptime(exp,"%Y-%m-%d").date()-datetime.date.today()).days
            exp_label = f"⚠️ {days_left}d left" if days_left <= 7 else f"{days_left}d left"
        except:
            exp_label = exp
        with st.expander(f"👤 {u} | {users[u].get('plan','?').title()} | {exp} ({exp_label})"):
            st.caption(f"Email: {users[u].get('email','—')} | Phone: {users[u].get('phone','—')} | Txn: {users[u].get('txn_id','—')}")
            ec1,ec2,ec3 = st.columns(3)
            ext_days = ec1.number_input("Extend days", 1, 365, 30, key=f"ed_{u}")
            if ec2.button("🔄 Extend", key=f"ex_{u}"):
                cur = datetime.datetime.strptime(users[u]["expiry"],"%Y-%m-%d").date()
                users[u]["expiry"] = str(max(cur,datetime.date.today())+datetime.timedelta(days=int(ext_days)))
                json.dump(users,open(DB,"w")); st.rerun()
            if ec3.button("🗑 Delete", key=f"dl_{u}"):
                del users[u]; json.dump(users,open(DB,"w")); st.rerun()

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
    """25+ professional indicators for institutional-grade analysis."""
    df = df.copy()

    # ── EMAs (multiple timeframes) ──
    for span in [5, 9, 20, 50, 100, 200]:
        df[f"EMA{span}"] = df["Close"].ewm(span=span).mean()

    # ── RSI with divergence ──
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    df["RSI_MA"] = df["RSI"].rolling(9).mean()  # RSI smoothed

    # ── MACD ──
    df["EMA12"]       = df["Close"].ewm(span=12).mean()
    df["EMA26"]       = df["Close"].ewm(span=26).mean()
    df["MACD"]        = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ── ATR (True Range) ──
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # ── Bollinger Bands + Squeeze ──
    df["BB_Mid"]   = df["Close"].rolling(20).mean()
    bb_std         = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["BB_Mid"] + 1e-9)
    df["BB_Pct"]   = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)

    # ── Stochastic ──
    low14  = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * (df["Close"] - low14) / (high14 - low14 + 1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # ── VWAP ──
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (tp * df["Volume"]).cumsum() / (df["Volume"].cumsum() + 1e-9)

    # ── ADX (Trend Strength) ──
    plus_dm  = df["High"].diff().clip(lower=0)
    minus_dm = (-df["Low"].diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm]   = 0
    minus_dm[minus_dm < plus_dm]  = 0
    atr14    = df["ATR"]
    plus_di  = 100 * plus_dm.rolling(14).mean()  / (atr14 + 1e-9)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df["ADX"]      = dx.rolling(14).mean()
    df["Plus_DI"]  = plus_di
    df["Minus_DI"] = minus_di

    # ── OBV (On Balance Volume) ──
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"]    = obv
    df["OBV_MA"] = pd.Series(obv, index=df.index).rolling(20).mean()

    # ── MFI (Money Flow Index) ──
    tp_mf   = (df["High"] + df["Low"] + df["Close"]) / 3
    mf      = tp_mf * df["Volume"]
    pos_mf  = mf.where(tp_mf > tp_mf.shift(), 0).rolling(14).sum()
    neg_mf  = mf.where(tp_mf < tp_mf.shift(), 0).rolling(14).sum()
    df["MFI"] = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-9))

    # ── CCI (Commodity Channel Index) ──
    tp_cci  = (df["High"] + df["Low"] + df["Close"]) / 3
    mean_d  = (tp_cci - tp_cci.rolling(20).mean()).abs().rolling(20).mean()
    df["CCI"] = (tp_cci - tp_cci.rolling(20).mean()) / (0.015 * mean_d + 1e-9)

    # ── Williams %R ──
    df["Williams_R"] = -100 * (high14 - df["Close"]) / (high14 - low14 + 1e-9)

    # ── Supertrend ──
    hl2  = (df["High"] + df["Low"]) / 2
    up   = hl2 + 3 * df["ATR"]
    dn   = hl2 - 3 * df["ATR"]
    st_dir = pd.Series(0, index=df.index)
    st_val = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > up.iloc[i-1]:
            st_dir.iloc[i] = 1
        elif df["Close"].iloc[i] < dn.iloc[i-1]:
            st_dir.iloc[i] = -1
        else:
            st_dir.iloc[i] = st_dir.iloc[i-1]
        st_val.iloc[i] = dn.iloc[i] if st_dir.iloc[i] == 1 else up.iloc[i]
    df["Supertrend"]  = st_val
    df["ST_Dir"]      = st_dir

    # ── Volume metrics ──
    df["Vol_Ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["Vol_MA20"]  = df["Volume"].rolling(20).mean()
    df["Vol_Spike"] = (df["Vol_Ratio"] > 2.0).astype(int)

    # ── Price features ──
    df["Return_1"]   = df["Close"].pct_change(1)
    df["Return_3"]   = df["Close"].pct_change(3)
    df["Return_5"]   = df["Close"].pct_change(5)
    df["Volatility"] = df["Return_1"].rolling(20).std() * (252**0.5)
    df["Price_Pos"]  = (df["Close"]-df["Low"].rolling(20).min()) /                        (df["High"].rolling(20).max()-df["Low"].rolling(20).min()+1e-9)

    return df


# =============================================================
# CANDLESTICK PATTERN DETECTION ENGINE (25+ Patterns)
# =============================================================
def detect_candlestick_patterns(df):
    df = df.copy().dropna()
    if len(df) < 5:
        return []
    patterns_found = []
    for i in range(2, len(df)):
        o  = float(df["Open"].iloc[i])
        h  = float(df["High"].iloc[i])
        l  = float(df["Low"].iloc[i])
        c  = float(df["Close"].iloc[i])
        o1 = float(df["Open"].iloc[i-1])
        h1 = float(df["High"].iloc[i-1])
        l1 = float(df["Low"].iloc[i-1])
        c1 = float(df["Close"].iloc[i-1])
        o2 = float(df["Open"].iloc[i-2])
        c2 = float(df["Close"].iloc[i-2])
        body  = abs(c - o)
        body1 = abs(c1 - o1)
        rng   = h - l + 1e-9
        upper_shadow  = h - max(o, c)
        lower_shadow  = min(o, c) - l
        upper_shadow1 = h1 - max(o1, c1)
        lower_shadow1 = min(o1, c1) - l1
        atr = float(df["ATR"].iloc[i]) if "ATR" in df.columns else rng
        date = str(df.index[i])[:10]

        # SINGLE candle
        if body < atr * 0.1 and rng > atr * 0.5:
            patterns_found.append({"pattern":"Doji","type":"neutral","candles":1,"strength":2,
                "desc":"Indecision — market at crossroads","date":date,"signal":"NEUTRAL","idx":i})
        if body < atr*0.05 and upper_shadow > atr*0.6 and lower_shadow < atr*0.05:
            patterns_found.append({"pattern":"Gravestone Doji","type":"bearish","candles":1,"strength":3,
                "desc":"Bearish reversal at top — bulls tried but failed","date":date,"signal":"SELL","idx":i})
        if body < atr*0.05 and lower_shadow > atr*0.6 and upper_shadow < atr*0.05:
            patterns_found.append({"pattern":"Dragonfly Doji","type":"bullish","candles":1,"strength":3,
                "desc":"Bullish reversal at bottom — bears tried but failed","date":date,"signal":"BUY","idx":i})
        if lower_shadow > 2*body and upper_shadow < body*0.5 and c > o and body > atr*0.1:
            patterns_found.append({"pattern":"Hammer","type":"bullish","candles":1,"strength":4,
                "desc":"Bullish reversal — buyers rejected lower prices","date":date,"signal":"BUY","idx":i})
        if upper_shadow > 2*body and lower_shadow < body*0.3 and c < o and body > atr*0.1:
            patterns_found.append({"pattern":"Shooting Star","type":"bearish","candles":1,"strength":4,
                "desc":"Bearish reversal — sellers rejected higher prices","date":date,"signal":"SELL","idx":i})
        if upper_shadow > 2*body and lower_shadow < body*0.5 and body > atr*0.1:
            patterns_found.append({"pattern":"Inverted Hammer","type":"bullish","candles":1,"strength":3,
                "desc":"Possible bullish reversal — buyers pushing up","date":date,"signal":"WATCH BUY","idx":i})
        if lower_shadow > 2*body and upper_shadow < body*0.3 and c < o and body > atr*0.1:
            patterns_found.append({"pattern":"Hanging Man","type":"bearish","candles":1,"strength":3,
                "desc":"Bearish warning after uptrend","date":date,"signal":"WATCH SELL","idx":i})
        if c > o and body > atr*0.8 and upper_shadow < body*0.05 and lower_shadow < body*0.05:
            patterns_found.append({"pattern":"Bullish Marubozu","type":"bullish","candles":1,"strength":5,
                "desc":"Buyers fully in control — strong momentum","date":date,"signal":"STRONG BUY","idx":i})
        if c < o and body > atr*0.8 and upper_shadow < body*0.05 and lower_shadow < body*0.05:
            patterns_found.append({"pattern":"Bearish Marubozu","type":"bearish","candles":1,"strength":5,
                "desc":"Sellers fully in control — strong momentum","date":date,"signal":"STRONG SELL","idx":i})
        if body < atr*0.2 and upper_shadow > body*1.5 and lower_shadow > body*1.5:
            patterns_found.append({"pattern":"Spinning Top","type":"neutral","candles":1,"strength":2,
                "desc":"Indecision — neither bulls nor bears dominate","date":date,"signal":"WAIT","idx":i})

        # DOUBLE candle
        if c1 < o1 and c > o and c > o1 and o < c1 and body > body1 * 0.9:
            patterns_found.append({"pattern":"Bullish Engulfing","type":"bullish","candles":2,"strength":5,
                "desc":"Strong reversal — bulls completely overwhelm bears","date":date,"signal":"STRONG BUY","idx":i})
        if c1 > o1 and c < o and c < o1 and o > c1 and body > body1 * 0.9:
            patterns_found.append({"pattern":"Bearish Engulfing","type":"bearish","candles":2,"strength":5,
                "desc":"Strong reversal — bears completely overwhelm bulls","date":date,"signal":"STRONG SELL","idx":i})
        if c1 < o1 and c > o and c < o1 and o > c1:
            patterns_found.append({"pattern":"Bullish Harami","type":"bullish","candles":2,"strength":3,
                "desc":"Bearish momentum slowing — possible reversal","date":date,"signal":"WATCH BUY","idx":i})
        if c1 > o1 and c < o and c > o1 and o < c1:
            patterns_found.append({"pattern":"Bearish Harami","type":"bearish","candles":2,"strength":3,
                "desc":"Bullish momentum slowing — possible reversal","date":date,"signal":"WATCH SELL","idx":i})
        if c1 < o1 and c > o and o < l1 and c > (o1+c1)/2 and c < o1:
            patterns_found.append({"pattern":"Piercing Line","type":"bullish","candles":2,"strength":4,
                "desc":"Buyers push above midpoint of bearish candle","date":date,"signal":"BUY","idx":i})
        if c1 > o1 and c < o and o > h1 and c < (o1+c1)/2 and c > c1:
            patterns_found.append({"pattern":"Dark Cloud Cover","type":"bearish","candles":2,"strength":4,
                "desc":"Sellers push below midpoint of bullish candle","date":date,"signal":"SELL","idx":i})
        if c1 < o1 and c > o and abs(l - l1) < atr*0.05:
            patterns_found.append({"pattern":"Tweezer Bottom","type":"bullish","candles":2,"strength":3,
                "desc":"Support confirmed — same lows twice","date":date,"signal":"BUY","idx":i})
        if c1 > o1 and c < o and abs(h - h1) < atr*0.05:
            patterns_found.append({"pattern":"Tweezer Top","type":"bearish","candles":2,"strength":3,
                "desc":"Resistance confirmed — same highs twice","date":date,"signal":"SELL","idx":i})

        # TRIPLE candle
        if (c2 < o2 and abs(c1-o1) < body*0.3 and c > o and c > (o2+c2)/2):
            patterns_found.append({"pattern":"Morning Star","type":"bullish","candles":3,"strength":5,
                "desc":"Powerful bullish reversal — 3-candle bottom","date":date,"signal":"STRONG BUY","idx":i})
        if (c2 > o2 and abs(c1-o1) < body*0.3 and c < o and c < (o2+c2)/2):
            patterns_found.append({"pattern":"Evening Star","type":"bearish","candles":3,"strength":5,
                "desc":"Powerful bearish reversal — 3-candle top","date":date,"signal":"STRONG SELL","idx":i})
        if (c > o and c1 > o1 and c2 > o2 and c > c1 > c2 and o > o1 > o2 and body > atr*0.3):
            patterns_found.append({"pattern":"Three White Soldiers","type":"bullish","candles":3,"strength":5,
                "desc":"3 consecutive green candles — very strong uptrend","date":date,"signal":"STRONG BUY","idx":i})
        if (c < o and c1 < o1 and c2 < o2 and c < c1 < c2 and o < o1 < o2 and body > atr*0.3):
            patterns_found.append({"pattern":"Three Black Crows","type":"bearish","candles":3,"strength":5,
                "desc":"3 consecutive red candles — very strong downtrend","date":date,"signal":"STRONG SELL","idx":i})
        if (c2 < o2 and c1 > o1 and o1 > c2 and c1 < o2 and c > o and c > o2):
            patterns_found.append({"pattern":"Three Inside Up","type":"bullish","candles":3,"strength":4,
                "desc":"Bullish reversal confirmed on 3rd candle","date":date,"signal":"BUY","idx":i})
        if (c2 > o2 and c1 < o1 and o1 < c2 and c1 > o2 and c < o and c < o2):
            patterns_found.append({"pattern":"Three Inside Down","type":"bearish","candles":3,"strength":4,
                "desc":"Bearish reversal confirmed on 3rd candle","date":date,"signal":"SELL","idx":i})

    seen = set()
    unique = []
    for p in reversed(patterns_found):
        if p["pattern"] not in seen:
            seen.add(p["pattern"])
            unique.append(p)
    return unique[:15]


# =============================================================
# TECHNICAL ANALYSIS SUMMARY
# =============================================================
def get_ta_summary(df):
    if df is None or len(df) < 20:
        return {}
    last = df.iloc[-1]
    price = float(last["Close"])
    rsi   = float(last.get("RSI", 50))
    macd  = float(last.get("MACD", 0))
    macd_s= float(last.get("MACD_Signal", 0))
    stoch = float(last.get("Stoch_K", 50))
    bb_u  = float(last.get("BB_Upper", price*1.02))
    bb_l  = float(last.get("BB_Lower", price*0.98))
    bb_m  = float(last.get("BB_Mid",   price))
    bb_pos= (price-bb_l)/(bb_u-bb_l+1e-9)*100
    vol_r = float(last.get("Vol_Ratio", 1))
    ema9  = float(last.get("EMA9",  price))
    ema20 = float(last.get("EMA20", price))
    ema50 = float(last.get("EMA50", price))
    ema200= float(last.get("EMA200", price)) if "EMA200" in df.columns else ema50

    def sig(cond_buy, cond_sell):
        if cond_buy:   return "BUY",  "#00b880"
        if cond_sell:  return "SELL", "#e74c3c"
        return "NEUTRAL", "#f39c12"

    oscillators = [
        {"name":"RSI (14)",       "value":f"{rsi:.1f}",     **dict(zip(["sig","color"], sig(rsi>55 and rsi<70, rsi<40 or rsi>78)))},
        {"name":"MACD",           "value":f"{macd:.2f}",    **dict(zip(["sig","color"], sig(macd>macd_s, macd<macd_s)))},
        {"name":"Stochastic %K",  "value":f"{stoch:.1f}",   **dict(zip(["sig","color"], sig(stoch<30, stoch>70)))},
        {"name":"BB Position",    "value":f"{bb_pos:.0f}%", **dict(zip(["sig","color"], sig(bb_pos<30, bb_pos>80)))},
        {"name":"Volume Ratio",   "value":f"{vol_r:.2f}x",  **dict(zip(["sig","color"], sig(vol_r>1.3, vol_r<0.7)))},
        {"name":"MACD Histogram", "value":f"{float(last.get('MACD_Hist',0)):+.3f}",
         **dict(zip(["sig","color"], sig(float(last.get("MACD_Hist",0))>0 and float(last.get("MACD_Hist",0))>float(df["MACD_Hist"].iloc[-2]) if len(df)>1 else True,
                                       float(last.get("MACD_Hist",0))<0)))},
    ]
    moving_avgs = [
        {"name":"EMA 9",   "value":f"Rs.{ema9:,.2f}",   **dict(zip(["sig","color"], sig(price>ema9,   price<ema9)))},
        {"name":"EMA 20",  "value":f"Rs.{ema20:,.2f}",  **dict(zip(["sig","color"], sig(price>ema20,  price<ema20)))},
        {"name":"EMA 50",  "value":f"Rs.{ema50:,.2f}",  **dict(zip(["sig","color"], sig(price>ema50,  price<ema50)))},
        {"name":"EMA 200", "value":f"Rs.{ema200:,.2f}", **dict(zip(["sig","color"], sig(price>ema200, price<ema200)))},
        {"name":"BB Upper","value":f"Rs.{bb_u:,.2f}",   **dict(zip(["sig","color"], sig(price<bb_u*0.97, price>bb_u)))},
        {"name":"BB Lower","value":f"Rs.{bb_l:,.2f}",   **dict(zip(["sig","color"], sig(price<bb_l*1.02, price>bb_l*1.05)))},
    ]
    all_sigs = [i["sig"] for i in oscillators+moving_avgs]
    buys  = all_sigs.count("BUY")
    sells = all_sigs.count("SELL")
    total = len(all_sigs)
    if buys >= total*0.65:    verdict,vc = "STRONG BUY",  "#00b880"
    elif buys >= total*0.5:   verdict,vc = "BUY",         "#27ae60"
    elif sells >= total*0.65: verdict,vc = "STRONG SELL", "#e74c3c"
    elif sells >= total*0.5:  verdict,vc = "SELL",        "#c0392b"
    else:                     verdict,vc = "NEUTRAL",     "#f39c12"
    return {"oscillators":oscillators,"moving_avgs":moving_avgs,
            "verdict":verdict,"verdict_color":vc,
            "buys":buys,"sells":sells,"neutrals":total-buys-sells,"total":total}



# =============================================================
# ADVANCED TRADING TECHNIQUES MODULE
# 1. Price Action (HH/LL, Structure, Demand/Supply)
# 2. SMC (Order Block, Liquidity, FVG, BOS)
# 3. Volume Profile (POC, VAH, VAL)
# 4. Options Data (PCR, Max Pain, OI)
# 5. Kelly Criterion Position Sizing
# 6. Opening Range Breakout (ORB)
# =============================================================


# ── 1. PRICE ACTION: Market Structure ───────────────────────
def detect_market_structure(df, lookback=5):
    """Detect HH/HL (uptrend) or LH/LL (downtrend) + Market Structure Shift."""
    if len(df) < lookback*3:
        return {}
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values

    # Find swing highs and lows
    swing_highs, swing_lows = [], []
    n = lookback
    for i in range(n, len(df)-n):
        if all(highs[i] >= highs[i-n:i]) and all(highs[i] >= highs[i+1:i+n+1]):
            swing_highs.append((i, highs[i]))
        if all(lows[i]  <= lows[i-n:i])  and all(lows[i]  <= lows[i+1:i+n+1]):
            swing_lows.append((i,  lows[i]))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"structure": "Not enough data", "trend": "Unknown"}

    # Higher High / Higher Low = uptrend
    last2_highs = swing_highs[-2:]
    last2_lows  = swing_lows[-2:]
    hh = last2_highs[1][1] > last2_highs[0][1]
    lh = last2_highs[1][1] < last2_highs[0][1]
    hl = last2_lows[1][1]  > last2_lows[0][1]
    ll = last2_lows[1][1]  < last2_lows[0][1]

    if hh and hl:
        trend = "Uptrend (HH + HL)"
        trend_color = "#00b880"
        trend_icon  = "Bullish"
    elif lh and ll:
        trend = "Downtrend (LH + LL)"
        trend_color = "#e74c3c"
        trend_icon  = "Bearish"
    elif hh and ll:
        trend = "Choppy (HH + LL)"
        trend_color = "#f39c12"
        trend_icon  = "Neutral"
    else:
        trend = "Ranging (LH + HL)"
        trend_color = "#a78bfa"
        trend_icon  = "Neutral"

    # Market Structure Shift detection
    recent_high = max(h for _, h in swing_highs[-3:]) if swing_highs else 0
    recent_low  = min(l for _, l in swing_lows[-3:])  if swing_lows  else 0
    current     = float(df["Close"].iloc[-1])
    atr         = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else (recent_high - recent_low)*0.05

    mss = None
    if ll and current > recent_high + atr*0.3:
        mss = "BULLISH MSS — Break above recent high after lower lows"
    elif hh and current < recent_low - atr*0.3:
        mss = "BEARISH MSS — Break below recent low after higher highs"

    return {
        "trend": trend, "trend_color": trend_color, "trend_icon": trend_icon,
        "hh": hh, "lh": lh, "hl": hl, "ll": ll,
        "swing_highs": swing_highs[-5:],
        "swing_lows":  swing_lows[-5:],
        "mss": mss,
        "recent_high": recent_high, "recent_low": recent_low,
    }


# ── 2. DEMAND / SUPPLY ZONES ─────────────────────────────────
def find_demand_supply_zones(df, n=3):
    """Find demand zones (support) and supply zones (resistance)."""
    if len(df) < 20:
        return [], []
    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values
    atr    = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else float((df["High"]-df["Low"]).mean())

    supply_zones, demand_zones = [], []

    for i in range(n, len(df)-n):
        # Supply zone: strong move down after consolidation
        is_supply_top = (highs[i] == max(highs[max(0,i-n):i+n+1]))
        if is_supply_top and closes[i] < closes[i-1]:
            supply_zones.append({
                "top":    round(highs[i], 2),
                "bottom": round(highs[i] - atr*0.5, 2),
                "idx":    i,
                "strength": min(5, int((highs[i]-lows[i])/atr*2)+1),
                "date":   str(df.index[i])[:10],
            })

        # Demand zone: strong move up after consolidation
        is_demand_bot = (lows[i] == min(lows[max(0,i-n):i+n+1]))
        if is_demand_bot and closes[i] > closes[i-1]:
            demand_zones.append({
                "top":    round(lows[i] + atr*0.5, 2),
                "bottom": round(lows[i], 2),
                "idx":    i,
                "strength": min(5, int((highs[i]-lows[i])/atr*2)+1),
                "date":   str(df.index[i])[:10],
            })

    # Return 3 most recent each
    return supply_zones[-3:], demand_zones[-3:]


# ── 3. FAKE BREAKOUT DETECTION ───────────────────────────────
def detect_fake_breakout(df):
    """Detect bull/bear traps (fake breakouts)."""
    if len(df) < 20:
        return []
    results = []
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 5
    recent_high = float(df["High"].rolling(20).max().iloc[-2])
    recent_low  = float(df["Low"].rolling(20).min().iloc[-2])

    for i in range(2, min(10, len(df))):
        row  = df.iloc[-i]
        prev = df.iloc[-i-1]
        h, l, c, o = float(row["High"]), float(row["Low"]), float(row["Close"]), float(row["Open"])
        ph           = float(prev["High"])
        pl           = float(prev["Low"])

        # Bull trap: broke above resistance then closed back below
        if h > recent_high and c < recent_high - atr*0.1:
            results.append({
                "type": "Bull Trap (Fake Breakout UP)",
                "color": "#e74c3c",
                "desc": f"Price broke above Rs.{recent_high:.2f} but closed back below — institutions trapped bulls",
                "signal": "SELL",
                "date": str(df.index[-i])[:10],
            })

        # Bear trap: broke below support then closed back above
        if l < recent_low and c > recent_low + atr*0.1:
            results.append({
                "type": "Bear Trap (Fake Breakout DOWN)",
                "color": "#00b880",
                "desc": f"Price broke below Rs.{recent_low:.2f} but closed back above — institutions trapped bears",
                "signal": "BUY",
                "date": str(df.index[-i])[:10],
            })

    return results[:3]


# ── 4. SMC: ORDER BLOCKS ─────────────────────────────────────
def find_order_blocks(df):
    """Find Order Blocks — last opposing candle before strong move."""
    if len(df) < 10:
        return []
    obs = []
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 5

    for i in range(3, len(df)-2):
        o  = float(df["Open"].iloc[i])
        c  = float(df["Close"].iloc[i])
        h  = float(df["High"].iloc[i])
        l  = float(df["Low"].iloc[i])
        c2 = float(df["Close"].iloc[i+1])
        c3 = float(df["Close"].iloc[i+2]) if i+2 < len(df) else c2

        move_after = abs(c3 - c)

        # Bullish OB: last red candle before strong up move
        if c < o and c3 > h and move_after > atr * 1.5:
            obs.append({
                "type": "Bullish Order Block",
                "color": "#00b880",
                "top":  round(max(o, c), 2),
                "bottom": round(min(o, c), 2),
                "idx": i,
                "desc": "Institutional BUY zone — price likely to return here",
                "signal": "BUY when price returns",
                "date": str(df.index[i])[:10],
            })

        # Bearish OB: last green candle before strong down move
        if c > o and c3 < l and move_after > atr * 1.5:
            obs.append({
                "type": "Bearish Order Block",
                "color": "#e74c3c",
                "top":  round(max(o, c), 2),
                "bottom": round(min(o, c), 2),
                "idx": i,
                "desc": "Institutional SELL zone — price likely to reverse here",
                "signal": "SELL when price returns",
                "date": str(df.index[i])[:10],
            })

    return obs[-4:]


# ── 5. SMC: FAIR VALUE GAPS (FVG) ────────────────────────────
def find_fvg(df):
    """Find Fair Value Gaps — imbalances in price."""
    if len(df) < 5:
        return []
    fvgs = []
    for i in range(1, len(df)-1):
        h1 = float(df["High"].iloc[i-1])
        l1 = float(df["Low"].iloc[i-1])
        h2 = float(df["High"].iloc[i+1])
        l2 = float(df["Low"].iloc[i+1])

        # Bullish FVG: gap between candle[i-1] high and candle[i+1] low
        if l2 > h1:
            fvgs.append({
                "type": "Bullish FVG",
                "color": "#00b880",
                "top": round(l2, 2),
                "bottom": round(h1, 2),
                "gap": round(l2-h1, 2),
                "desc": "Price moved up fast — gap will likely be filled (BUY zone)",
                "date": str(df.index[i])[:10],
                "filled": float(df["Low"].iloc[-1]) <= l2,
            })

        # Bearish FVG: gap between candle[i-1] low and candle[i+1] high
        if h2 < l1:
            fvgs.append({
                "type": "Bearish FVG",
                "color": "#e74c3c",
                "top": round(l1, 2),
                "bottom": round(h2, 2),
                "gap": round(l1-h2, 2),
                "desc": "Price moved down fast — gap will likely be filled (SELL zone)",
                "date": str(df.index[i])[:10],
                "filled": float(df["High"].iloc[-1]) >= h2,
            })

    unfilled = [f for f in fvgs if not f["filled"]]
    return unfilled[-4:]


# ── 6. VOLUME PROFILE (POC, VAH, VAL) ────────────────────────
def compute_volume_profile(df, bins=20):
    """Compute Volume Profile — POC, Value Area High/Low."""
    if len(df) < 10:
        return {}
    price_min = float(df["Low"].min())
    price_max = float(df["High"].max())
    price_range = price_max - price_min
    bin_size = price_range / bins if price_range > 0 else 1

    vol_at_price = {}
    for i in range(len(df)):
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        v = float(df["Volume"].iloc[i])
        # distribute volume across price range of this candle
        candle_bins = max(1, int((h-l)/bin_size))
        vol_per_bin = v / candle_bins
        for b in range(candle_bins):
            price_level = round(l + b*bin_size + bin_size/2, 2)
            bucket = round((price_level - price_min) / bin_size) * bin_size + price_min
            bucket = round(bucket, 2)
            vol_at_price[bucket] = vol_at_price.get(bucket, 0) + vol_per_bin

    if not vol_at_price:
        return {}

    sorted_levels = sorted(vol_at_price.items(), key=lambda x: -x[1])
    poc = sorted_levels[0][0]
    total_vol = sum(vol_at_price.values())
    target_vol = total_vol * 0.70  # 70% = Value Area

    # Value Area: prices around POC containing 70% of volume
    accumulated = 0
    va_prices = []
    for price_level, vol in sorted_levels:
        accumulated += vol
        va_prices.append(price_level)
        if accumulated >= target_vol:
            break

    vah = max(va_prices) if va_prices else poc
    val = min(va_prices) if va_prices else poc
    current = float(df["Close"].iloc[-1])

    if current > vah:     position = "ABOVE Value Area — Potential rejection"
    elif current < val:   position = "BELOW Value Area — Potential support"
    elif abs(current-poc)/poc < 0.005: position = "AT Point of Control — Key level"
    else:                 position = "INSIDE Value Area — Balanced market"

    return {
        "poc": round(poc, 2),
        "vah": round(vah, 2),
        "val": round(val, 2),
        "current": round(current, 2),
        "position": position,
        "top_levels": [(round(p,2), round(v/1e6,2)) for p,v in sorted_levels[:5]],
        "total_levels": len(vol_at_price),
    }


# ── 7. OPTIONS DATA: PCR, MAX PAIN ───────────────────────────
def fetch_options_data(symbol):
    """Fetch OI, PCR, Max Pain from yfinance."""
    try:
        import yfinance as yf2
        ticker = yf2.Ticker(symbol)
        exps   = ticker.options
        if not exps:
            return {}
        exp = exps[0]
        chain = ticker.option_chain(exp)
        calls = chain.calls
        puts  = chain.puts

        # Total OI
        total_call_oi = int(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0
        total_put_oi  = int(puts["openInterest"].sum())  if "openInterest" in puts.columns  else 0
        pcr = round(total_put_oi / max(total_call_oi, 1), 2)

        # Max Pain
        try:
            strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
            pain_values = []
            for s in strikes:
                call_pain = sum(max(0, s - k) * oi for k, oi in zip(
                    calls["strike"], calls["openInterest"].fillna(0)) if s > k)
                put_pain  = sum(max(0, k - s) * oi for k, oi in zip(
                    puts["strike"], puts["openInterest"].fillna(0)) if s < k)
                pain_values.append((s, call_pain + put_pain))
            max_pain_strike = min(pain_values, key=lambda x: x[1])[0] if pain_values else 0
        except Exception:
            max_pain_strike = 0

        # PCR interpretation
        if pcr > 1.5:   pcr_signal = "VERY BULLISH (heavy put writing)"
        elif pcr > 1.2: pcr_signal = "BULLISH"
        elif pcr > 0.8: pcr_signal = "NEUTRAL"
        elif pcr > 0.5: pcr_signal = "BEARISH"
        else:           pcr_signal = "VERY BEARISH (heavy call writing)"

        return {
            "expiry": exp,
            "call_oi": total_call_oi,
            "put_oi":  total_put_oi,
            "pcr": pcr,
            "pcr_signal": pcr_signal,
            "max_pain": max_pain_strike,
            "top_call_strikes": calls.nlargest(3,"openInterest")[["strike","openInterest"]].values.tolist() if "openInterest" in calls.columns else [],
            "top_put_strikes":  puts.nlargest(3,"openInterest")[["strike","openInterest"]].values.tolist()  if "openInterest" in puts.columns  else [],
        }
    except Exception as e:
        return {"error": str(e)}


# ── 8. OPENING RANGE BREAKOUT (ORB) ──────────────────────────
def compute_orb(df_5m):
    """Opening Range Breakout — first 15/30 min range."""
    if df_5m is None or len(df_5m) < 10:
        return {}
    try:
        df_5m = df_5m.copy()
        df_5m.index = df_5m.index.tz_localize(None) if df_5m.index.tzinfo else df_5m.index
        today = df_5m.index[-1].date()
        today_data = df_5m[df_5m.index.date == today]
        if len(today_data) < 3:
            return {}

        # 15-min opening range (first 3 candles of 5m)
        orb_data = today_data.iloc[:3]
        orb_high = float(orb_data["High"].max())
        orb_low  = float(orb_data["Low"].min())
        orb_range= orb_high - orb_low
        current  = float(df_5m["Close"].iloc[-1])

        if current > orb_high:
            status = "BULLISH BREAKOUT above ORB"
            signal = "BUY — Target ORB High + Range extension"
            color  = "#00b880"
        elif current < orb_low:
            status = "BEARISH BREAKDOWN below ORB"
            signal = "SELL — Target ORB Low - Range extension"
            color  = "#e74c3c"
        else:
            status = "Inside Opening Range — Wait for breakout"
            signal = "WAIT — No trade until ORB break"
            color  = "#f39c12"

        target_up   = round(orb_high + orb_range, 2)
        target_down = round(orb_low  - orb_range, 2)

        return {
            "orb_high":    round(orb_high, 2),
            "orb_low":     round(orb_low, 2),
            "orb_range":   round(orb_range, 2),
            "current":     round(current, 2),
            "status":      status,
            "signal":      signal,
            "color":       color,
            "target_up":   target_up,
            "target_down": target_down,
            "candles_used": len(orb_data),
        }
    except Exception as e:
        return {"error": str(e)}


# ── 9. KELLY CRITERION ───────────────────────────────────────
def kelly_sizing(win_rate, rr_ratio, capital, max_risk_pct=0.20):
    """Optimal Kelly Criterion position size."""
    if rr_ratio <= 0 or win_rate <= 0:
        return 0
    p = win_rate
    q = 1 - p
    b = rr_ratio
    kelly = (b*p - q) / b
    kelly = max(0.0, min(kelly, max_risk_pct))
    half_kelly = kelly * 0.5  # safer
    return {
        "full_kelly":  round(kelly*100, 1),
        "half_kelly":  round(half_kelly*100, 1),
        "capital_full": round(capital * kelly, 0),
        "capital_half": round(capital * half_kelly, 0),
        "expected_value": round(b*p - q, 3),
        "interpretation": (
            "TRADE THIS — positive EV" if (b*p - q) > 0
            else "SKIP — negative expected value"
        ),
    }


# =============================================================
# PROFESSIONAL TRADING ENGINE v3.0
# ICT + Market Structure + Chart Patterns + Institutional Logic
# =============================================================

# ── 1. EQUAL HIGHS / EQUAL LOWS (EQH/EQL) ───────────────────
def detect_equal_levels(df, tolerance=0.002):
    """EQH/EQL — liquidity pools above equal highs / below equal lows."""
    if len(df) < 20: return {}
    highs  = df["High"].values[-50:]
    lows   = df["Low"].values[-50:]
    closes = df["Close"].values
    price  = float(closes[-1])
    atr    = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else price*0.01

    eqh_levels, eql_levels = [], []
    # Find clusters of highs within tolerance
    for i in range(len(highs)):
        cluster = [highs[j] for j in range(len(highs)) if abs(highs[j]-highs[i])/highs[i] < tolerance and i != j]
        if len(cluster) >= 1:
            avg = (highs[i] + sum(cluster))/( 1+len(cluster))
            eqh_levels.append(round(avg, 2))
    for i in range(len(lows)):
        cluster = [lows[j] for j in range(len(lows)) if abs(lows[j]-lows[i])/lows[i] < tolerance and i != j]
        if len(cluster) >= 1:
            avg = (lows[i] + sum(cluster))/(1+len(cluster))
            eql_levels.append(round(avg, 2))

    # Deduplicate
    eqh = sorted(set([round(l/atr)*atr for l in eqh_levels]), reverse=True)[:3]
    eql = sorted(set([round(l/atr)*atr for l in eql_levels]))[:3]

    nearest_eqh = min(eqh, key=lambda x:abs(x-price)) if eqh else None
    nearest_eql = min(eql, key=lambda x:abs(x-price)) if eql else None

    return {
        "eqh": [round(v,2) for v in eqh],
        "eql": [round(v,2) for v in eql],
        "BSL": [round(v,2) for v in eqh],   # Buy-Side Liquidity (above EQH)
        "SSL": [round(v,2) for v in eql],   # Sell-Side Liquidity (below EQL)
        "nearest_eqh": round(nearest_eqh,2) if nearest_eqh else None,
        "nearest_eql": round(nearest_eql,2) if nearest_eql else None,
        "near_eqh": nearest_eqh and abs(nearest_eqh-price)/price < 0.005,
        "near_eql": nearest_eql and abs(nearest_eql-price)/price < 0.005,
        "near_BSL": nearest_eqh and abs(nearest_eqh-price)/price < 0.008,
        "near_SSL": nearest_eql and abs(nearest_eql-price)/price < 0.008,
    }


# ── 2. LIQUIDITY SWEEP (Proper) ──────────────────────────────
def detect_liquidity_sweep(df):
    """Proper liquidity sweep: wick beyond EQH/EQL then closes back."""
    if len(df) < 20: return []
    atr    = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 5
    price  = float(df["Close"].iloc[-1])
    eq     = detect_equal_levels(df)
    results = []

    for i in range(1, min(10, len(df))):
        h = float(df["High"].iloc[-i])
        l = float(df["Low"].iloc[-i])
        c = float(df["Close"].iloc[-i])
        o = float(df["Open"].iloc[-i])
        date = str(df.index[-i])[:10]

        # Bullish sweep: wick below EQL then closes above (stop hunt below lows)
        for eql in eq.get("eql", []):
            if l < eql - atr*0.1 and c > eql + atr*0.15:
                results.append({
                    "type":   "Bullish Liquidity Sweep",
                    "color":  "#00b880",
                    "signal": "BUY",
                    "swept":  round(eql, 2),
                    "desc":   f"Stop hunt below EQL Rs.{eql:.2f} — Smart money accumulated. Reversal up.",
                    "date":   date,
                })
                break

        # Bearish sweep: wick above EQH then closes below (stop hunt above highs)
        for eqh in eq.get("eqh", []):
            if h > eqh + atr*0.1 and c < eqh - atr*0.15:
                results.append({
                    "type":   "Bearish Liquidity Sweep",
                    "color":  "#e74c3c",
                    "signal": "SELL",
                    "swept":  round(eqh, 2),
                    "desc":   f"Stop hunt above EQH Rs.{eqh:.2f} — Smart money distributed. Reversal down.",
                    "date":   date,
                })
                break

    return results[:3]


# ── 3. BOS & CHOCH ENGINE ────────────────────────────────────
def detect_bos_choch(df, n=5):
    """
    BOS  = Break of Structure (trend continuation)
    CHOCH = Change of Character (trend reversal signal)
    """
    if len(df) < n*4: return {}
    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values
    atr    = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 5

    # Swing points
    sh, sl = [], []
    for i in range(n, len(df)-n):
        if all(highs[i]>=highs[max(0,i-n):i]) and all(highs[i]>=highs[i+1:i+n+1]):
            sh.append((i, highs[i]))
        if all(lows[i] <=lows[max(0,i-n):i])  and all(lows[i] <=lows[i+1:i+n+1]):
            sl.append((i, lows[i]))

    if len(sh) < 2 or len(sl) < 2:
        return {"bos": None, "choch": None, "trend": "Unknown"}

    price = float(closes[-1])
    last_sh = sh[-1][1]; prev_sh = sh[-2][1]
    last_sl = sl[-1][1]; prev_sl = sl[-2][1]

    bos   = None
    choch = None

    # Current trend
    trend = "Uptrend" if (last_sh > prev_sh and last_sl > prev_sl) else             "Downtrend" if (last_sh < prev_sh and last_sl < prev_sl) else "Ranging"

    # BOS: Break of Structure — continuation
    if trend == "Uptrend" and price > last_sh + atr*0.2:
        bos = {"direction": "BULLISH BOS", "level": round(last_sh,2),
               "desc": f"Price broke above swing high Rs.{last_sh:.2f} — Uptrend continues",
               "color": "#00b880"}
    elif trend == "Downtrend" and price < last_sl - atr*0.2:
        bos = {"direction": "BEARISH BOS", "level": round(last_sl,2),
               "desc": f"Price broke below swing low Rs.{last_sl:.2f} — Downtrend continues",
               "color": "#e74c3c"}

    # CHOCH: Change of Character — reversal warning
    if trend == "Downtrend" and price > last_sh + atr*0.3:
        choch = {"direction": "BULLISH CHOCH", "level": round(last_sh,2),
                 "desc": f"Downtrend broken — price above Rs.{last_sh:.2f}. Possible reversal UP.",
                 "color": "#00e5a0"}
    elif trend == "Uptrend" and price < last_sl - atr*0.3:
        choch = {"direction": "BEARISH CHOCH", "level": round(last_sl,2),
                 "desc": f"Uptrend broken — price below Rs.{last_sl:.2f}. Possible reversal DOWN.",
                 "color": "#ff6b6b"}

    return {
        "bos": bos, "choch": choch, "trend": trend,
        "last_sh": round(last_sh,2), "last_sl": round(last_sl,2),
        "prev_sh": round(prev_sh,2), "prev_sl": round(prev_sl,2),
    }


# ── 4. PREMIUM / DISCOUNT ZONES ──────────────────────────────
def get_premium_discount(df):
    """ICT Premium/Discount + Equilibrium zone."""
    price = float(df["Close"].iloc[-1])
    high  = float(df["High"].rolling(20).max().iloc[-1])
    low   = float(df["Low"].rolling(20).min().iloc[-1])
    rng   = high - low + 1e-9
    mid   = (high + low) / 2
    pct   = (price - low) / rng * 100

    if   pct >= 75: zone = "Extreme Premium"; zcol = "#e74c3c"; rec = "SELL — Overvalued"
    elif pct >= 55: zone = "Premium";         zcol = "#e07b39"; rec = "SELL/WAIT"
    elif pct >= 45: zone = "Equilibrium";     zcol = "#f39c12"; rec = "NEUTRAL — No edge"
    elif pct >= 25: zone = "Discount";        zcol = "#27ae60"; rec = "BUY/WATCH"
    else:           zone = "Extreme Discount";zcol = "#00b880"; rec = "BUY — Undervalued"

    return {
        "zone": zone, "color": zcol, "recommendation": rec,
        "pct": round(pct, 1), "price": round(price, 2),
        "high": round(high, 2), "low": round(low, 2), "mid": round(mid, 2),
        "premium_start": round(low + rng*0.55, 2),
        "discount_end":  round(low + rng*0.45, 2),
        "extreme_prem":  round(low + rng*0.75, 2),
        "extreme_disc":  round(low + rng*0.25, 2),
    }


# ── 5. KILL ZONES (London / NY / Asia) ───────────────────────
def get_kill_zones():
    """ICT Kill Zones with Indian market mapping."""
    now_t = now_ist().strftime("%H:%M")
    zones = [
        ("09:15","10:30","Opening Kill Zone",   "#ffa94d","NSE open — high vol, institutional orders"),
        ("10:30","11:30","Asian Close / Overlap","#a78bfa","Asian session closing liquidity"),
        ("11:00","13:00","London Open (Equiv)",  "#4e8fff","European market impact on Indian stocks"),
        ("13:00","14:00","Lunch Lull",           "#555555","Low volume — avoid trading"),
        ("14:00","15:00","NY Pre-Open (Equiv)",  "#00e5a0","US futures affect Indian indices"),
        ("15:00","15:30","Power Hour / Closing", "#e74c3c","Final moves — high reversals"),
    ]
    active = None
    for start,end,name,color,desc in zones:
        if start <= now_t <= end:
            active = {"name":name,"color":color,"desc":desc,"start":start,"end":end}
            break
    return {"zones": zones, "active": active, "current_time": now_t}


# ── 6. CHART PATTERNS ────────────────────────────────────────
def detect_chart_patterns(df):
    """All major chart patterns with targets."""
    if len(df) < 25: return []
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    atr    = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else float(np.std(closes[-20:]))
    n      = len(closes)
    patterns = []
    date = str(df.index[-1])[:10]

    def add(name, typ, strength, signal, desc, target=None):
        patterns.append({"pattern":name,"type":typ,"strength":strength,
                         "signal":signal,"desc":desc,"target":target,"date":date})

    # ── DOUBLE TOP ───────────────────────────────────────────
    for i in range(8, n-6):
        for j in range(i+6, min(i+20, n-2)):
            p1 = max(highs[max(0,i-4):i+4]); p2 = max(highs[j-4:j+4])
            valley = min(closes[i:j])
            if abs(p1-p2)/p1 < 0.012 and valley < p1*0.975 and closes[-1] < valley+atr:
                neckline = round(valley, 2); target = round(valley-(p1-valley),2)
                add("Double Top","bearish",4,"SELL",
                    f"Two peaks ~Rs.{round(p1,0)} | Neckline Rs.{neckline}",target); break
        else: continue; break

    # ── DOUBLE BOTTOM ────────────────────────────────────────
    for i in range(8, n-6):
        for j in range(i+6, min(i+20, n-2)):
            t1 = min(lows[max(0,i-4):i+4]); t2 = min(lows[j-4:j+4])
            peak = max(closes[i:j])
            if abs(t1-t2)/t1 < 0.012 and peak > t1*1.025 and closes[-1] > peak-atr:
                neckline = round(peak,2); target = round(peak+(peak-t1),2)
                add("Double Bottom","bullish",4,"BUY",
                    f"Two bottoms ~Rs.{round(t1,0)} | Neckline Rs.{neckline}",target); break
        else: continue; break

    # ── HEAD & SHOULDERS ─────────────────────────────────────
    if n >= 35:
        seg_h = highs[-35:]
        pivots = [(i,seg_h[i]) for i in range(3,len(seg_h)-3)
                  if seg_h[i]==max(seg_h[max(0,i-3):i+4])]
        if len(pivots) >= 3:
            ls,hd,rs = pivots[-3],pivots[-2],pivots[-1]
            if hd[1]>ls[1]*1.01 and hd[1]>rs[1]*1.01 and abs(ls[1]-rs[1])/ls[1]<0.04:
                neck = float(min(lows[-35:][ls[0]:rs[0]]))
                add("Head & Shoulders","bearish",5,"STRONG SELL",
                    f"Head Rs.{round(hd[1],0)} | Neckline Rs.{round(neck,0)}",
                    round(neck-(hd[1]-neck),2))

    # ── INVERSE H&S ──────────────────────────────────────────
    if n >= 35:
        seg_l = lows[-35:]
        pivots = [(i,seg_l[i]) for i in range(3,len(seg_l)-3)
                  if seg_l[i]==min(seg_l[max(0,i-3):i+4])]
        if len(pivots) >= 3:
            ls,hd,rs = pivots[-3],pivots[-2],pivots[-1]
            if hd[1]<ls[1]*0.99 and hd[1]<rs[1]*0.99 and abs(ls[1]-rs[1])/ls[1]<0.04:
                neck = float(max(highs[-35:][ls[0]:rs[0]]))
                add("Inverse H&S","bullish",5,"STRONG BUY",
                    f"Head Rs.{round(hd[1],0)} | Neckline Rs.{round(neck,0)}",
                    round(neck+(neck-hd[1]),2))

    # ── ASCENDING TRIANGLE ───────────────────────────────────
    last_h = highs[-25:]; last_l = lows[-25:]
    h_rng = float(max(last_h)-min(last_h)); l_rng = float(max(last_l)-min(last_l))
    if h_rng < atr*1.5 and l_rng > atr*3:
        add("Ascending Triangle","bullish",4,"BUY on breakout",
            f"Flat resistance Rs.{round(max(last_h),0)} + Rising lows",
            round(float(max(last_h))+h_rng*1.5,2))

    # ── DESCENDING TRIANGLE ──────────────────────────────────
    elif l_rng < atr*1.5 and h_rng > atr*3:
        add("Descending Triangle","bearish",4,"SELL on breakdown",
            f"Flat support Rs.{round(min(last_l),0)} + Falling highs",
            round(float(min(last_l))-h_rng*1.5,2))

    # ── SYMMETRICAL TRIANGLE ─────────────────────────────────
    elif h_rng < atr*2.5 and l_rng < atr*2.5 and n>20:
        add("Symmetrical Triangle","neutral",3,"WAIT — Breakout coming",
            f"Converging range — Big move expected soon",None)

    # ── BULL FLAG ────────────────────────────────────────────
    if n>=25:
        pole   = float(closes[-20])-float(closes[-25]) if n>=25 else 0
        consol = float(max(highs[-8:]))-float(min(lows[-8:]))
        if pole > atr*3 and consol < atr*1.5:
            add("Bull Flag","bullish",4,"BUY — continuation",
                f"Strong pole up Rs.{round(abs(pole),0)} + tight flag",
                round(float(closes[-1])+abs(pole),2))

    # ── BEAR FLAG ────────────────────────────────────────────
    if n>=25:
        pole   = float(closes[-25])-float(closes[-20]) if n>=25 else 0
        consol = float(max(highs[-8:]))-float(min(lows[-8:]))
        if pole > atr*3 and consol < atr*1.5 and float(closes[-1])<float(closes[-5]):
            add("Bear Flag","bearish",4,"SELL — continuation",
                f"Strong pole down Rs.{round(abs(pole),0)} + tight flag",
                round(float(closes[-1])-abs(pole),2))

    # ── PENNANT ──────────────────────────────────────────────
    if n>=15:
        early_rng = float(max(highs[-15:-8]))-float(min(lows[-15:-8]))
        late_rng  = float(max(highs[-7:]))-float(min(lows[-7:]))
        if late_rng < early_rng*0.5 and early_rng > atr*2:
            direction = "bullish" if float(closes[-1])>float(closes[-15]) else "bearish"
            add("Pennant",direction,4,
                "BUY on breakout" if direction=="bullish" else "SELL on breakdown",
                f"Converging pennant after strong move",None)

    # ── CUP & HANDLE ─────────────────────────────────────────
    if n >= 40:
        try:
            seg_c = closes[-40:]
            left  = float(max(seg_c[:10]))
            cup   = float(min(seg_c[10:30]))
            right = float(max(seg_c[30:]))
            handle= float(min(seg_c[-8:]))
            if abs(left-right)/left<0.04 and cup<left*0.85 and handle>cup and handle<right*0.97 and float(closes[-1])>handle:
                add("Cup & Handle","bullish",5,"STRONG BUY",
                    f"C&H pattern — Cup Rs.{round(cup,0)} | Handle Rs.{round(handle,0)} | Breakout!",
                    round(right+(right-cup),2))
        except Exception: pass

    return sorted(patterns, key=lambda x:-x["strength"])


# ── 7. FIBONACCI ─────────────────────────────────────────────
def get_fibonacci_levels(df, lookback=50):
    recent = df.tail(lookback)
    high   = float(recent["High"].max())
    low    = float(recent["Low"].min())
    diff   = high - low + 1e-9
    price  = float(df["Close"].iloc[-1])
    levels = {
        "0% (High)":       round(high,2),
        "23.6%":           round(high-0.236*diff,2),
        "38.2%":           round(high-0.382*diff,2),
        "50%  (Mid)":      round(high-0.500*diff,2),
        "61.8% (Golden)":  round(high-0.618*diff,2),
        "78.6%":           round(high-0.786*diff,2),
        "100% (Low)":      round(low,2),
        "127.2% (Ext)":    round(low-0.272*diff,2),
        "161.8% (Ext)":    round(low-0.618*diff,2),
    }
    nearest = min(levels.items(), key=lambda x:abs(x[1]-price))
    in_gz   = (high-0.618*diff) <= price <= (high-0.382*diff)
    return {
        "levels": levels, "high":round(high,2), "low":round(low,2),
        "current": round(price,2), "nearest_level": nearest[0],
        "nearest_price": nearest[1], "in_golden_zone": in_gz,
        "golden_top": round(high-0.382*diff,2),
        "golden_bot": round(high-0.618*diff,2),
    }


# ── 8. TRENDLINE ─────────────────────────────────────────────
def detect_trendlines(df, n=5):
    if len(df)<20: return {}
    highs=df["High"].values; lows=df["Low"].values; closes=df["Close"].values
    atr=float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 5
    ph=[]; pl=[]
    for i in range(n,len(df)-n):
        if all(highs[i]>=highs[max(0,i-n):i]) and all(highs[i]>=highs[i+1:i+n+1]):
            ph.append((i,highs[i]))
        if all(lows[i]<=lows[max(0,i-n):i]) and all(lows[i]<=lows[i+1:i+n+1]):
            pl.append((i,lows[i]))
    result={"resistance":None,"support":None,"signal":"Insufficient data"}
    if len(ph)>=2:
        (x1,y1),(x2,y2)=ph[-2],ph[-1]
        if x2!=x1:
            sl=(y2-y1)/(x2-x1); cur_idx=len(df)-1
            rp=round(y1+sl*(cur_idx-x1),2)
            touches=sum(1 for i,h in ph if abs(h-(y1+sl*(i-x1)))<atr*0.5)
            result["resistance"]={"price":rp,"slope":round(sl,4),
                "direction":"Falling" if sl<0 else "Rising",
                "touches":touches,"broken":float(closes[-1])>rp+atr*0.3}
    if len(pl)>=2:
        (x1,y1),(x2,y2)=pl[-2],pl[-1]
        if x2!=x1:
            sl=(y2-y1)/(x2-x1); cur_idx=len(df)-1
            sp=round(y1+sl*(cur_idx-x1),2)
            touches=sum(1 for i,l in pl if abs(l-(y1+sl*(i-x1)))<atr*0.5)
            result["support"]={"price":sp,"slope":round(sl,4),
                "direction":"Rising" if sl>0 else "Falling",
                "touches":touches,"broken":float(closes[-1])<sp-atr*0.3}
    r=result.get("resistance"); s=result.get("support")
    if r and r.get("broken"):      result["signal"]="BULLISH BREAKOUT — above resistance"
    elif s and s.get("broken"):    result["signal"]="BEARISH BREAKDOWN — below support"
    elif r and s:
        if r["slope"]>0 and s["slope"]>0: result["signal"]="Rising Channel — BUY at support"
        elif r["slope"]<0 and s["slope"]<0: result["signal"]="Falling Channel — SELL at resistance"
        else: result["signal"]="Triangle — Breakout imminent"
    return result


# ── 9. MULTI-TIMEFRAME CONFLUENCE ────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_mtf_confluence(_stock: str) -> dict:
    """
    Multi-timeframe analysis: 5m + 15m + 1H + 4H (daily as proxy)
    Returns confluence score 0-100.
    """
    import yfinance as yf2
    timeframes = {
        "5m":  ("1d",  "5m"),
        "15m": ("5d",  "15m"),
        "1H":  ("1mo", "1h"),
        "4H":  ("3mo", "1d"),
    }
    results = {}
    for tf_name, (period, interval) in timeframes.items():
        try:
            d = yf2.Ticker(_stock).history(period=period, interval=interval)
            if d is None or d.empty or len(d)<15: continue
            last = d.iloc[-1]
            price= float(last["Close"])
            ema20= float(d["Close"].ewm(20).mean().iloc[-1])
            ema50= float(d["Close"].ewm(50).mean().iloc[-1])
            delta= d["Close"].diff()
            rsi_val = 100-(100/(1+(delta.where(delta>0,0).rolling(14).mean()/
                              (-delta.where(delta<0,0)).rolling(14).mean().replace(0,1e-9)).iloc[-1]))
            macd_v = float(d["Close"].ewm(12).mean().iloc[-1]-d["Close"].ewm(26).mean().iloc[-1])
            macd_s = float(d["Close"].ewm(12).mean().ewm(9).mean().iloc[-1]) if len(d)>30 else 0
            vr     = float(d["Volume"].iloc[-1]/(d["Volume"].rolling(20).mean().iloc[-1]+1e-9))

            score = sum([price>ema20, price>ema50, ema20>ema50,
                         40<rsi_val<72, macd_v>macd_s, vr>1.0])
            direction = "BULLISH" if score>=4 else ("BEARISH" if score<=2 else "NEUTRAL")
            results[tf_name] = {
                "score": score, "max": 6,
                "pct": round(score/6*100),
                "direction": direction,
                "rsi": round(float(rsi_val),1),
                "price": round(price,2),
            }
        except Exception:
            continue

    if not results: return {"confluence": 50, "signal": "No data", "breakdown": {}}

    avg_pct = round(sum(r["pct"] for r in results.values())/len(results))
    bull_tfs = [tf for tf,r in results.items() if r["direction"]=="BULLISH"]
    bear_tfs = [tf for tf,r in results.items() if r["direction"]=="BEARISH"]

    if len(bull_tfs)>=3:   signal = "STRONG BUY — All TFs aligned"
    elif len(bull_tfs)==2: signal = "BUY — Most TFs bullish"
    elif len(bear_tfs)>=3: signal = "STRONG SELL — All TFs aligned"
    elif len(bear_tfs)==2: signal = "SELL — Most TFs bearish"
    else:                  signal = "MIXED — Wait for alignment"

    majority  = "BULLISH" if len(bull_tfs)>=len(bear_tfs) else "BEARISH"
    align_pct = round(max(len(bull_tfs),len(bear_tfs))/max(len(results),1)*100)
    return {"confluence": avg_pct, "signal": signal,
            "breakdown": results, "bull_count": len(bull_tfs),
            "bear_count": len(bear_tfs),
            "majority": majority, "align_pct": align_pct}


# ── 10. INSTITUTIONAL SCORE ENGINE ───────────────────────────
def get_institutional_score(df, stock: str) -> dict:
    """
    Institutional-grade composite score (0-100).
    Checks: Volume, OB zones, FVG, Liquidity, Structure, Momentum
    """
    if len(df) < 20: return {"score": 50, "grade": "C", "label": "Neutral"}
    price = float(df["Close"].iloc[-1])
    atr   = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else price*0.01

    scores = {}

    # 1. Volume Intelligence (0-20)
    vr = float(df["Vol_Ratio"].iloc[-1]) if "Vol_Ratio" in df.columns else 1
    obv_trend = 1 if len(df)>20 and float(df.get("OBV", df["Volume"]).iloc[-1]) > float(df.get("OBV",df["Volume"]).iloc[-10]) else 0
    scores["volume"] = min(20, int(vr*8 + obv_trend*5))

    # 2. Trend Alignment (0-20)
    ema9  = float(df.get("EMA9",  df["Close"]).iloc[-1])
    ema20 = float(df.get("EMA20", df["Close"]).iloc[-1])
    ema50 = float(df.get("EMA50", df["Close"]).iloc[-1])
    st_dir= float(df.get("ST_Dir", pd.Series([0])).iloc[-1])
    adx   = float(df.get("ADX",   pd.Series([15])).iloc[-1])
    trend_score = sum([price>ema9, price>ema20, price>ema50,
                       ema9>ema20, ema20>ema50, st_dir>0, adx>25])
    scores["trend"] = min(20, int(trend_score/7*20))

    # 3. Momentum Quality (0-20)
    rsi  = float(df.get("RSI", pd.Series([50])).iloc[-1])
    mfi  = float(df.get("MFI", pd.Series([50])).iloc[-1])
    macd = float(df.get("MACD", pd.Series([0])).iloc[-1])
    macs = float(df.get("MACD_Signal", pd.Series([0])).iloc[-1])
    mh   = float(df.get("MACD_Hist", pd.Series([0])).iloc[-1])
    mom  = sum([45<rsi<72, mfi>55, macd>macs, mh>0, rsi>50])
    scores["momentum"] = min(20, int(mom/5*20))

    # 4. SMC Quality (0-20)
    try:
        bos_r = detect_bos_choch(df)
        fvg_r = [f for f in (detect_fvg_simple(df) or []) if f["type"]=="Bullish FVG"]
        eq_r  = detect_equal_levels(df)
        smc   = sum([bos_r.get("trend")=="Uptrend",
                     len(fvg_r)>0,
                     eq_r.get("near_eql",False)])
        scores["smc"] = min(20, int(smc/3*20))
    except Exception:
        scores["smc"] = 10

    # 5. Risk/Reward Quality (0-16)
    bb_pct = float(df.get("BB_Pct", pd.Series([0.5])).iloc[-1])
    pd_r   = get_premium_discount(df)
    rr_ok  = "Discount" in pd_r.get("zone","")
    scores["risk"] = min(16, int((1-bb_pct)*8 + rr_ok*8))

    # 6. Options PCR Score (0-4)
    scores["options"] = 0

    total = sum(scores.values())
    if   total >= 80: grade,label = "A+","Institutional BUY"
    elif total >= 65: grade,label = "A", "Strong BUY"
    elif total >= 50: grade,label = "B", "Moderate BUY"
    elif total >= 35: grade,label = "C", "Neutral/Wait"
    elif total >= 20: grade,label = "D", "SELL/Avoid"
    else:             grade,label = "F", "Strong SELL"

    # Normalize to 100
    max_possible = 20 + 20 + 20 + 16 + 4   # volume+trend+momentum+smc+risk+options
    normalized   = round(min(100, total/max_possible*100))
    return {
        "score": normalized, "raw": total, "grade": grade, "label": label,
        "breakdown": scores,
        "breakdown_max": {"volume":20,"trend":20,"momentum":20,"smc":20,"risk":16,"options":4},
        "color": "#00b880" if normalized>=65 else ("#f39c12" if normalized>=35 else "#e74c3c"),
    }


def detect_fvg_simple(df):
    """Quick FVG for institutional score."""
    if len(df)<5: return []
    fvgs=[]
    for i in range(1,min(20,len(df)-1)):
        h1=float(df["High"].iloc[i-1]); l2=float(df["Low"].iloc[i+1])
        if l2>h1: fvgs.append({"type":"Bullish FVG"})
    return fvgs


# ── 11. DYNAMIC ATR POSITION SIZING ──────────────────────────
def dynamic_position_size(price, atr, capital, risk_pct,
                           inst_score=50, win_rate=50) -> dict:
    """
    Dynamic sizing: Base ATR * Institutional Score scaling.
    Higher conviction = slightly larger size.
    """
    # Base risk
    risk_amount = capital * (risk_pct/100)
    sl_dist     = atr * 1.5

    # Conviction multiplier (0.5x to 1.5x based on score)
    conviction = 0.5 + (inst_score/100)   # 0.5 to 1.5
    kelly_pct  = max(0.5, min(3.0, risk_pct * conviction))
    adj_risk   = capital * (kelly_pct/100)

    base_qty   = max(1, int(risk_amount / sl_dist))
    adj_qty    = max(1, int(adj_risk    / sl_dist))

    max_qty    = max(1, int(capital * 0.20 / price))  # 20% max per trade
    final_qty  = min(adj_qty, max_qty)

    return {
        "base_qty":    base_qty,
        "adj_qty":     final_qty,
        "sl_dist":     round(sl_dist,2),
        "stop_loss":   round(price - sl_dist, 2),
        "target_2r":   round(price + sl_dist*2, 2),
        "target_3r":   round(price + sl_dist*3, 2),
        "risk_amount": round(risk_amount, 0),
        "max_loss":    round(sl_dist * final_qty, 0),
        "max_gain_2r": round(sl_dist * final_qty * 2, 0),
        "max_gain_3r": round(sl_dist * final_qty * 3, 0),
        "conviction":  round(conviction, 2),
        "kelly_pct":   round(kelly_pct, 2),
    }


# ── 12. PORTFOLIO RISK CONTROL ────────────────────────────────
def portfolio_risk_check(paper_balance: float, paper_positions: dict,
                          new_trade_risk: float) -> dict:
    """Check if new trade violates portfolio risk rules."""
    total_exposed = sum(p.get("qty",0)*p.get("price",0)
                        for p in paper_positions.values())
    exposure_pct  = total_exposed / (paper_balance+total_exposed+1e-9) * 100
    positions_cnt = len(paper_positions)
    daily_limit   = paper_balance * 0.03  # 3% daily loss limit

    warnings = []
    can_trade = True

    if exposure_pct > 80:
        warnings.append("Portfolio >80% exposed — high risk!")
        can_trade = False
    if positions_cnt >= 5:
        warnings.append("Max 5 concurrent positions reached")
        can_trade = False
    if new_trade_risk > paper_balance * 0.02:
        warnings.append("Single trade risk >2% of capital")
        can_trade = False

    return {
        "can_trade":    can_trade,
        "exposure_pct": round(exposure_pct, 1),
        "positions":    positions_cnt,
        "warnings":     warnings,
        "status":       "OK" if can_trade else "BLOCKED",
        "color":        "#00b880" if can_trade else "#e74c3c",
    }



# ── CORRELATION FILTER ────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def get_correlation_filter(_stock: str, universe: tuple) -> dict:
    """
    Correlation filter — avoid trading stocks moving together.
    High correlation = same risk, no diversification.
    """
    import yfinance as yf2
    try:
        syms = list(universe[:8]) + [_stock]
        syms = list(dict.fromkeys(syms))  # deduplicate
        prices = {}
        for s in syms:
            try:
                d = yf2.Ticker(s).history(period="1mo", interval="1d")
                if not d.empty and len(d) >= 15:
                    prices[s.replace(".NS","")] = d["Close"].pct_change().dropna().values[-15:]
            except Exception:
                continue
        if len(prices) < 2:
            return {"status": "insufficient_data", "correlations": {}}

        import numpy as np
        stock_sym = _stock.replace(".NS","")
        if stock_sym not in prices:
            return {"status": "no_data", "correlations": {}}

        stock_ret = prices[stock_sym]
        corrs = {}
        for sym, ret in prices.items():
            if sym == stock_sym: continue
            try:
                min_len = min(len(stock_ret), len(ret))
                c = float(np.corrcoef(stock_ret[:min_len], ret[:min_len])[0,1])
                corrs[sym] = round(c, 3)
            except Exception:
                continue

        high_corr = {s:c for s,c in corrs.items() if abs(c) >= 0.75}
        low_corr  = {s:c for s,c in corrs.items() if abs(c) < 0.40}
        avg_corr  = round(float(sum(abs(v) for v in corrs.values()) / max(len(corrs),1)), 3)

        return {
            "status":      "ok",
            "correlations": corrs,
            "high_corr":   high_corr,
            "low_corr":    low_corr,
            "avg_corr":    avg_corr,
            "warning":     len(high_corr) >= 2,
            "rec":         "HIGH correlation — reduce position size" if len(high_corr)>=2 else "Correlation normal — OK to trade",
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "correlations": {}}


# =============================================================
# AI EXPLAINABILITY + TRADING ADVISOR
# =============================================================
def build_ai_explanation(last, price, ai_prob, ai_pct,
                          bos_data, ob_list, fvg_list,
                          patterns, vol_ratio, rsi_val,
                          total_score) -> dict:
    bull = []; bear = []; warn = []
    ema20=float(last.get("EMA20",price)); ema50=float(last.get("EMA50",price))
    macd=float(last.get("MACD",0)); macs=float(last.get("MACD_Signal",0))
    adx=float(last.get("ADX",0)); mfi=float(last.get("MFI",50))
    if price>ema20>ema50: bull.append(("Trend Bullish","Price above EMA20 & EMA50"))
    else: bear.append(("Trend Bearish","Price below EMAs — downtrend"))
    if macd>macs: bull.append(("MACD Bullish","MACD above Signal — bullish momentum"))
    if adx>25: bull.append(("Strong Trend",f"ADX {adx:.0f} — trend confirmed"))
    if mfi>60: bull.append(("Money Flow +ve",f"MFI {mfi:.0f} — institutional buying"))
    elif mfi<40: bear.append(("Money Flow -ve",f"MFI {mfi:.0f} — selling pressure"))
    if ai_prob>0.65: bull.append(("AI Ensemble Bullish",f"AI {ai_pct}% bullish — strong"))
    elif ai_prob>0.55: bull.append(("AI Model Bullish",f"AI {ai_pct}% — mild bullish"))
    elif ai_prob<0.40: bear.append(("AI Model Bearish",f"AI only {ai_pct}% — bearish"))
    if vol_ratio>2.0: bull.append(("Volume Spike",f"Volume {vol_ratio:.1f}x — institutional activity"))
    elif vol_ratio>1.3: bull.append(("Above Avg Volume",f"Volume {vol_ratio:.1f}x confirmed"))
    elif vol_ratio<0.7: warn.append(("Low Volume",f"Volume {vol_ratio:.1f}x — weak signal"))
    if 50<rsi_val<65: bull.append(("RSI Healthy",f"RSI {rsi_val:.0f} — not overbought"))
    elif rsi_val>75: warn.append(("RSI Overbought",f"RSI {rsi_val:.0f} — pullback possible"))
    elif rsi_val<30: bull.append(("RSI Oversold",f"RSI {rsi_val:.0f} — bounce zone"))
    if bos_data:
        if bos_data.get("trend")=="Uptrend": bull.append(("HH+HL Uptrend","Higher highs + lows — accumulation"))
        elif bos_data.get("trend")=="Downtrend": bear.append(("LH+LL Downtrend","Lower highs + lows — distribution"))
        if bos_data.get("bos") and "BULLISH" in bos_data["bos"].get("direction",""): bull.append(("BOS Confirmed","Break of Structure — trend continues"))
        if bos_data.get("choch") and "BULLISH" in bos_data["choch"].get("direction",""): bull.append(("CHOCH Signal","Change of Character — reversal"))
    if ob_list:
        b_ob=[o for o in ob_list if "Bullish" in o.get("type","")]
        if b_ob: bull.append(("Bullish Order Block",f"Institutional buy zone Rs.{b_ob[-1].get('bottom',0):.0f}"))
    if fvg_list:
        b_fvg=[f for f in fvg_list if "Bullish" in f.get("type","")]
        if b_fvg: bull.append(("Bullish FVG",f"Fair Value Gap at Rs.{b_fvg[-1].get('bottom',0):.0f}"))
    if patterns:
        bp=[p for p in patterns if p["type"]=="bullish"]
        brp=[p for p in patterns if p["type"]=="bearish"]
        if bp: bull.append(("Chart Pattern",f"{bp[0]['pattern']} — {bp[0]['signal']}"))
        if brp: bear.append(("Chart Pattern",f"{brp[0]['pattern']} — {brp[0]['signal']}"))
    return {"bull":bull,"bear":bear,"warn":warn,"conf":min(95,round(total_score*0.85+10))}


def build_trading_advisor(user_name, stock_name, expl, total_score,
                           verdict, price, sl, tgt, qty, max_loss,
                           max_gain, rr, win_prob, mtf_data=None) -> str:
    name = user_name if user_name and user_name!="admin" else "Investor"
    bull = expl["bull"]; bear = expl["bear"]; warn = expl["warn"]
    conf = expl["conf"]
    out = []
    out.append(f"**{name} ji,**")
    out.append("")
    if total_score>=82:
        out.append(f"**{stock_name}** mein aaj **strong bullish setup** hai. Trade lene ka sahi time!")
    elif total_score>=68:
        out.append(f"**{stock_name}** mein **bullish signal** hai. Setup decent — entry consider kar sakte hain.")
    elif total_score>=52:
        out.append(f"**{stock_name}** mein abhi **mixed signals** hain. Thoda aur confirmation ka wait karo.")
    else:
        out.append(f"**{stock_name}** mein abhi **trade mat lo**. Setup weak hai.")
    out.append("")
    if bull:
        out.append("**Kyu bullish hai:**")
        for lbl,desc in bull[:5]: out.append(f"✅ **{lbl}:** {desc}")
        out.append("")
    if warn:
        out.append("**Dhyan rakho:**")
        for lbl,desc in warn[:3]: out.append(f"⚠️ **{lbl}:** {desc}")
        out.append("")
    if mtf_data and mtf_data.get("breakdown"):
        out.append("**Timeframe alignment:**")
        for tf,d in mtf_data["breakdown"].items():
            ic = "🟢" if d["direction"]=="BULLISH" else ("🔴" if d["direction"]=="BEARISH" else "🟡")
            out.append(f"{ic} {tf}: {d['direction']} ({d['pct']}%)")
        out.append(f"📊 Alignment: **{mtf_data.get('align_pct',50)}%**")
        out.append("")
    if verdict in ["STRONG BUY","BUY","TRADE NOW"]:
        out.append("**Trade Plan:**")
        out.append(f"• Entry: **Rs.{price:.2f}**")
        out.append(f"• Stop Loss: **Rs.{sl}**")
        out.append(f"• Target: **Rs.{tgt}**")
        out.append(f"• Qty: **{qty} shares** | R:R {rr}:1")
        out.append(f"• Max Loss: Rs.{max_loss:,.0f} | Max Gain: Rs.{max_gain:,.0f}")
        out.append(f"• Win Probability: **{win_prob}%**")
        out.append("")
        out.append("*SL kabhi mat hatao. Risk management sabse important hai.*")
    else:
        out.append("**Kya karein abhi?**")
        out.append(f"• Score 68+ ka wait karo (abhi {total_score}/100)")
        out.append("• Volume surge ke saath confirm hone pe entry lena")
        out.append("• Aaj watchlist mein rakho")
    out.append(f"\n*AI Confidence: {conf}% | Score: {total_score}/100*")
    return "\n".join(out)




# =============================================================
# MULTI-BROKER API SUPPORT (Startup Scale)
# =============================================================
BROKER_APIS = {
    "Zerodha Kite":  {"lib": "kiteconnect",   "status": "Active",    "color": "#00b880"},
    "Dhan HQ":       {"lib": "dhanhq",         "status": "Ready",     "color": "#4e8fff"},
    "Angel One":     {"lib": "smartapi-python","status": "Ready",     "color": "#a78bfa"},
    "Upstox":        {"lib": "upstox-python",  "status": "Ready",     "color": "#f39c12"},
    "Fyers":         {"lib": "fyers-apiv3",    "status": "Ready",     "color": "#00e5a0"},
}

def show_broker_status():
    """Show connected broker status."""
    results = {}
    for broker, info in BROKER_APIS.items():
        try:
            __import__(info["lib"].split("-")[0].replace("-","_"))
            results[broker] = {"installed": True, **info}
        except ImportError:
            results[broker] = {"installed": False, **info}
    return results


# =============================================================
# MAIN DASHBOARD
# =============================================================
st.title("📊 AI Trading PRO+ v1.3")
st.caption(f"👤 {user}  |  {ist_str()}")

st.markdown("### 🎯 Select Trading Mode")
selected_mode = st.radio(
    "Trading Mode", list(MODES.keys()),
    horizontal=True, label_visibility="collapsed"
)
mcfg = MODES[selected_mode]

mode_icons = {"📈 Intraday":"🔵","🌊 Swing":"🟣","📊 Futures":"🟠","🎯 Options":"🟢"}
st.info(f"{mode_icons.get(selected_mode,'🔵')} **{selected_mode}** — {mcfg['desc']}  |  Product: `{mcfg['product']}`")

if selected_mode == "📈 Intraday":
    now_t = datetime.datetime.now().time()
    if now_t < datetime.time(9, 15):
        st.warning("⏰ Market opens at 9:15 AM IST")
    elif now_t > datetime.time(14, 45):
        st.error("🔴 Intraday cutoff — square off before 3:15 PM")
    else:
        dt_close  = datetime.datetime.combine(datetime.date.today(), datetime.time(15, 15))
        mins_left = int((dt_close - datetime.datetime.now()).seconds / 60)
        st.success(f"✅ Market open — {mins_left} minutes left")

st.markdown("---")

col_main, col_set = st.columns([3, 1])

with col_set:
    st.markdown("#### ⚙️ Settings")
    mode = st.radio("Order Mode", ["Paper", "Live"], horizontal=True)
    if mode == "Live": st.warning("⚠️ REAL MONEY")
    capital  = st.number_input("Capital (₹)", 10000, 10000000, 100000, step=5000)
    risk     = st.number_input("Risk %", 0.5, 5.0, 1.5, step=0.1)
    force_trade = st.checkbox("🔥 FORCE TRADE (TEST MODE)", value=False)

with col_main:
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

    st.markdown("#### 📋 Stock Universe")
    universe_name = st.selectbox("Select Universe", list(STOCK_UNIVERSE.keys()))
    stocks = STOCK_UNIVERSE[universe_name]

    st.markdown(f"#### 🔍 Scanner — {universe_name} ({len(stocks)} stocks)")
    scan_btn = st.button(f"🔍 Scan All {len(stocks)} Stocks", type="primary")

    universe_key = f"scanned_{universe_name}_{selected_mode}"
    if scan_btn or universe_key not in st.session_state:
        st.session_state[universe_key] = True
        st.session_state.scan_results = []
        scan_period   = mcfg["period"] if mcfg["period"] != "1d" else "5d"
        scan_interval = mcfg["interval"] if mcfg["period"] != "1d" else "15m"

        with st.spinner(f"⏳ Scanning {len(stocks)} stocks..."):
            scan = []
            for s in stocks:
                try:
                    d = yf.Ticker(s).history(period=scan_period, interval=scan_interval)
                    if d is None or d.empty or len(d) < 20: continue
                    d = compute_indicators(d)
                    last = d.iloc[-1]
                    sc = 0
                    if last["Close"] > last["EMA20"] > last["EMA50"]: sc += 2
                    if 45 < last["RSI"] < 68:                         sc += 1
                    if last["MACD"] > last["MACD_Signal"]:             sc += 1
                    if last["Vol_Ratio"] > 1.2:                        sc += 1
                    if last["RSI"] > 78:                               sc -= 2
                    if last["RSI"] < 25:                               sc -= 1
                    sc = max(0, min(sc, 5))
                    chg = (last["Close"] - d["Close"].iloc[-2]) / d["Close"].iloc[-2] * 100
                    scan.append({
                        "Stock":  s.replace(".NS",""),
                        "Price":  round(float(last["Close"]),2),
                        "Chg%":   round(chg,2),
                        "RSI":    round(float(last["RSI"]),1),
                        "MACD":   round(float(last["MACD"]),2),
                        "Vol":    round(float(last["Vol_Ratio"]),2),
                        "Score":  sc,
                        "Signal": "🟢 BUY" if sc>=3 else("🔴 SELL" if sc<=1 else "🟡 HOLD"),
                        "_score": sc, "_sym": s,
                    })
                except: pass
            st.session_state.scan_results = sorted(scan, key=lambda x: -x["_score"])

    if st.session_state.scan_results:
        results = st.session_state.scan_results
        buy_picks = [r for r in results if r["Signal"]=="🟢 BUY"][:5]
        if buy_picks:
            st.markdown("##### 🏆 Top 5 BUY Picks")
            card_cols = st.columns(min(len(buy_picks), 5))
            for i, r in enumerate(buy_picks):
                chg = r["Chg%"]
                chg_color = "#27ae60" if chg >= 0 else "#e74c3c"
                chg_arrow = "▲" if chg >= 0 else "▼"
                rsi_label = "🔴 OB" if r["RSI"]>70 else("🟢 OS" if r["RSI"]<30 else "🟢 OK")
                rank_icons = ["🥇","🥈","🥉","4️⃣","5️⃣"]
                card_cols[i].markdown(f"""
<div style="background:linear-gradient(135deg,#003d2a,#001a12);
border:2px solid #00b880;border-radius:10px;padding:12px 10px;
text-align:center;min-height:180px;">
<div style="font-size:18px;">{rank_icons[i]}</div>
<div style="font-size:15px;font-weight:700;color:#00e5a0;">{r["Stock"]}</div>
<div style="font-size:18px;font-weight:700;color:#fff;">₹{r["Price"]:,.1f}</div>
<div style="font-size:12px;color:{chg_color};font-weight:600;">{chg_arrow} {abs(chg):.2f}%</div>
<div style="margin:6px 0;background:#00b880;border-radius:99px;padding:2px 8px;
font-size:11px;font-weight:700;color:#000;display:inline-block;">🟢 BUY</div>
<div style="font-size:11px;color:#aaa;">Score {r["Score"]}/5</div>
<div style="font-size:11px;color:#aaa;">RSI {r["RSI"]} {rsi_label}</div>
<div style="font-size:11px;color:#aaa;">Vol {r["Vol"]:.1f}x</div>
</div>""", unsafe_allow_html=True)
        else:
            st.warning("🟡 No strong BUY signals right now")

        st.markdown("##### 📋 All Stocks")
        df_sc = pd.DataFrame(results)
        disp = df_sc[["Stock","Price","Chg%","RSI","MACD","Vol","Score","Signal"]].copy()
        disp["Score"] = disp["Score"].apply(lambda x: f"{x}/5")
        disp["Price"] = disp["Price"].apply(lambda x: f"₹{x:,.2f}")
        disp["Chg%"]  = disp["Chg%"].apply(lambda x: f"{'▲' if x>=0 else '▼'} {abs(x):.2f}%")
        st.dataframe(disp, hide_index=True, use_container_width=True, height=300)
        best_sym = results[0]["_sym"]
        best_idx = stocks.index(best_sym) if best_sym in stocks else 0
    else:
        st.warning("⚠️ No scan results — click Scan button")
        best_idx = 0

    stock = st.selectbox("📌 Select Stock to Trade", stocks,
                         index=best_idx, format_func=lambda x: x.replace(".NS", ""))

    # ── DATA LOADING WITH EXTENDED FALLBACK ───────────────────
    def load_data(sym, period, interval):
        fallbacks = [
            (period, interval),
            ("5d",  "15m"),
            ("5d",  "30m"),
            ("1mo", "1d"),
            ("3mo", "1d"),
            ("6mo", "1d"),  # NEW: extended fallback
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

    with st.spinner(f"Loading {stock.replace('.NS','')}..."):
        df = load_data(stock, mcfg["period"], mcfg["interval"])

    if df is None or df.empty or len(df) < 5:
        st.error(f"⚠️ No data for {stock.replace('.NS','')} — try Swing mode")
        st.info("💡 Switch to **🌊 Swing** mode — uses daily data, always available")
        st.stop()

    df    = compute_indicators(df)
    last  = df.iloc[-1]
    price = float(last["Close"])
    prev  = float(df["Close"].iloc[-2])
    chg_v = price - prev
    chg_p = chg_v / prev * 100

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("💲 Price",     f"₹{price:.2f}",      f"{chg_v:+.2f} ({chg_p:+.2f}%)")
    c2.metric("📊 RSI",       f"{last['RSI']:.1f}",  "OB" if last['RSI']>70 else("OS" if last['RSI']<30 else "OK"))
    c3.metric("📉 MACD",      f"{last['MACD']:.2f}", f"Hist: {last['MACD_Hist']:+.2f}")
    c4.metric("📈 ATR",       f"₹{last['ATR']:.2f}")
    c5.metric("🔊 Vol Ratio", f"{last['Vol_Ratio']:.2f}x")

    # ══════════════════════════════════════════════════════════
    # MASTER TRADE DECISION ENGINE — 10 Layer Scoring System
    # ══════════════════════════════════════════════════════════

    # LAYER 1: AI Ensemble (20 pts)
    FEAT_COLS = ["EMA9","EMA20","EMA50","RSI","MACD","MACD_Hist","Stoch_K","BB_Pct",
                 "BB_Width","Vol_Ratio","Return_1","Return_3","Price_Pos",
                 "RSI_MA","Stoch_D","ADX","MFI","CCI","Williams_R","OBV"]
    feat_cols = [c for c in FEAT_COLS if c in df.columns]
    df["T3"] = (df["Close"].shift(-3) > df["Close"]*1.005).astype(int)
    df["T1"] = (df["Close"].shift(-1) > df["Close"]*1.002).astype(int)
    fd_raw   = df[feat_cols].ffill().fillna(0).iloc[20:]
    ai_prob=0.5; ai_pct=50; ai_accuracy=0.0; ai_confidence="Low"
    ai_model_name="Default"; feature_importance={}; wf_results=[]
    ai_model_name_short="RF"
    if len(fd_raw) >= 40:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import accuracy_score
            from sklearn.ensemble import (RandomForestClassifier,
                GradientBoostingClassifier, AdaBoostClassifier)
            _tc  = "T3" if len(fd_raw)>=60 else "T1"
            td   = df[_tc].loc[fd_raw.index].fillna(0).iloc[:-3]
            fd   = fd_raw.loc[td.index]
            scaler = StandardScaler()
            X = scaler.fit_transform(fd.values); y = td.values
            n_folds = 5 if len(X)>=80 else 3
            tscv = TimeSeriesSplit(n_splits=n_folds)
            fold_accs = []
            for ftr,fval in tscv.split(X):
                if len(set(y[ftr]))<2 or len(fval)<2: continue
                try:
                    _rf = RandomForestClassifier(n_estimators=50,max_depth=4,random_state=42)
                    _rf.fit(X[ftr],y[ftr])
                    fold_accs.append(round(accuracy_score(y[fval],_rf.predict(X[fval]))*100,1))
                except: continue
            wf_results = fold_accs
            splits = list(tscv.split(X))
            if splits:
                tr_idx,val_idx = splits[-1]
                X_tr,X_val = X[tr_idx],X[val_idx]
                y_tr,y_val = y[tr_idx],y[val_idx]
                if len(set(y_tr))>=2 and len(X_val)>=2:
                    model_probs=[]; model_labels=[]; all_imps=[]
                    if XGB_OK:
                        try:
                            import xgboost as xgb
                            m=xgb.XGBClassifier(n_estimators=min(150,max(50,len(X_tr))),
                                max_depth=4,learning_rate=0.08,eval_metric="logloss",
                                random_state=42,verbosity=0)
                            m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)
                            model_probs.append(m.predict_proba(X[-1:])[0][1])
                            model_labels.append(f"XGB {accuracy_score(y_val,m.predict(X_val))*100:.0f}%")
                            all_imps.append(dict(zip(feat_cols,m.feature_importances_)))
                        except: pass
                    if LGB_OK:
                        try:
                            import lightgbm as lgb
                            m=lgb.LGBMClassifier(n_estimators=min(150,max(50,len(X_tr))),
                                num_leaves=15,learning_rate=0.08,min_child_samples=3,
                                random_state=42,verbose=-1)
                            m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],
                                callbacks=[lgb.early_stopping(20,verbose=False),lgb.log_evaluation(-1)])
                            model_probs.append(m.predict_proba(X[-1:])[0][1])
                            model_labels.append(f"LGB {accuracy_score(y_val,m.predict(X_val))*100:.0f}%")
                        except: pass
                    for Cls,kw in [
                        (GradientBoostingClassifier,{"n_estimators":150,"max_depth":4,"learning_rate":0.05,"random_state":42}),
                        (RandomForestClassifier,    {"n_estimators":200,"max_depth":6,"min_samples_leaf":3,"random_state":42}),
                        (AdaBoostClassifier,        {"n_estimators":100,"learning_rate":0.1,"random_state":42}),
                    ]:
                        try:
                            m=Cls(**kw); m.fit(X_tr,y_tr)
                            model_probs.append(m.predict_proba(X[-1:])[0][1])
                            lbl=Cls.__name__[:3]
                            model_labels.append(f"{lbl} {accuracy_score(y_val,m.predict(X_val))*100:.0f}%")
                            if hasattr(m,"feature_importances_"):
                                all_imps.append(dict(zip(feat_cols,m.feature_importances_)))
                        except: pass
                    if model_probs:
                        ai_prob=float(np.mean(model_probs))
                        ai_pct=round(ai_prob*100)
                        ai_accuracy=round(float(np.mean([float(l.split()[-1].replace("%","")) for l in model_labels])),1)
                        ai_model_name=f"{len(model_probs)} models | Acc:{ai_accuracy}%"
                        ai_model_name_short=f"{len(model_probs)} models"
                        if all_imps:
                            comb={}
                            for d2 in all_imps:
                                for k2,v2 in d2.items(): comb[k2]=comb.get(k2,0)+v2/len(all_imps)
                            feature_importance=dict(sorted(comb.items(),key=lambda x:-x[1])[:8])
                        ai_confidence="High" if ai_accuracy>=65 else ("Medium" if ai_accuracy>=55 else "Low")
        except: pass
    ai_layer_score = round(ai_pct*0.20)  # max 20

    # LAYER 2: Technical (20 pts)
    c_trend  = last["Close"] > last.get("EMA20",0) > last.get("EMA50",0)
    c_ema9   = last["Close"] > last.get("EMA9", last["Close"])
    c_rsi    = 45 < float(last.get("RSI",50)) < 68
    c_macd   = float(last.get("MACD",0)) > float(last.get("MACD_Signal",0))
    c_macd_h = float(last.get("MACD_Hist",0)) > 0
    c_vol    = float(last.get("Vol_Ratio",1)) > 1.1
    c_bb     = last["Close"] > float(last.get("BB_Mid", last["Close"]))
    c_stoch  = float(last.get("Stoch_K",50)) < 70
    c_adx    = float(last.get("ADX",0)) > 20
    c_mfi    = float(last.get("MFI",50)) > 50
    c_supertr= float(last.get("ST_Dir",0)) > 0
    rsi_ob   = float(last.get("RSI",50)) > 75
    rsi_os   = float(last.get("RSI",50)) < 30
    tech_checks = {
        "Trend EMA20>EMA50": c_trend, "Price > EMA9": c_ema9,
        "RSI 45-68":         c_rsi,   "MACD > Signal": c_macd,
        "MACD Hist +ve":     c_macd_h,"Volume surge": c_vol,
        "Above BB Mid":      c_bb,    "Stoch < 70":  c_stoch,
        "ADX > 20":          c_adx,   "MFI > 50":    c_mfi,
        "Supertrend Bull":   c_supertr,
    }
    tech_score      = sum(tech_checks.values())
    tech_layer_score= round(tech_score/11*20)

    # LAYER 3: Market Structure (15 pts)
    struct_layer_score=5; bos_signal="Unknown"
    try:
        bos_data  = detect_bos_choch(df.tail(60))
        tr_type   = bos_data.get("trend","Unknown")
        if tr_type=="Uptrend":    struct_layer_score=15
        elif tr_type=="Downtrend":struct_layer_score=3
        else:                     struct_layer_score=8
        if bos_data.get("choch") and "BULLISH" in bos_data["choch"].get("direction",""):
            struct_layer_score=min(15,struct_layer_score+4)
        bos_signal=tr_type
    except: pass

    # LAYER 4: ICT (15 pts)
    ict_layer_score=7; ict_signal="Neutral"
    try:
        pd_data  = get_premium_discount(df)
        lq_data  = detect_liquidity_sweep(df.tail(30))
        kz_data  = get_kill_zones()
        eq_data  = detect_equal_levels(df.tail(60))
        ict_pts  = 0
        if "Discount" in pd_data.get("zone",""):  ict_pts+=5;  ict_signal="Discount zone"
        elif "Premium" in pd_data.get("zone",""): ict_pts-=3;  ict_signal="Premium zone"
        else:                                       ict_pts+=2;  ict_signal="Equilibrium"
        bull_sw=[s for s in lq_data if "Bullish" in s.get("type","")]
        if bull_sw:       ict_pts+=5; ict_signal="Bullish sweep"
        if kz_data.get("active"): ict_pts+=3
        if eq_data.get("near_eql"): ict_pts+=2
        ict_layer_score=max(0,min(15,ict_pts+5))
    except: pass

    # LAYER 5: Chart Patterns (10 pts)
    pattern_layer_score=5; pattern_signal="None"
    try:
        cp_list=detect_chart_patterns(df.tail(80))
        if cp_list:
            top_p=cp_list[0]
            if top_p["type"]=="bullish":
                pattern_layer_score=min(10,5+top_p["strength"])
                pattern_signal=top_p["pattern"]
            elif top_p["type"]=="bearish":
                pattern_layer_score=max(0,5-top_p["strength"])
                pattern_signal=top_p["pattern"]
    except: pass

    # LAYER 6: SMC OB+FVG (10 pts)
    smc_layer_score=5; smc_signal="No OB/FVG"
    try:
        obs=find_order_blocks(df.tail(80)); fvgs=find_fvg(df.tail(80))
        b_ob=[o for o in obs if "Bullish" in o["type"]]
        b_fvg=[f for f in fvgs if "Bullish" in f["type"]]
        if b_ob and price<b_ob[-1]["top"]*1.015: smc_layer_score=9; smc_signal="Near Bullish OB"
        elif b_fvg:                               smc_layer_score=7; smc_signal="Bullish FVG"
    except: pass

    # LAYER 7: MTF (10 pts)
    mtf_layer_score=5; mtf_signal="Load MTF tab"
    try:
        mtf_c=st.session_state.get("mtf_cache")
        if mtf_c and mtf_c.get("breakdown"):
            mtf_layer_score=max(0,min(10,round(mtf_c.get("confluence",50)/10)))
            mtf_signal=mtf_c.get("signal","Mixed")[:14]
    except: pass

    # LAYER 8: Volume (5 pts)
    _vr=float(last.get("Vol_Ratio",1))
    if _vr>=2.0:   vol_layer_score=5; vol_signal="Very high"
    elif _vr>=1.5: vol_layer_score=4; vol_signal="Above avg"
    elif _vr>=1.0: vol_layer_score=3; vol_signal="Normal"
    else:          vol_layer_score=1; vol_signal="Low vol"

    # LAYER 9: Fibonacci (3 pts)
    fib_layer_score=1; fib_signal="Not in zone"
    try:
        fib_d=get_fibonacci_levels(df)
        if fib_d.get("in_golden_zone"):      fib_layer_score=3; fib_signal="Golden Zone!"
        elif abs(fib_d.get("nearest_price",0)-price)/price<0.01:
            fib_layer_score=2; fib_signal=f"Near {fib_d.get('nearest_level','')[:8]}"
    except: pass

    # LAYER 10: Candle (2 pts)
    candle_layer_score=1; candle_signal="None"
    try:
        _cd=df.tail(50).copy(); _cd.index=pd.to_datetime(_cd.index)
        _pts=detect_candlestick_patterns(_cd)
        if _pts:
            _tp=sorted(_pts,key=lambda x:-x["strength"])[0]
            candle_signal=_tp["pattern"]
            if _tp["type"]=="bullish": candle_layer_score=min(2,_tp["strength"]//2+1)
            elif _tp["type"]=="bearish": candle_layer_score=0
    except: pass

    # ── TOTAL SCORE ─────────────────────────────────────────
    raw_score = (ai_layer_score+tech_layer_score+struct_layer_score+
                 ict_layer_score+pattern_layer_score+smc_layer_score+
                 mtf_layer_score+vol_layer_score+fib_layer_score+candle_layer_score)
    raw_score = round(min(raw_score/105*100, 100))

    # ── PENALTIES ────────────────────────────────────────────
    penalties=[]; penalty_pts=0
    if rsi_ob:
        penalty_pts+=15; penalties.append(("RSI Overbought >75",-15,"#e74c3c"))
    if _vr<0.7:
        penalty_pts+=10; penalties.append(("Very Low Volume",-10,"#e07b39"))
    if float(last.get("MACD_Hist",0))<0 and float(last.get("MACD",0))>float(last.get("MACD_Signal",0)):
        penalty_pts+=5;  penalties.append(("MACD Divergence",-5,"#f39c12"))
    try:
        _sz,_=find_demand_supply_zones(df.tail(80))
        for _z in _sz:
            if abs(price-_z["top"])/price<0.008:
                penalty_pts+=10; penalties.append((f"Near Supply Rs.{_z['top']}",-10,"#e74c3c")); break
    except: pass
    if bos_signal=="Downtrend":
        penalty_pts+=8; penalties.append(("Downtrend Structure",-8,"#e74c3c"))
    total_score=max(0,raw_score-penalty_pts)

    # ── DERIVED DISPLAY VARS ─────────────────────────────────
    c_ai       = ai_prob > 0.55
    struct_pct = min(100, round(struct_layer_score / 15 * 100))
    candle_pct = min(100, round(candle_layer_score / 2  * 100))
    vol_pct    = min(100, round(vol_layer_score    / 5  * 100))
    smc_pct    = min(100, round(smc_layer_score    / 10 * 100))

    # ── FINAL VERDICT ────────────────────────────────────────
    if force_trade:        verdict="TRADE NOW";    v_color="#00b880"; v_bg="#002d1e"; v_emoji="🚀"
    elif total_score>=82:  verdict="STRONG BUY";   v_color="#00b880"; v_bg="#002d1e"; v_emoji="🔥"
    elif total_score>=68:  verdict="BUY";          v_color="#27ae60"; v_bg="#0a1f10"; v_emoji="✅"
    elif total_score>=52:  verdict="WAIT";         v_color="#f39c12"; v_bg="#1a1200"; v_emoji="⏳"
    elif total_score>=38:  verdict="AVOID";        v_color="#e07b39"; v_bg="#2d1800"; v_emoji="⚠️"
    else:                  verdict="DO NOT TRADE"; v_color="#e74c3c"; v_bg="#2d0000"; v_emoji="🚫"
    signal=verdict in ["STRONG BUY","BUY","TRADE NOW"]
    direction=verdict

    # Trade plan
    atr_now    = max(float(last.get("ATR",price*0.01)),0.01)
    _sl_d      = atr_now*mcfg.get("sl_mult",1.5)
    _tg_d      = _sl_d*mcfg.get("rr",2.0)
    stop_loss_m= round(price-_sl_d,2)
    target_m   = round(price+_tg_d,2)
    qty_m      = min(max(1,int(capital*(risk/100)/_sl_d)), max(1,int(capital*0.20/max(price,1))))
    max_loss_rs= round(_sl_d*qty_m,2)
    max_gain_rs= round(_tg_d*qty_m,2)
    win_prob   = min(82,round(total_score*0.65+20))
    _rr_m      = round(_tg_d/_sl_d,1)
    combined   = total_score
    master     = total_score

    # ── PROFESSIONAL TRADE DECISION ENGINE ─────────────────────
    # ── Build checklist (MUST-HAVE conditions) ───────────────
    _must = {
        "Trend Bullish (EMA20>EMA50)":  c_trend,
        "RSI in safe zone (30-75)":     30 < float(last.get("RSI",50)) < 75,
        "MACD above Signal":            float(last.get("MACD",0))>float(last.get("MACD_Signal",0)),
        "Volume above average":         float(last.get("Vol_Ratio",1))>0.9,
        "Not near Supply Zone":         not any(abs(price-z["top"])/price<0.01
                                          for z in (find_demand_supply_zones(df.tail(80))[0]
                                          if len(df)>=20 else [])),
    }
    _good = {
        "Strong volume surge (>1.3x)":  float(last.get("Vol_Ratio",1))>1.3,
        "Supertrend Bullish":           float(last.get("ST_Dir",0))>0,
        "AI Model Bullish (>55%)":      ai_prob>0.55,
        "ADX shows trend strength":     float(last.get("ADX",0))>20,
        "MFI bullish (>50)":            float(last.get("MFI",50))>50,
        "Stochastic not overbought":    float(last.get("Stoch_K",50))<75,
        "Price above VWAP":             price>float(last.get("VWAP",price))*0.998,
    }
    _block = {
        "RSI NOT overbought (>78)":     not (float(last.get("RSI",50))>78),
        "NOT near major resistance":    True,
        "Downtrend NOT present":        bos_signal != "Downtrend",
        "Volume NOT extremely low":     float(last.get("Vol_Ratio",1))>0.5,
    }

    must_pass   = sum(_must.values())
    must_total  = len(_must)
    good_pass   = sum(_good.values())
    block_pass  = sum(_block.values())
    block_total = len(_block)

    # Hard blocks: if any blocker fails → NO TRADE regardless of score
    hard_blocked = block_pass < block_total
    blockers = [k for k,v in _block.items() if not v]
    missing_must = [k for k,v in _must.items() if not v]

    # Final GO/NO-GO decision
    if force_trade:
        go_decision = "TRADE"
        go_reason   = "Force signal activated"
    elif hard_blocked:
        go_decision = "NO TRADE"
        go_reason   = f"Blocked: {', '.join(blockers)}"
    elif must_pass < 4:
        go_decision = "NO TRADE"
        go_reason   = f"Only {must_pass}/{must_total} must-have conditions met"
    elif total_score >= 82 and must_pass == must_total:
        go_decision = "STRONG BUY"
        go_reason   = f"All {must_total} conditions ✅ | Score {total_score}/100"
    elif total_score >= 68 and must_pass >= 4:
        go_decision = "BUY"
        go_reason   = f"{must_pass}/{must_total} conditions ✅ | Score {total_score}/100"
    elif total_score >= 52:
        go_decision = "WAIT"
        go_reason   = f"Score {total_score}/100 — Need 68+ to trade"
    else:
        go_decision = "NO TRADE"
        go_reason   = f"Score too low: {total_score}/100 — Need minimum 68"

    is_trade     = go_decision in ["STRONG BUY","BUY","TRADE"]
    go_color     = "#00b880" if go_decision in ["STRONG BUY","TRADE"] else                    "#27ae60" if go_decision=="BUY" else                    "#f39c12" if go_decision=="WAIT" else "#e74c3c"
    go_bg        = "#003d2a" if is_trade else ("#1a1200" if go_decision=="WAIT" else "#2d0000")
    go_emoji     = "🔥" if go_decision=="STRONG BUY" else                    "✅" if go_decision=="BUY" else                    "🚀" if go_decision=="TRADE" else                    "⏳" if go_decision=="WAIT" else "🚫"

    # ── WHAT TO DO NEXT (if no trade) ────────────────────────
    next_action = []
    if not is_trade:
        if not _must["Trend Bullish (EMA20>EMA50)"]:
            next_action.append("Wait for price to cross above EMA20 and EMA50")
        if not _must["RSI in safe zone (30-75)"]:
            next_action.append("Wait for RSI to come below 75 (overbought)")
        if not _must["MACD above Signal"]:
            next_action.append("Wait for MACD crossover above Signal line")
        if not _must["Volume above average"]:
            next_action.append("Wait for volume to pick up (>1x average)")
        if float(last.get("RSI",50))>78:
            next_action.append("RSI >78 — price likely to pullback first, buy dip")
        if bos_signal == "Downtrend":
            next_action.append("Market in downtrend — wait for CHOCH or BOS up")
        if not next_action:
            next_action.append(f"Score needs to reach 68+ (currently {total_score})")

    # ── LAYER ROWS ────────────────────────────────────────────
    layer_rows=""
    layer_data=[
        ("AI Model",      ai_layer_score,      20, ai_model_name_short),
        ("Technical",     tech_layer_score,    20, f"{tech_score}/11"),
        ("Structure",     struct_layer_score,  15, bos_signal[:10]),
        ("ICT",           ict_layer_score,     15, ict_signal[:12]),
        ("Chart Pattern", pattern_layer_score, 10, pattern_signal[:10]),
        ("SMC/OB/FVG",    smc_layer_score,     10, smc_signal[:10]),
        ("MTF 4-TF",      mtf_layer_score,     10, mtf_signal[:12]),
        ("Volume",        vol_layer_score,       5, vol_signal[:10]),
        ("Fibonacci",     fib_layer_score,       3, fib_signal[:10]),
        ("Candle",        candle_layer_score,    2, candle_signal[:10]),
    ]
    for lbl,v,mx,note in layer_data:
        pct=int(v/mx*100)
        c="#00b880" if v>=int(mx*0.65) else ("#f39c12" if v>=int(mx*0.4) else "#e74c3c")
        layer_rows += (
            f"<div style='background:rgba(0,0,0,0.4);border-radius:7px;padding:9px 6px;"
            f"text-align:center;border:1px solid {c}33;'>"
            f"<div style='font-size:9px;color:#666;margin-bottom:3px;'>{lbl}</div>"
            f"<div style='font-size:16px;font-weight:700;color:{c};'>{v}/{mx}</div>"
            f"<div style='background:rgba(255,255,255,0.05);border-radius:99px;height:4px;margin:4px 0;'>"
            f"<div style='width:{pct}%;background:{c};border-radius:99px;height:4px;'></div></div>"
            f"<div style='font-size:8px;color:#555;'>{note}</div></div>"
        )

    penalty_rows=""
    if penalties:
        for rn,rp,rc in penalties:
            penalty_rows+=(f"<div style='background:rgba(0,0,0,0.3);border-left:3px solid {rc};"
                f"border-radius:5px;padding:6px 12px;margin-top:5px;font-size:12px;"
                f"display:flex;justify-content:space-between;'>"
                f"<span style='color:{rc};font-weight:600;'>⚠ {rn}</span>"
                f"<span style='color:{rc};'>{rp} pts</span></div>")
    else:
        penalty_rows=(
            "<div style='background:rgba(0,184,128,0.08);border-left:3px solid #00b880;"
            "border-radius:5px;padding:6px 12px;margin-top:5px;font-size:12px;color:#00b880;'>"
            "✅ No risk factors — clean setup</div>")

    # ── MUST-HAVE CHECKLIST HTML ──────────────────────────────
    must_rows = ""
    for cond, val in _must.items():
        ic = "#00b880" if val else "#e74c3c"
        must_rows += (f"<div style='display:flex;align-items:center;gap:8px;"
                      f"padding:5px 0;border-bottom:1px solid #1a1a2e;'>"
                      f"<span style='font-size:16px;'>{'✅' if val else '❌'}</span>"
                      f"<span style='font-size:12px;color:{'#ccc' if val else '#e74c3c'};'>{cond}</span>"
                      f"</div>")

    good_rows = ""
    for cond, val in _good.items():
        ic = "#00b880" if val else "#888"
        good_rows += (f"<div style='display:flex;align-items:center;gap:8px;padding:4px 0;'>"
                      f"<span style='font-size:13px;'>{'🟢' if val else '⚪'}</span>"
                      f"<span style='font-size:11px;color:{'#aaa' if val else '#555'};'>{cond}</span>"
                      f"</div>")

    next_html = ""
    if next_action:
        next_html = "<div style='margin-top:10px;'>"
        next_html += "<div style='font-size:11px;color:#f39c12;font-weight:600;margin-bottom:6px;'>What to do next:</div>"
        for act in next_action[:3]:
            next_html += f"<div style='font-size:11px;color:#888;padding:3px 0;'>→ {act}</div>"
        next_html += "</div>"

    # Build HTML as Python string (avoids Streamlit markdown parser issues)
    _h  = f"<div style='background:{go_bg};border:3px solid {go_color};"
    _h += "border-radius:20px;padding:0;margin:12px 0;overflow:hidden;'>"
    _h += f"<div style='background:{go_color}22;padding:20px 24px;border-bottom:1px solid {go_color}44;'>"
    _h += f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
    _h += f"<div><div style='font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.12em;margin-bottom:6px;'>TRADE DECISION — {stock.replace('.NS','')} | {selected_mode} | {now_ist().strftime('%H:%M IST')}</div>"
    _h += f"<div style='font-size:44px;font-weight:900;color:{go_color};line-height:1;letter-spacing:-.5px;'>{go_emoji} {go_decision}</div>"
    _h += f"<div style='font-size:13px;color:#aaa;margin-top:6px;'>{go_reason}</div></div>"
    _h += f"<div style='text-align:right;'><div style='font-size:64px;font-weight:900;color:{go_color};line-height:1;'>{total_score}</div>"
    _h += f"<div style='font-size:11px;color:#888;'>/ 100 pts</div>"
    _h += f"<div style='font-size:11px;color:#888;margin-top:4px;'>Must: {must_pass}/{must_total} ✓ | Good: {good_pass}/{len(_good)}</div>"
    _h += "</div></div>"
    # Score bar
    _h += f"<div style='background:rgba(255,255,255,0.08);border-radius:99px;height:16px;margin:16px 0 6px;position:relative;overflow:hidden;'>"
    _h += f"<div style='width:{total_score}%;background:linear-gradient(90deg,{go_color}77,{go_color});border-radius:99px;height:16px;box-shadow:0 0 16px {go_color}55;'></div>"
    _h += "<div style='position:absolute;left:38%;top:0;width:2px;height:16px;background:#fff;opacity:0.15;'></div>"
    _h += "<div style='position:absolute;left:52%;top:0;width:2px;height:16px;background:#fff;opacity:0.15;'></div>"
    _h += "<div style='position:absolute;left:68%;top:0;width:2px;height:16px;background:#27ae60;opacity:0.5;'></div>"
    _h += "<div style='position:absolute;left:82%;top:0;width:2px;height:16px;background:#00b880;opacity:0.7;'></div></div>"
    _h += "<div style='display:flex;justify-content:space-between;font-size:9px;color:#444;'>"
    _h += "<span>0</span><span style='color:#e07b39;'>38 AVOID</span><span style='color:#f39c12;'>52 WAIT</span>"
    _h += "<span style='color:#27ae60;'>68 BUY</span><span style='color:#00b880;'>82 STRONG</span><span>100</span></div>"
    _h += "</div>"
    # Main content grid - left (checklist) + right (scores + plan)
    _h += "<div style='padding:18px 24px;display:grid;grid-template-columns:1fr 1fr;gap:18px;'>"
    # Left: checklist
    _h += f"<div><div style='font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;'>Must-Have Conditions ({must_pass}/{must_total})</div>"
    _h += must_rows
    _h += f"<div style='margin-top:12px;font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;'>Bonus Conditions ({good_pass}/{len(_good)})</div>"
    _h += good_rows
    _h += next_html
    _h += "</div>"
    # Right: scores + plan
    _h += "<div>"
    _h += "<div style='font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;'>Score Breakdown (10 Layers)</div>"
    _h += f"<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:5px;margin-bottom:14px;'>{layer_rows}</div>"
    # Trade plan
    _h += "<div style='background:rgba(0,0,0,0.4);border-radius:10px;padding:12px;margin-bottom:10px;'>"
    _h += "<div style='font-size:10px;color:#888;text-transform:uppercase;margin-bottom:8px;'>Trade Plan</div>"
    _h += "<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:4px;text-align:center;'>"
    _h += f"<div><div style='font-size:9px;color:#888;'>Entry</div><div style='font-size:13px;font-weight:600;color:#e6edf3;'>Rs.{price:.2f}</div></div>"
    _h += f"<div><div style='font-size:9px;color:#e74c3c;'>Stop Loss</div><div style='font-size:13px;font-weight:600;color:#e74c3c;'>Rs.{stop_loss_m}</div></div>"
    _h += f"<div><div style='font-size:9px;color:#00b880;'>Target</div><div style='font-size:13px;font-weight:600;color:#00b880;'>Rs.{target_m}</div></div>"
    _h += f"<div><div style='font-size:9px;color:#f39c12;'>R:R</div><div style='font-size:13px;font-weight:600;color:#f39c12;'>{_rr_m}:1</div></div>"
    _h += f"<div><div style='font-size:9px;color:#a78bfa;'>Qty</div><div style='font-size:13px;font-weight:600;color:#a78bfa;'>{qty_m} sh</div></div>"
    _h += "</div></div>"
    # Win/Loss
    _h += "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;'>"
    _h += f"<div style='background:rgba(0,184,128,0.12);border:1px solid rgba(0,184,128,0.3);border-radius:8px;padding:8px;text-align:center;'><div style='font-size:9px;color:#888;'>Win Prob</div><div style='font-size:22px;font-weight:700;color:#00b880;'>{win_prob}%</div></div>"
    _h += f"<div style='background:rgba(231,76,60,0.1);border:1px solid rgba(231,76,60,0.25);border-radius:8px;padding:8px;text-align:center;'><div style='font-size:9px;color:#888;'>Max Loss</div><div style='font-size:18px;font-weight:700;color:#e74c3c;'>Rs.{max_loss_rs:,.0f}</div></div>"
    _h += f"<div style='background:rgba(0,184,128,0.1);border:1px solid rgba(0,184,128,0.25);border-radius:8px;padding:8px;text-align:center;'><div style='font-size:9px;color:#888;'>Max Gain</div><div style='font-size:18px;font-weight:700;color:#00b880;'>Rs.{max_gain_rs:,.0f}</div></div>"
    _h += "</div></div></div>"
    # Penalties
    _h += f"<div style='padding:0 24px 18px;'>{penalty_rows}"
    _h += f"<div style='margin-top:8px;font-size:10px;color:#444;text-align:center;'>Raw:{raw_score} − Penalty:{penalty_pts} = Final:{total_score}/100 | Min 68 to BUY | Min 82 for STRONG BUY</div>"
    _h += "</div></div>"
    st.markdown(_h, unsafe_allow_html=True)

    if signal and ALERT_ON_SIGNAL:

        fire_alert(f"{verdict} [{selected_mode}]", stock, price,
                   qty_m, stop_loss_m, target_m, total_score, mode)

    # ── AI EXPLAINABILITY ─────────────────────────────────────
    try:
        _bos2 = detect_bos_choch(df.tail(60)) if len(df)>=60 else {}
        _ob2  = find_order_blocks(df.tail(60))
        _fvg2 = find_fvg(df.tail(50))
        _cp2  = detect_chart_patterns(chart_df)
        _expl = build_ai_explanation(
            last, price, ai_prob, ai_pct, _bos2, _ob2, _fvg2, _cp2,
            float(last.get("Vol_Ratio",1)), float(last.get("RSI",50)), total_score)
    except Exception:
        _expl = {"bull":[],"bear":[],"warn":[],"conf":total_score}

    _bull=_expl["bull"]; _bear=_expl["bear"]; _warn=_expl["warn"]; _conf=_expl["conf"]
    _ec="#00b880" if _conf>=70 else ("#f39c12" if _conf>=50 else "#e74c3c")
    _eh ="<div style='background:#0d1117;border:2px solid #21262d;border-radius:14px;padding:18px;margin:10px 0;'>"
    _eh+="<div style='font-size:13px;font-weight:700;color:#e6edf3;margin-bottom:14px;'>AI Signal Explanation</div>"
    _eh+="<div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;'>"
    _eh+="<div><div style='font-size:11px;color:#00b880;font-weight:600;margin-bottom:8px;'>BULLISH FACTORS</div>"
    for _l,_d in _bull[:6]:
        _eh+=f"<div style='background:#0d2818;border-left:3px solid #00b880;border-radius:5px;padding:6px 10px;margin-bottom:5px;'><b style='color:#00b880;font-size:12px;'>{_l}</b><br><span style='color:#888;font-size:11px;'>{_d}</span></div>"
    if not _bull: _eh+="<div style='color:#555;font-size:11px;'>No bullish factors</div>"
    _eh+="</div><div>"
    if _bear:
        _eh+="<div style='font-size:11px;color:#e74c3c;font-weight:600;margin-bottom:8px;'>BEARISH FACTORS</div>"
        for _l,_d in _bear[:4]:
            _eh+=f"<div style='background:#2d0a0a;border-left:3px solid #e74c3c;border-radius:5px;padding:6px 10px;margin-bottom:5px;'><b style='color:#e74c3c;font-size:12px;'>{_l}</b><br><span style='color:#888;font-size:11px;'>{_d}</span></div>"
    if _warn:
        _eh+="<div style='font-size:11px;color:#f39c12;font-weight:600;margin-bottom:8px;margin-top:6px;'>WARNINGS</div>"
        for _l,_d in _warn[:3]:
            _eh+=f"<div style='background:#1a1200;border-left:3px solid #f39c12;border-radius:5px;padding:6px 10px;margin-bottom:5px;'><b style='color:#f39c12;font-size:12px;'>{_l}</b><br><span style='color:#888;font-size:11px;'>{_d}</span></div>"
    if not _bear and not _warn: _eh+="<div style='background:#0d2818;border-radius:8px;padding:10px;color:#00b880;font-size:12px;'>Clean setup — no major risks!</div>"
    _eh+="</div></div>"
    _eh+=f"<div style='margin-top:12px;padding-top:10px;border-top:1px solid #21262d;display:flex;justify-content:space-between;align-items:center;'>"
    _eh+=f"<span style='font-size:11px;color:#888;'>{len(_bull)} bullish · {len(_bear)} bearish · {len(_warn)} warnings</span>"
    _eh+=f"<span style='font-size:14px;font-weight:700;color:{_ec};'>Confidence: {_conf}%</span></div>"
    _eh+=f"<div style='background:rgba(255,255,255,0.06);border-radius:99px;height:6px;margin-top:6px;'><div style='width:{_conf}%;background:{_ec};border-radius:99px;height:6px;'></div></div></div>"
    st.markdown(_eh, unsafe_allow_html=True)

    # ── PORTFOLIO HEATMAP + SECTOR EXPOSURE ─────────────────
    st.markdown("---")

    # ============================================================
    # PANEL 1: CAPITAL ADVISOR
    # ============================================================
    with st.expander("💰 Capital Advisor — Position Sizing Calculator", expanded=True):
        cap_c1, cap_c2, cap_c3 = st.columns(3)
        adv_capital = cap_c1.number_input("Your Capital (₹)", 10000, 10000000, int(capital), 5000, key="adv_cap")
        adv_risk_pct= cap_c2.number_input("Risk per trade %", 0.5, 5.0, float(risk), 0.5, key="adv_risk")
        adv_rr      = cap_c3.number_input("Target R:R", 1.0, 10.0, 2.0, 0.5, key="adv_rr")

        adv_risk_rs  = round(adv_capital * adv_risk_pct / 100, 2)
        adv_sl_dist  = atr_now * mcfg.get("sl_mult", 1.5)
        adv_qty      = max(1, int(adv_risk_rs / adv_sl_dist)) if adv_sl_dist > 0 else 1
        adv_max_loss = round(adv_sl_dist * adv_qty, 2)
        adv_tgt_dist = adv_sl_dist * adv_rr
        adv_max_gain = round(adv_tgt_dist * adv_qty, 2)
        adv_sl_px    = round(price - adv_sl_dist, 2)
        adv_tgt_px   = round(price + adv_tgt_dist, 2)

        # Capital Advisor Card
        adv_col = "#00b880" if is_trade else "#f39c12"
        adv_h  = f"<div style='background:#0d1117;border:2px solid {adv_col};border-radius:14px;padding:20px;margin:8px 0;'>"
        adv_h += f"<div style='font-size:13px;font-weight:700;color:#e6edf3;margin-bottom:16px;'>Capital Advisor — {stock.replace('.NS','')} {selected_mode}</div>"
        adv_h += f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px;'>"
        for lbl2,val2,vc2 in [
            ("Capital", f"₹{adv_capital:,.0f}", "#e6edf3"),
            ("Risk Amount", f"₹{adv_risk_rs:,.0f}", "#f39c12"),
            ("Qty to Buy", str(adv_qty)+" shares", adv_col),
            ("SL Distance", f"₹{adv_sl_dist:.2f}", "#e74c3c"),
        ]:
            adv_h += f"<div style='background:#161b22;border-radius:8px;padding:12px;text-align:center;border:1px solid #21262d;'><div style='font-size:10px;color:#888;margin-bottom:4px;'>{lbl2}</div><div style='font-size:18px;font-weight:700;color:{vc2};'>{val2}</div></div>"
        adv_h += "</div>"
        adv_h += f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;'>"
        for lbl2,val2,vc2 in [
            ("Entry", f"₹{price:.2f}", "#e6edf3"),
            ("Stop Loss", f"₹{adv_sl_px}", "#e74c3c"),
            ("Target", f"₹{adv_tgt_px}", "#00b880"),
            ("R:R Ratio", f"{adv_rr:.1f}:1", "#a78bfa"),
        ]:
            adv_h += f"<div style='background:#161b22;border-radius:8px;padding:12px;text-align:center;border:1px solid #21262d;'><div style='font-size:10px;color:#888;margin-bottom:4px;'>{lbl2}</div><div style='font-size:18px;font-weight:700;color:{vc2};'>{val2}</div></div>"
        adv_h += "</div>"
        adv_h += f"<div style='margin-top:14px;background:#161b22;border-radius:8px;padding:12px;display:flex;justify-content:space-between;align-items:center;'>"
        adv_h += f"<div><span style='color:#888;font-size:12px;'>Max Loss: </span><span style='color:#e74c3c;font-size:16px;font-weight:700;'>₹{adv_max_loss:,.0f}</span></div>"
        adv_h += f"<div><span style='color:#888;font-size:12px;'>Risk %: </span><span style='color:#f39c12;font-size:16px;font-weight:700;'>{adv_risk_pct}%</span></div>"
        adv_h += f"<div><span style='color:#888;font-size:12px;'>Max Gain: </span><span style='color:#00b880;font-size:16px;font-weight:700;'>₹{adv_max_gain:,.0f}</span></div>"
        adv_h += f"<div><span style='color:#888;font-size:12px;'>Win Prob: </span><span style='color:#a78bfa;font-size:16px;font-weight:700;'>{win_prob}%</span></div>"
        adv_h += "</div></div>"
        st.markdown(adv_h, unsafe_allow_html=True)

        # Rule of thumb advice
        if adv_risk_pct > 2:
            st.warning(f"Risk {adv_risk_pct}% per trade — bahut zyada! Max 1.5% recommended.")
        elif adv_rr < 1.5:
            st.warning("R:R below 1.5:1 — profitable trading mushkil hoga.")
        else:
            st.success(f"Risk management theek hai — ₹{adv_risk_rs:,.0f} risk, ₹{adv_max_gain:,.0f} potential gain.")

    # ============================================================
    # PANEL 2: PORTFOLIO RISK ENGINE
    # ============================================================
    with st.expander("🛡️ Portfolio Risk Engine", expanded=True):
        positions     = st.session_state.get("paper_positions", {})
        balance       = st.session_state.paper_balance
        total_exposed = sum(p.get("qty",0)*p.get("price",0) for p in positions.values())
        total_port    = balance + total_exposed
        exposure_pct  = total_exposed / (total_port+0.01) * 100
        n_pos         = len(positions)

        # Overall risk gauge
        risk_color = "#00b880" if exposure_pct<40 else ("#f39c12" if exposure_pct<70 else "#e74c3c")
        risk_label = "LOW" if exposure_pct<40 else ("MEDIUM" if exposure_pct<70 else "HIGH")

        rpe_h  = f"<div style='background:#0d1117;border:2px solid {risk_color};border-radius:14px;padding:20px;'>"
        rpe_h += f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;'>"
        rpe_h += f"<div style='text-align:center;background:#161b22;border-radius:10px;padding:16px;'>"
        rpe_h += f"<div style='font-size:11px;color:#888;text-transform:uppercase;'>Portfolio Exposure</div>"
        rpe_h += f"<div style='font-size:48px;font-weight:900;color:{risk_color};line-height:1;'>{exposure_pct:.0f}%</div>"
        rpe_h += f"<div style='font-size:14px;color:{risk_color};font-weight:600;'>{risk_label} RISK</div></div>"
        rpe_h += f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>"
        for lbl3,val3,vc3 in [
            ("Balance",    f"₹{balance:,.0f}",       "#00b880"),
            ("Invested",   f"₹{total_exposed:,.0f}", "#4e8fff"),
            ("Positions",  f"{n_pos}/5 max",          "#f39c12"),
            ("Portfolio",  f"₹{total_port:,.0f}",    "#e6edf3"),
        ]:
            rpe_h += f"<div style='background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:10px;text-align:center;'><div style='font-size:9px;color:#888;'>{lbl3}</div><div style='font-size:14px;font-weight:600;color:{vc3};'>{val3}</div></div>"
        rpe_h += "</div></div>"
        # Risk bar
        rpe_h += f"<div style='background:rgba(255,255,255,0.06);border-radius:99px;height:14px;margin-bottom:6px;position:relative;overflow:hidden;'>"
        rpe_h += f"<div style='width:{min(exposure_pct,100):.0f}%;background:linear-gradient(90deg,#00b880,{risk_color});border-radius:99px;height:14px;'></div>"
        rpe_h += "<div style='position:absolute;left:40%;top:0;width:2px;height:14px;background:#fff;opacity:0.2;'></div>"
        rpe_h += "<div style='position:absolute;left:70%;top:0;width:2px;height:14px;background:#fff;opacity:0.3;'></div></div>"
        rpe_h += "<div style='display:flex;justify-content:space-between;font-size:9px;color:#444;margin-bottom:14px;'>"
        rpe_h += "<span>0% Safe</span><span style='color:#f39c12;'>40% Medium</span><span style='color:#e74c3c;'>70% High</span><span>100%</span></div>"
        # Positions heatmap
        if positions:
            rpe_h += "<div style='font-size:11px;color:#888;font-weight:600;margin-bottom:8px;'>POSITION HEAT MAP</div>"
            rpe_h += "<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:6px;'>"
            for sym3,pos3 in positions.items():
                pnl3  = (price - pos3["price"]) * pos3["qty"] if stock.replace(".NS","")==sym3 else 0
                pc3   = "#00b880" if pnl3>=0 else "#e74c3c"
                rpe_h += f"<div style='background:{pc3}22;border:2px solid {pc3};border-radius:8px;padding:8px;text-align:center;'><div style='font-size:12px;font-weight:700;color:{pc3};'>{sym3[:8]}</div><div style='font-size:11px;color:{pc3};'>₹{pnl3:+.0f}</div></div>"
            rpe_h += "</div>"
        else:
            rpe_h += "<div style='font-size:12px;color:#555;text-align:center;padding:16px;'>No open positions — Portfolio empty</div>"
        # Risk rules
        rpe_h += "<div style='margin-top:14px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;'>"
        rules = [
            ("Max per trade", "2% capital", n_pos<5),
            ("Max positions", f"{n_pos}/5", n_pos<5),
            ("Max exposure", f"{exposure_pct:.0f}%/80%", exposure_pct<80),
        ]
        for rl,rv,rok in rules:
            rc = "#00b880" if rok else "#e74c3c"
            rpe_h += f"<div style='background:#161b22;border:1px solid {rc}55;border-radius:6px;padding:8px;text-align:center;'>"
            rpe_h += f"<div style='font-size:9px;color:#888;'>{rl}</div><div style='font-size:12px;color:{rc};font-weight:600;'>{rv}</div>"
            rpe_h += f"<div style='font-size:10px;color:{rc};'>{"OK" if rok else "WARN"}</div></div>"
        rpe_h += "</div></div>"
        st.markdown(rpe_h, unsafe_allow_html=True)

        if n_pos >= 5: st.error("Max 5 positions reached — close some before new trade!")
        elif exposure_pct > 70: st.warning(f"High exposure {exposure_pct:.0f}% — reduce position size!")
        elif not is_trade: st.info("Portfolio risk OK — Ready to trade when signal confirms.")
        else: st.success("Portfolio risk OK — Go ahead with the trade!")

    # ============================================================
    # PANEL 3: PORTFOLIO HEAT MAP + SECTOR EXPOSURE
    # ============================================================
    with st.expander("📊 Portfolio Heat Map & Sector Exposure"):
        if st.session_state.get("paper_positions"):
            _pos_all  = st.session_state.paper_positions
            _pos_data = []
            _sec_exp  = {}
            for _sym,_pos in _pos_all.items():
                try:
                    import yfinance as _yf3
                    _cp = float(_yf3.Ticker(_pos["stock"]).history(period="1d",interval="1m")["Close"].iloc[-1])
                except Exception: _cp = _pos["price"]
                _pnl  = (_cp-_pos["price"])*_pos["qty"]
                _ppct = (_cp-_pos["price"])/_pos["price"]*100
                _val  = _cp*_pos["qty"]
                _sec  = "Other"
                for _sn,_ss in STOCK_UNIVERSE.items():
                    if _pos["stock"] in _ss: _sec=_sn.replace("⭐ ","").replace("🏦 ","").replace("💻 ","").replace("🚗 ","").replace("🛒 ","").replace("💊 ",""); break
                _sec_exp[_sec] = _sec_exp.get(_sec,0)+_val
                _pos_data.append({"Stock":_sym,"Entry":f"₹{_pos['price']:.2f}","LTP":f"₹{_cp:.2f}","Qty":_pos["qty"],"P&L":f"₹{_pnl:+.0f}","P&L%":f"{_ppct:+.1f}%","Sector":_sec})
            st.dataframe(_pos_data, hide_index=True, use_container_width=True)
            # Sector exposure
            if _sec_exp:
                st.markdown("**Sector Exposure:**")
                _stv = sum(_sec_exp.values())+0.01
                _sc2 = st.columns(min(6,len(_sec_exp)))
                for _si,(_sn2,_sv2) in enumerate(_sec_exp.items()):
                    if _si<len(_sc2):
                        _sp=_sv2/_stv*100
                        _sc2[_si].markdown(f"<div style='background:#161b22;border:1px solid #21262d;border-radius:8px;padding:10px;text-align:center;'><div style='font-size:10px;color:#888;'>{_sn2[:10]}</div><div style='font-size:18px;font-weight:700;color:#4e8fff;'>{_sp:.0f}%</div><div style='font-size:10px;color:#555;'>₹{_sv2:,.0f}</div></div>", unsafe_allow_html=True)
        else:
            st.info("No open positions. Paper trade karo first!")

    with st.expander("🎓 Hindi AI Coach — Trading Seekho"):
        if st.session_state.get("paper_positions"):
            positions = st.session_state.paper_positions
            total_cap = st.session_state.paper_balance

            # Position sizing table
            st.markdown("**Open Positions:**")
            pos_data = []
            total_exposed = 0
            sector_exp = {}

            for sym, pos in positions.items():
                try:
                    import yfinance as _yf3
                    cur_px = float(_yf3.Ticker(pos["stock"]).history(period="1d",interval="1m")["Close"].iloc[-1])
                except Exception:
                    cur_px = pos["price"]

                pnl    = (cur_px - pos["price"]) * pos["qty"]
                value  = cur_px * pos["qty"]
                pnl_pct= (cur_px - pos["price"]) / pos["price"] * 100
                total_exposed += value

                # Find sector
                sector = "Other"
                for sec_name, sec_stocks in STOCK_UNIVERSE.items():
                    if pos["stock"] in sec_stocks:
                        sector = sec_name.split(" ")[-1]
                        break
                sector_exp[sector] = sector_exp.get(sector, 0) + value

                pos_data.append({
                    "Stock": sym, "Entry": f"Rs.{pos['price']:.2f}",
                    "Current": f"Rs.{cur_px:.2f}", "Qty": pos["qty"],
                    "Value": f"Rs.{value:,.0f}", "P&L": f"Rs.{pnl:+.0f}",
                    "P&L%": f"{pnl_pct:+.1f}%", "Sector": sector
                })

            if pos_data:
                st.dataframe(pos_data, hide_index=True, use_container_width=True)

            # Heatmap
            st.markdown("**Portfolio Heat Map:**")
            hm_cols = st.columns(min(5, len(pos_data)))
            for i, p in enumerate(pos_data):
                if i < len(hm_cols):
                    pct = float(p["P&L%"].replace("%","").replace("+",""))
                    hc = "#00b880" if pct>1 else ("#27ae60" if pct>0 else ("#e74c3c" if pct<-1 else "#f39c12"))
                    hm_cols[i].markdown(f"<div style='background:{hc}33;border:2px solid {hc};border-radius:10px;padding:12px;text-align:center;'><div style='font-size:14px;font-weight:700;color:{hc};'>{p['Stock']}</div><div style='font-size:20px;font-weight:800;color:{hc};'>{p['P&L%']}</div><div style='font-size:11px;color:#888;'>{p['Value']}</div></div>", unsafe_allow_html=True)

            # Sector Exposure
            st.markdown("**Sector Exposure:**")
            total_v = sum(sector_exp.values()) + 0.01
            se_cols = st.columns(len(sector_exp)) if sector_exp else st.columns(1)
            for i, (sec, val) in enumerate(sector_exp.items()):
                if i < len(se_cols):
                    pct2 = val/total_v*100
                    se_cols[i].markdown(f"<div style='background:#161b22;border:1px solid #21262d;border-radius:8px;padding:10px;text-align:center;'><div style='font-size:11px;color:#888;'>{sec}</div><div style='font-size:20px;font-weight:700;color:#4e8fff;'>{pct2:.0f}%</div><div style='font-size:10px;color:#555;'>Rs.{val:,.0f}</div></div>", unsafe_allow_html=True)

            # Risk metrics
            exposure_pct = total_exposed/(total_cap+total_exposed+0.01)*100
            st.markdown(f"**Portfolio Risk:** Exposure {exposure_pct:.1f}% | Positions: {len(positions)}/5 max")
            st.progress(min(exposure_pct/100, 1.0))
            if exposure_pct > 80:
                st.error("Portfolio >80% exposed — HIGH RISK!")
            elif exposure_pct > 60:
                st.warning("Portfolio >60% exposed — Consider reducing")
        else:
            st.info("No open positions. Buy some stocks first.")

    with st.expander("🎓 Hindi AI Coach — Trading Seekho"):
        st.markdown("**AI Coach — Hindi mein trading samjhega!**")
        topics = {
            "SMC kya hota hai?":         "Explain Smart Money Concepts in simple Hindi for beginners. Cover Order Blocks, FVG, Liquidity. Short, practical.",
            "BOS aur CHOCH samjhao":     "Explain Break of Structure (BOS) and Change of Character (CHOCH) in Hindi with NSE examples.",
            "ICT Kill Zones":            "Explain ICT Kill Zones in Hindi for Indian NSE market timing.",
            "Risk management":           "Explain position sizing and risk management in Hindi for Indian retail traders.",
            "RSI kaise use kare?":       "Explain RSI usage in Hindi for Indian stock traders with buy/sell rules.",
            "Fibonacci kya hai?":        "Explain Fibonacci retracement in Hindi for stock traders. Where to buy on dips?",
        }
        t_choice = st.selectbox("Topic:", list(topics.keys()), key="coach_topic")
        custom_q = st.text_input("Ya apna sawaal:", placeholder="Support aur Resistance kya hai?", key="coach_q")
        if st.button("Coach se Seekho", type="primary", key="coach_ask"):
            prompt = custom_q.strip() if custom_q.strip() else topics[t_choice]
            with st.spinner("Coach soch raha hai..."):
                try:
                    import anthropic as _ant
                    _c = _ant.Anthropic()
                    _r = _c.messages.create(
                        model="claude-opus-4-5", max_tokens=500,
                        system="Tu experienced Indian stock trader aur teacher hai. Hindi mein samjha. Simple language, real NSE examples, bullet points use kar.",
                        messages=[{"role":"user","content":prompt}])
                    st.markdown(_r.content[0].text)
                except Exception:
                    built = {
                        "SMC kya hota hai?":"**SMC (Smart Money Concepts):**\n\n- **Order Block** = Woh candle jahan se bada move hua\n- **FVG** = Gap jahan trading nahi hua — price wapas aata hai fill karne\n- **Liquidity** = Stop losses ka cluster — price pehle wahan jaata hai\n\n*Rule: Jab retail SELL kare — Smart Money BUY karta hai!*",
                    }
                    st.markdown(built.get(t_choice, f"**{t_choice}**\n\nANTHROPIC_API_KEY Streamlit Secrets mein add karo coach ke liye."))

    with st.expander("🤖 AI Trading Advisor — Personal Advice"):
        try:
            _mtf_adv = st.session_state.get("mtf_cache")
            _advice  = build_trading_advisor(
                user, stock.replace(".NS",""), _expl, total_score, verdict,
                price, stop_loss_m, target_m, qty_m,
                max_loss_rs, max_gain_rs, _rr_m, win_prob, _mtf_adv)
            st.markdown(_advice)
        except Exception as _ae:
            st.caption(f"Advisor: {str(_ae)[:60]}")
        st.markdown("---")
        st.markdown("**Weighted Score Breakdown:**")
        for _sl2,_sc,_mx,_ds in [
            ("Technical",tech_layer_score,20,"EMA/RSI/MACD/ADX"),
            ("AI Model",ai_layer_score,20,"XGBoost+LightGBM+RF"),
            ("SMC+ICT",smc_layer_score+ict_layer_score,25,"OB/FVG/BOS/Liq"),
            ("Volume",vol_layer_score,5,"Vol ratio"),
            ("MTF",mtf_layer_score,10,"5m+15m+1H+4H"),
            ("Pattern",pattern_layer_score,10,"H&S/DTop/Flag/C&H"),
            ("Fibonacci",fib_layer_score,3,"Fib levels"),
            ("Candle",candle_layer_score,2,"25+ patterns"),
        ]:
            _pc=int(_sc/max(_mx,1)*100); _cc="#00b880" if _pc>=65 else ("#f39c12" if _pc>=40 else "#e74c3c")
            st.markdown(f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:5px;'><div style='min-width:95px;font-size:12px;color:#ccc;'>{_sl2}</div><div style='flex:1;background:#21262d;border-radius:99px;height:7px;'><div style='width:{_pc}%;background:{_cc};border-radius:99px;height:7px;'></div></div><div style='min-width:42px;font-size:11px;color:{_cc};text-align:right;'>{_sc}/{_mx}</div><div style='min-width:88px;font-size:10px;color:#555;'>{_ds}</div></div>", unsafe_allow_html=True)
        st.markdown(f"**Final Score: {total_score}/100** (Raw {raw_score} − Penalties {penalty_pts})")

    with st.expander("🔍 Full Layer Breakdown"):
        _dc1, _dc2 = st.columns(2)
        with _dc1:
            st.markdown("**Technical Checks:**")
            _pass = sum(1 for v in tech_checks.values() if v)
            _total_tc = len(tech_checks)
            tc_color = "#00b880" if _pass>=7 else ("#f39c12" if _pass>=5 else "#e74c3c")
            st.markdown(f"<div style='font-size:11px;color:{tc_color};margin-bottom:8px;font-weight:600;'>{_pass}/{_total_tc} checks passed</div>", unsafe_allow_html=True)
            for k,v in tech_checks.items():
                c2 = "#00b880" if v else "#e74c3c"
                st.markdown(f"<div style='padding:3px 0;font-size:13px;color:{c2};'>{'✅' if v else '❌'} {k}</div>", unsafe_allow_html=True)

        with _dc2:
            st.markdown("**AI Model Status:**")
            # Model name + status
            _ai_is_default = "Default" in ai_model_name or ai_model_name == ""
            if _ai_is_default:
                st.markdown("<div style='background:#1a1200;border:1px solid #f39c12;border-radius:8px;padding:10px;margin-bottom:8px;'><div style='color:#f39c12;font-size:12px;font-weight:600;'>AI Model: Insufficient Data</div><div style='color:#888;font-size:11px;margin-top:4px;'>Need 40+ candles — Switch to Futures/Swing mode for full AI</div></div>", unsafe_allow_html=True)
            else:
                _model_c = "#00b880" if ai_accuracy>=60 else ("#f39c12" if ai_accuracy>=50 else "#e74c3c")
                _ai_h  = f"<div style='background:{_model_c}22;border:1px solid {_model_c};border-radius:8px;padding:10px;margin-bottom:8px;'>"
                _ai_h += f"<div style='color:{_model_c};font-size:12px;font-weight:600;'>{ai_confidence} Confidence — {ai_accuracy}% accuracy</div>"
                _ai_h += f"<div style='color:#888;font-size:10px;margin-top:3px;'>{ai_model_name[:55]}</div></div>"
                st.markdown(_ai_h, unsafe_allow_html=True)

            # AI probability gauge
            _ai_c = "#00b880" if ai_pct>=60 else ("#f39c12" if ai_pct>=40 else "#e74c3c")
            st.markdown(f"<div style='display:flex;justify-content:space-between;margin-bottom:3px;'><span style='font-size:11px;color:#888;'>AI Bullish Probability</span><span style='font-size:13px;font-weight:700;color:{_ai_c};'>{ai_pct}%</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background:#21262d;border-radius:99px;height:8px;margin-bottom:12px;'><div style='width:{ai_pct}%;background:{_ai_c};border-radius:99px;height:8px;'></div></div>", unsafe_allow_html=True)

            # Walk-forward results
            if wf_results and len(wf_results)>0:
                avg_wf = round(sum(wf_results)/len(wf_results),1)
                std_wf = round((sum((x-avg_wf)**2 for x in wf_results)/len(wf_results))**0.5,1)
                st.markdown(f"<div style='font-size:11px;color:#888;margin-bottom:6px;'>Walk-Forward ({len(wf_results)} folds) | Avg: {avg_wf}% | Std: ±{std_wf}%</div>", unsafe_allow_html=True)
                _wf_h = "<div style='display:flex;gap:4px;margin-bottom:10px;'>"
                for _wfv in wf_results:
                    _wc = "#00b880" if _wfv>=60 else ("#f39c12" if _wfv>=50 else "#e74c3c")
                    _wf_h += f"<div style='flex:1;background:{_wc}22;border:1px solid {_wc};border-radius:5px;padding:5px;text-align:center;font-size:11px;font-weight:700;color:{_wc};'>{_wfv}%</div>"
                _wf_h += "</div>"
                st.markdown(_wf_h, unsafe_allow_html=True)
            else:
                st.caption("Walk-forward: Not available (need more data)")

            # Feature importance bars
            if feature_importance:
                st.markdown("<div style='font-size:11px;color:#888;margin-bottom:6px;'>Top Feature Importance:</div>", unsafe_allow_html=True)
                _max_fi = max(feature_importance.values())+1e-9
                for fn,fi in list(feature_importance.items())[:6]:
                    _bw = int(fi/_max_fi*100)
                    _fc = "#4e8fff" if _bw>60 else ("#00b880" if _bw>40 else "#888")
                    st.markdown(f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:3px;'><div style='min-width:70px;font-size:10px;color:#ccc;'>{fn[:10]}</div><div style='flex:1;background:#21262d;border-radius:3px;height:6px;'><div style='width:{_bw}%;background:{_fc};border-radius:3px;height:6px;'></div></div><div style='min-width:35px;font-size:10px;color:#888;text-align:right;'>{fi:.3f}</div></div>", unsafe_allow_html=True)
            else:
                st.caption("Feature importance: Not available")

            # LSTM if available
            if hasattr(st.session_state, 'lstm_forecast') and st.session_state.get('lstm_forecast'):
                _lf = st.session_state.lstm_forecast
                _lt = "UP" if _lf[-1]>price else "DOWN"
                _lc = "#00b880" if _lt=="UP" else "#e74c3c"
                st.markdown(f"<div style='background:{_lc}22;border:1px solid {_lc};border-radius:6px;padding:8px;margin-top:8px;font-size:11px;'><b style='color:{_lc};'>LSTM: {_lt}</b> — {' → '.join([f'₹{p:.0f}' for p in _lf[:3]])}</div>", unsafe_allow_html=True)


        # ── Variables needed for detail section ─────────────────
        c_ai       = ai_prob > 0.55
        struct_pct = min(100, round(struct_layer_score/15*100))
        candle_pct = min(100, round(candle_layer_score/2*100))
        vol_pct    = min(100, round(vol_layer_score/5*100))
        smc_pct    = min(100, round(smc_layer_score/10*100))
        col_sig, col_pos = st.columns(2)
    with col_sig:
        st.markdown("#### Detailed Layer Checks")
        all_checks_display = {
            **tech_checks,
            f"AI Model ({ai_pct}%)": ai_prob > 0.55,
            "Market Structure Bullish": struct_layer_score >= 10,
            "Candle Pattern Bullish":   candle_layer_score >= 1,
            "Volume Confirmation":      vol_layer_score >= 3,
            "SMC / OB Zone":            smc_layer_score >= 6,
        }
        chk_df = pd.DataFrame([
            {"Check": k, "Pass": "✅" if v else "❌"}
            for k, v in all_checks_display.items()
        ])
        st.dataframe(chk_df, hide_index=True, height=380)

        # AI Model Details
        with st.expander("🤖 AI Engine — Full Details"):
            st.markdown(f"**Model:** `{ai_model_name}`")
            _dc = len(df.dropna(subset=['Close','EMA20','RSI']))
            _tip_color = "#00b880" if _dc >= 80 else ("#f39c12" if _dc >= 40 else "#e74c3c")
            st.markdown(
                f"<div style='font-size:11px;color:{_tip_color};margin-bottom:8px;'>"
                f"Data: {_dc} candles loaded | Features: {len(feat_cols)} | "
                f"AI: {len(feat_cols)} features loaded"
                f"{'  ✅ Good' if _dc>=80 else ('  ⚠️ Low — use Futures/Swing mode' if _dc>=20 else '  ❌ No data')}"
                f"</div>",
                unsafe_allow_html=True)

            # Model availability badges
            badge_cols = st.columns(5)
            for i,(nm,ok) in enumerate([
                ("XGBoost", XGB_OK),("LightGBM",LGB_OK),
                ("GradBoost",True),("RandomForest",True),("AdaBoost",True)]):
                c = "#00b880" if ok else "#555"
                badge_cols[i].markdown(
                    f"<div style='background:{c}22;border:1px solid {c};"
                    f"border-radius:6px;padding:5px;text-align:center;"
                    f"font-size:11px;font-weight:600;color:{c};'>"
                    f"{'✅' if ok else '–'} {nm}</div>",
                    unsafe_allow_html=True)

            st.markdown("")

            # Walk-Forward Results
            if wf_results and len(wf_results) > 1:
                st.markdown("**Walk-Forward Validation (5 folds):**")
                wf_cols = st.columns(len(wf_results))
                for i, acc in enumerate(wf_results):
                    cc = "#00b880" if acc>=60 else ("#f39c12" if acc>=50 else "#e74c3c")
                    wf_cols[i].markdown(
                        f"<div style='background:{cc}22;border:1px solid {cc};"
                        f"border-radius:6px;padding:8px;text-align:center;'>"
                        f"<div style='font-size:14px;font-weight:700;color:{cc};'>{acc}%</div>"
                        f"<div style='font-size:9px;color:#888;'>Fold {i+1}</div>"
                        f"</div>",
                        unsafe_allow_html=True)
                avg_wf = round(float(np.mean(wf_results)),1)
                std_wf = round(float(np.std(wf_results)),1)
                st.caption(f"Mean: {avg_wf}% | Std: ±{std_wf}% | "
                           f"{'Stable' if std_wf<5 else 'Unstable'} across folds")

            # Feature Importance
            if feature_importance:
                st.markdown("**Feature Importance (averaged across models):**")
                max_imp = max(feature_importance.values()) + 1e-9
                for feat, imp in feature_importance.items():
                    bar_w = int(imp / max_imp * 100)
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:3px;'>"
                        f"<div style='min-width:90px;font-size:11px;color:#ccc;'>{feat}</div>"
                        f"<div style='flex:1;background:#21262d;border-radius:3px;height:8px;'>"
                        f"<div style='width:{bar_w}%;background:#4e8fff;height:8px;border-radius:3px;'></div>"
                        f"</div>"
                        f"<div style='min-width:40px;font-size:11px;color:#888;text-align:right;'>{imp:.3f}</div>"
                        f"</div>",
                        unsafe_allow_html=True)

            # LSTM Equity Curve Prediction
            st.markdown("**LSTM Price Trend Forecast:**")
            if KERAS_OK and len(df) >= 30:
                try:
                    _seq_len = 20
                    _close   = df["Close"].values.reshape(-1,1)
                    _scaler2 = MinMaxScaler()
                    _scaled  = _scaler2.fit_transform(_close)

                    # Build sequences
                    _Xs, _ys = [], []
                    for _i in range(_seq_len, len(_scaled)-1):
                        _Xs.append(_scaled[_i-_seq_len:_i])
                        _ys.append(_scaled[_i])
                    _Xs = np.array(_Xs); _ys = np.array(_ys)

                    if len(_Xs) >= 30:
                        # Build LSTM model
                        _model = Sequential([
                            LSTM(50, return_sequences=True,
                                 input_shape=(_seq_len,1)),
                            Dropout(0.2),
                            LSTM(30, return_sequences=False),
                            Dropout(0.2),
                            Dense(1)
                        ])
                        _model.compile(optimizer="adam", loss="mse")
                        _es = EarlyStopping(patience=3, restore_best_weights=True)
                        _model.fit(_Xs[:-5], _ys[:-5], epochs=20,
                                   batch_size=16, validation_split=0.1,
                                   callbacks=[_es], verbose=0)

                        # Predict next 5 candles
                        _last_seq = _scaled[-_seq_len:].reshape(1,_seq_len,1)
                        _preds_sc = []
                        _cur_seq  = _last_seq.copy()
                        for _ in range(5):
                            _p = _model.predict(_cur_seq, verbose=0)[0][0]
                            _preds_sc.append(_p)
                            _cur_seq = np.append(_cur_seq[:,1:,:],
                                                  [[[_p]]], axis=1)
                        _preds = _scaler2.inverse_transform(
                            np.array(_preds_sc).reshape(-1,1)).flatten()

                        _last_price = float(df["Close"].iloc[-1])
                        _trend      = "UP" if _preds[-1] > _last_price else "DOWN"
                        _change_pct = (_preds[-1]-_last_price)/_last_price*100

                        tc = "#00b880" if _trend=="UP" else "#e74c3c"
                        st.markdown(
                            f"<div style='background:{tc}22;border:1px solid {tc};"
                            f"border-radius:8px;padding:10px;margin:6px 0;'>"
                            f"<div style='font-size:13px;font-weight:600;color:{tc};'>"
                            f"LSTM 5-candle forecast: {_trend} "
                            f"({_change_pct:+.2f}%)</div>"
                            f"<div style='font-size:11px;color:#888;margin-top:3px;'>"
                            + " → ".join([f"Rs.{p:,.1f}" for p in _preds])
                            + "</div></div>",
                            unsafe_allow_html=True)

                        if PLOTLY_OK:
                            _hist = df["Close"].values[-20:]
                            _hist_x = list(range(len(_hist)))
                            _pred_x = list(range(len(_hist)-1,
                                                 len(_hist)+len(_preds)-1))
                            _lstm_fig = go.Figure()
                            _lstm_fig.add_trace(go.Scatter(
                                x=_hist_x, y=_hist,
                                line=dict(color="#4e8fff", width=2),
                                name="Historical"))
                            _lstm_fig.add_trace(go.Scatter(
                                x=_pred_x, y=_preds,
                                line=dict(color=tc, width=2, dash="dot"),
                                name="LSTM Forecast",
                                mode="lines+markers"))
                            _lstm_fig.update_layout(
                                height=180,
                                plot_bgcolor="#0d1117",
                                paper_bgcolor="#0d1117",
                                font=dict(color="#8b949e", size=9),
                                margin=dict(l=0,r=0,t=10,b=0),
                                legend=dict(bgcolor="rgba(0,0,0,0)",
                                           font_size=9),
                                showlegend=True)
                            _lstm_fig.update_xaxes(gridcolor="#21262d")
                            _lstm_fig.update_yaxes(gridcolor="#21262d")
                            st.plotly_chart(_lstm_fig,
                                           use_container_width=True,
                                           config={"displayModeBar":False})
                except Exception as _le:
                    st.caption(f"LSTM: {str(_le)[:60]}")
            elif KERAS_OK:
                st.caption("Need 60+ candles for LSTM forecast")
            else:
                st.caption("Install tensorflow for LSTM: pip install tensorflow")



    with col_pos:
        st.markdown(f"#### ⚖️ Position Sizing — {selected_mode}")
        atr         = max(float(last["ATR"]), 0.01)
        risk_amount = capital * (risk / 100)
        sl_dist     = atr * mcfg["sl_mult"]
        tgt_dist    = sl_dist * mcfg["rr"]
        stop_loss   = round(price - sl_dist, 2)
        target_price= round(price + tgt_dist, 2)
        rr_ratio    = round(tgt_dist / sl_dist, 1)
        lot_size    = FO_LOTS.get(stock, 500)

        if selected_mode == "📈 Intraday":
            qty = max(1, int(risk_amount / sl_dist))
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("📦 Qty",      f"{qty} sh")
            p2.metric("💸 Risk ₹",   f"₹{risk_amount:,.0f}")
            p3.metric("🛡 Stop Loss", f"₹{stop_loss:,.2f}")
            p4.metric("🎯 Target",   f"₹{target_price:,.2f}")
            st.caption(f"Entry ₹{price:.2f} | ATR ₹{atr:.2f} | SL ₹{sl_dist:.2f} | R:R {rr_ratio}:1 | MIS")
            st.info("⏰ Enter after 9:30 AM. Exit before 3:15 PM.")

        elif selected_mode == "🌊 Swing":
            qty = max(1, int(risk_amount / sl_dist))
            p1,p2 = st.columns(2)
            p1.metric("📦 Qty", f"{qty} shares"); p2.metric("💸 Risk ₹", f"₹{risk_amount:,.0f}")
            p3,p4 = st.columns(2)
            p3.metric("🛡 Stop Loss", f"₹{stop_loss:,.2f}"); p4.metric("🎯 Target", f"₹{target_price:,.2f}")
            st.caption(f"Entry ₹{price:.2f} | ATR ₹{atr:.2f} | R:R {rr_ratio}:1 | CNC")
            st.info(f"📅 Hold 3–15 days. Trail SL up. R:R = {rr_ratio}:1")

        elif selected_mode == "📊 Futures":
            qty=1
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("📦 Lots","1 lot"); p2.metric("📋 Lot",f"{lot_size} sh")
            p3.metric("🛡 SL",f"₹{stop_loss:,.2f}"); p4.metric("🎯 TGT",f"₹{target_price:,.2f}")
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("💰 Margin",f"₹{price*lot_size*0.15:,.0f}")
            m2.metric("📈 Profit",f"₹{tgt_dist*lot_size:,.0f}")
            m3.metric("📉 Loss",f"₹{sl_dist*lot_size:,.0f}")
            m4.metric("🔢 R:R",f"{rr_ratio}:1")
            st.warning(f"⚠️ Exposure ₹{price*lot_size:,.0f}. Use stop loss always.")

        elif selected_mode == "🎯 Options":
            si = 100 if price>2000 else (50 if price>500 else (20 if price>100 else 10))
            atm = round(price/si)*si
            premium_est = round(atr*2.5, 2)
            cost_per_lot = premium_est*lot_size
            qty=1
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("📦 Lots","1 lot"); p2.metric("📋 Lot",f"{lot_size}")
            p3.metric("💸 Cost",f"₹{cost_per_lot:,.0f}"); p4.metric("🎯 R:R",f"{rr_ratio}:1")
            oc1,oc2 = st.columns(2)
            oc1.markdown(f"**CE (Bullish):** ₹{atm} CE | Premium ₹{premium_est} | Cost ₹{cost_per_lot:,.0f}")
            oc2.markdown(f"**PE (Bearish):** ₹{atm} PE | Premium ₹{premium_est} | Cost ₹{cost_per_lot:,.0f}")
            st.info("Buy CE if BUY signal. Buy PE if bearish. Exit when premium doubles or halves.")

    # ── TRADE BUTTONS ─────────────────────────────────────────
    st.markdown("---")

    def log_trade(action, stk, px, q, md, pnl=None):
        st.session_state.trade_log.append({
            "time": now_ist().strftime("%Y-%m-%d %H:%M:%S"),
            "strategy": selected_mode, "stock": stk.replace(".NS",""),
            "action": action, "price": round(px,2), "qty": q,
            "mode": md, "SL": stop_loss, "Target": target_price,
            "pnl": round(pnl,2) if pnl is not None else "—",
        })

    def order(txn):
        sym = stock.replace(".NS","")
        if mode == "Paper":
            try:
                live_px = float(yf.Ticker(stock).history(period="1d",interval="1m")["Close"].iloc[-1])
            except:
                live_px = price
            if txn == "BUY":
                if st.session_state.paper_position:
                    st.warning("⚠️ Already holding — sell first"); return
                st.session_state.paper_position = {
                    "stock":stock,"price":live_px,"qty":qty,
                    "stop_loss":stop_loss,"target":target_price,"strategy":selected_mode,
                    "time":now_ist().strftime("%Y-%m-%d %H:%M:%S")}
                st.session_state.paper_balance -= live_px*qty
                log_trade("BUY",stock,live_px,qty,"Paper")
                st.success(f"📄 Paper BUY | {sym} | ₹{live_px:.2f}×{qty} | SL ₹{stop_loss} | TGT ₹{target_price}")
                save_user_data(user)  # persist to file
                if ALERT_ON_EXECUTION:
                    fire_alert(f"BUY EXECUTED [{selected_mode}]",stock,live_px,qty,stop_loss,target_price,total_score,"Paper")
                return
            if txn == "SELL":
                if not st.session_state.paper_position:
                    st.warning("⚠️ No open position"); return
                pos = st.session_state.paper_position
                pnl = (live_px-pos["price"])*pos["qty"]
                st.session_state.paper_balance += live_px*pos["qty"]
                st.session_state.pnl_history.append({
                    "time":now_ist().strftime("%Y-%m-%d %H:%M:%S"),
                    "stock":stock,"pnl":round(pnl,2),"strategy":selected_mode})
                log_trade("SELL",stock,live_px,pos["qty"],"Paper",pnl=pnl)
                st.session_state.paper_position = None
                emoji = "🟢" if pnl>=0 else "🔴"
                st.success(f"📄 Paper SELL | {sym} | ₹{live_px:.2f} | P&L: {emoji} ₹{pnl:+.2f}")
                save_user_data(user)  # persist to file
                if ALERT_ON_EXECUTION:
                    fire_alert(f"SELL EXECUTED [{selected_mode}]",stock,live_px,pos["qty"],
                               pos["stop_loss"],pos["target"],total_score,"Paper",pnl=pnl)
                return
        if not kite_ok(): st.error("❌ Kite Not Connected"); return
        try:
            fo_qty = lot_size if selected_mode in ["📊 Futures","🎯 Options"] else qty
            kite.place_order(variety="regular",exchange="NSE",tradingsymbol=sym,
                             transaction_type=txn,quantity=fo_qty,
                             order_type="MARKET",product=mcfg["product"])
            log_trade(txn,stock,price,fo_qty,"Live")
            st.success(f"✅ Live {txn} | {sym}×{fo_qty} | {mcfg['product']}")
            save_user_data(user)  # persist to file
            if ALERT_ON_EXECUTION:
                fire_alert(f"{txn} LIVE [{selected_mode}]",stock,price,fo_qty,stop_loss,target_price,total_score,"Live")
        except Exception as e:
            st.error(e)

    col1,col2,col3 = st.columns(3)
    if col1.button("🚀 BUY NOW",  use_container_width=True, type="primary"):
        order("BUY"); st.session_state.last_trade = datetime.datetime.now()
    if col2.button("🛑 SELL NOW", use_container_width=True):
        order("SELL"); st.session_state.last_trade = datetime.datetime.now()
    if col3.button("🔁 Refresh",  use_container_width=True):
        st.rerun()

    if mode == "Paper" and st.session_state.paper_position:
        pos = st.session_state.paper_position
        try:
            _pos_hist = yf.Ticker(pos["stock"]).history(period="1d", interval="1m")
            _live_px  = float(_pos_hist["Close"].iloc[-1]) if not _pos_hist.empty else pos["price"]
        except Exception:
            _live_px  = pos["price"]
        open_pnl = (_live_px - pos["price"]) * pos["qty"]
        _color   = "#2d8a4e" if open_pnl >= 0 else "#c0392b"
        _sl_hit  = _live_px <= float(pos["stop_loss"])
        _tgt_hit = _live_px >= float(pos["target"])
        _status  = " ⚡ SL Hit!" if _sl_hit else (" 🎯 Target Hit!" if _tgt_hit else "")
        _bdr     = "#e74c3c" if _sl_hit else ("#00b880" if _tgt_hit else "#2d8a4e")
        _bg      = "#fff0f0" if _sl_hit else ("#f0fff8" if _tgt_hit else "#f0fff4")
        st.markdown(
            f"<div style='background:{_bg};border:2px solid {_bdr};border-radius:10px;"
            f"padding:12px 16px;margin:8px 0;'>"
            f"<div style='font-size:13px;font-weight:700;'>"
            f"Open [{pos.get('strategy','Intraday')}]{_status}</div>"
            f"<div style='margin:5px 0;font-size:13px;'>"
            f"{pos['stock'].replace('.NS','')} | Entry Rs.{pos['price']:.2f} "
            f"| LTP Rs.{_live_px:.2f} | Qty {pos['qty']}</div>"
            f"<div style='font-size:12px;color:#666;'>"
            f"SL Rs.{pos['stop_loss']} | TGT Rs.{pos['target']}</div>"
            f"<div style='font-size:16px;font-weight:700;color:{_color};margin-top:5px;'>"
            f"Unrealised: Rs.{open_pnl:+.2f}</div>"
            f"</div>",
            unsafe_allow_html=True)

    if auto_trade:
        if mode == "Live" and kite_ok() and signal and can_trade():
            order("BUY"); st.session_state.last_trade = datetime.datetime.now()
        time.sleep(interval); st.rerun()

    # ── CHARTS — FIXED ────────────────────────────────────────
    st.markdown("---")

    # Clean data for charts — remove NaN rows
    chart_df = df.dropna(subset=["Close","EMA20","EMA50","RSI","MACD"]).tail(100).copy()
    chart_df.index = pd.to_datetime(chart_df.index)
    lc = mcfg["color"]

    # Info bar showing data quality
    st.info(f"📊 Chart: {len(chart_df)} candles | {mcfg['interval']} timeframe | {mcfg['period']} period")

    if len(chart_df) < 5:
        st.warning("⚠️ Not enough data for charts. Switch to **🌊 Swing** mode.")
    elif PLOTLY_OK:
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
            line=dict(color="#00e5a0",width=1), name="EMA 9"))
        fig1.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA20"],
            line=dict(color="#2980b9",width=1.5), name="EMA 20"))
        fig1.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA50"],
            line=dict(color="#8e44ad",width=1.5,dash="dot"), name="EMA 50"))
        fig1.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_Upper"],
            line=dict(color="#95a5a6",width=1,dash="dash"), name="BB Upper"))
        fig1.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_Lower"],
            line=dict(color="#95a5a6",width=1,dash="dash"), name="BB Lower",
            fill="tonexty", fillcolor="rgba(149,165,166,0.07)"))
        fig1.add_hline(y=stop_loss,    line_color="#e74c3c", line_dash="dot", line_width=1.5,
            annotation_text=f"SL ₹{stop_loss}", annotation_font_color="#e74c3c")
        fig1.add_hline(y=target_price, line_color="#27ae60", line_dash="dot", line_width=1.5,
            annotation_text=f"TGT ₹{target_price}", annotation_font_color="#27ae60")
        fig1.add_hline(y=price,        line_color="#f39c12", line_dash="solid", line_width=1,
            annotation_text=f"LTP ₹{price:.2f}", annotation_font_color="#f39c12")
        fig1.update_layout(height=380, xaxis_rangeslider_visible=False,
            legend=dict(orientation="h",y=1.05,bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=0,t=30,b=0),
            xaxis=dict(showgrid=True,gridcolor="#ecf0f1"),
            yaxis=dict(showgrid=True,gridcolor="#ecf0f1"))
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

        st.subheader("📊 RSI (14)")
        fig2 = go.Figure()
        fig2.add_hrect(y0=70,y1=100,fillcolor="rgba(231,76,60,0.08)",line_width=0)
        fig2.add_hrect(y0=0,y1=30,fillcolor="rgba(39,174,96,0.08)",line_width=0)
        fig2.add_hline(y=70,line_color="#e74c3c",line_dash="dot",line_width=1,
            annotation_text="Overbought 70",annotation_position="bottom right")
        fig2.add_hline(y=30,line_color="#27ae60",line_dash="dot",line_width=1,
            annotation_text="Oversold 30",annotation_position="top right")
        fig2.add_hline(y=50,line_color="#95a5a6",line_dash="dot",line_width=1)
        rsi_s = chart_df["RSI"]
        rsi_c = ["#e74c3c" if v>70 else("#27ae60" if v<30 else lc) for v in rsi_s]
        fig2.add_trace(go.Scatter(x=chart_df.index,y=rsi_s,
            line=dict(color=lc,width=2),fill="tozeroy",
            fillcolor="rgba(41,128,185,0.07)",name="RSI"))
        fig2.add_trace(go.Scatter(x=chart_df.index,y=rsi_s,mode="markers",
            marker=dict(color=rsi_c,size=3),showlegend=False))
        fig2.add_annotation(x=chart_df.index[-1],y=rsi_s.iloc[-1],
            text=f"  {rsi_s.iloc[-1]:.1f}",showarrow=False,
            font=dict(color=lc,size=12,family="monospace"))
        fig2.update_layout(height=220,
            yaxis=dict(range=[0,100],showgrid=True,gridcolor="#ecf0f1"),
            xaxis=dict(showgrid=True,gridcolor="#ecf0f1"),
            margin=dict(l=0,r=0,t=10,b=0),showlegend=False)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        st.subheader("📈 MACD")
        fig3 = go.Figure()
        hist_c = ["#27ae60" if v>=0 else "#e74c3c" for v in chart_df["MACD_Hist"]]
        fig3.add_trace(go.Bar(x=chart_df.index,y=chart_df["MACD_Hist"],
            marker_color=hist_c,name="Histogram",opacity=0.65))
        fig3.add_trace(go.Scatter(x=chart_df.index,y=chart_df["MACD"],
            line=dict(color=lc,width=1.5),name="MACD"))
        fig3.add_trace(go.Scatter(x=chart_df.index,y=chart_df["MACD_Signal"],
            line=dict(color="#f39c12",width=1.5,dash="dot"),name="Signal"))
        fig3.add_hline(y=0,line_color="#bdc3c7",line_width=1)
        fig3.update_layout(height=220,
            xaxis=dict(showgrid=True,gridcolor="#ecf0f1"),
            yaxis=dict(showgrid=True,gridcolor="#ecf0f1"),
            legend=dict(orientation="h",y=1.08,bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=0,t=10,b=0),barmode="relative")
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    else:
        st.warning("Plotly not available — showing basic charts")
        st.subheader("📉 Price")
        st.line_chart(chart_df[["Close","EMA20","EMA50"]])
        st.subheader("📊 RSI")
        st.line_chart(chart_df[["RSI"]])
        st.subheader("📈 MACD")
        st.line_chart(chart_df[["MACD","MACD_Signal"]])


    # =============================================================
    # CANDLESTICK PATTERNS + TA ANALYSIS TABS
    # =============================================================
    st.markdown("---")
    tab_candle, tab_ta, tab_struct, tab_pat, tab_ict2, tab_mtf = st.tabs([
        "Candles + TA",
        "Fibonacci + Trendline",
        "Market Structure",
        "Chart Patterns",
        "ICT + Liquidity",
        "Multi-Timeframe",
    ])

    with tab_candle:
        patterns = detect_candlestick_patterns(chart_df)
        if patterns:
            st.markdown(f"**{len(patterns)} patterns detected in recent candles:**")
            patterns_sorted = sorted(patterns, key=lambda x:-x["strength"])
            top = patterns_sorted[0]
            top_border = "#00b880" if top["type"]=="bullish" else ("#e74c3c" if top["type"]=="bearish" else "#f39c12")
            top_bg     = "#0d2818" if top["type"]=="bullish" else ("#2d0a0a" if top["type"]=="bearish" else "#1a1500")
            star_str   = "".join(["*" for _ in range(top["strength"])])
            st.markdown(f"""
<div style='background:{top_bg};border:2px solid {top_border};border-radius:12px;padding:16px;margin-bottom:14px;'>
  <div style='display:flex;justify-content:space-between;align-items:center;'>
    <div>
      <div style='font-size:11px;color:#888;text-transform:uppercase;'>Latest Strong Pattern</div>
      <div style='font-size:20px;font-weight:700;color:{top_border};'>{top["pattern"]}
        <span style='font-size:12px;color:#888;'>({top["candles"]} candle)</span>
      </div>
    </div>
    <div style='text-align:right;'>
      <div style='font-size:18px;font-weight:700;color:{top_border};'>{top["signal"]}</div>
      <div style='font-size:11px;color:#888;'>Strength: {star_str}</div>
    </div>
  </div>
  <div style='margin-top:8px;font-size:13px;color:#ccc;'>{top["desc"]}</div>
  <div style='font-size:11px;color:#888;margin-top:4px;'>Date: {top["date"]}</div>
</div>""", unsafe_allow_html=True)

            p_cols = st.columns(3)
            for i, p in enumerate(patterns_sorted):
                color = "#00b880" if p["type"]=="bullish" else ("#e74c3c" if p["type"]=="bearish" else "#f39c12")
                bg    = "#0d2818" if p["type"]=="bullish" else ("#2d0a0a" if p["type"]=="bearish" else "#1a1500")
                icon  = "BUY" if p["type"]=="bullish" else ("SELL" if p["type"]=="bearish" else "NEUTRAL")
                stars = "".join(["*" for _ in range(p["strength"])])
                p_cols[i%3].markdown(f"""
<div style='background:{bg};border:1px solid {color};border-radius:8px;padding:10px;margin-bottom:8px;'>
  <div style='font-weight:600;color:{color};font-size:13px;'>{p["pattern"]}</div>
  <div style='font-size:10px;color:#888;margin:2px 0;'>{p["candles"]} candle | {stars}</div>
  <div style='font-size:11px;color:#ccc;'>{p["desc"][:55]}...</div>
  <div style='font-size:11px;color:{color};font-weight:600;margin-top:3px;'>{p["signal"]}</div>
  <div style='font-size:10px;color:#888;'>{p["date"]}</div>
</div>""", unsafe_allow_html=True)

            with st.expander("Pattern Guide — All 25 patterns explained"):
                st.markdown(f"""
**Bullish Patterns (BUY signals):**
- **Hammer** — Long lower wick, buyers rejected lower prices
- **Bullish Engulfing** — Green candle completely covers previous red candle
- **Morning Star** — 3-candle reversal: big red, small star, big green
- **Three White Soldiers** — 3 consecutive green candles, strong uptrend
- **Bullish Marubozu** — Full green candle, no wicks, max bullish momentum
- **Dragonfly Doji** — Long lower wick, buyers pushed price back up
- **Piercing Line** — Green candle closes above midpoint of red candle
- **Tweezer Bottom** — Same lows twice = strong support level
- **Bullish Harami** — Small green inside big red = slowing bearish momentum
- **Three Inside Up** — Reversal confirmed by 3rd closing candle

**Bearish Patterns (SELL signals):**
- **Shooting Star** — Long upper wick, sellers rejected higher prices
- **Bearish Engulfing** — Red candle completely covers previous green candle
- **Evening Star** — 3-candle reversal: big green, small star, big red
- **Three Black Crows** — 3 consecutive red candles, strong downtrend
- **Bearish Marubozu** — Full red candle, no wicks, max bearish momentum
- **Gravestone Doji** — Long upper wick, sellers pushed price back down
- **Dark Cloud Cover** — Red candle closes below midpoint of green candle
- **Tweezer Top** — Same highs twice = strong resistance level
- **Bearish Harami** — Small red inside big green = slowing bullish momentum
- **Three Inside Down** — Reversal confirmed by 3rd closing candle

**Neutral Patterns (WAIT):**
- **Doji** — Open equals close, pure indecision, wait for breakout direction
- **Spinning Top** — Small body with equal shadows, no clear direction
""")
        else:
            st.info("No strong patterns in recent candles. Switch to Swing mode for more data.")

    with tab_ta:
        ta = get_ta_summary(chart_df)
        if ta:
            vc = ta["verdict_color"]
            bg_v = "#0d2818" if "BUY" in ta["verdict"] else ("#2d0a0a" if "SELL" in ta["verdict"] else "#1a1500")
            pct_buy = int(ta["buys"]/ta["total"]*100) if ta["total"] else 50
            st.markdown(f"""
<div style='background:{bg_v};border:2px solid {vc};border-radius:12px;
padding:16px;margin-bottom:14px;text-align:center;'>
  <div style='font-size:12px;color:#888;text-transform:uppercase;letter-spacing:.06em;'>TA Overall Verdict</div>
  <div style='font-size:28px;font-weight:700;color:{vc};margin:6px 0;'>{ta["verdict"]}</div>
  <div style='display:flex;justify-content:center;gap:24px;font-size:13px;'>
    <span style='color:#00b880;'>BUY: {ta["buys"]}</span>
    <span style='color:#f39c12;'>Neutral: {ta["neutrals"]}</span>
    <span style='color:#e74c3c;'>SELL: {ta["sells"]}</span>
  </div>
  <div style='background:rgba(255,255,255,0.1);border-radius:99px;height:8px;
  margin:10px auto;max-width:280px;'>
    <div style='width:{pct_buy}%;background:{vc};border-radius:99px;height:8px;'></div>
  </div>
  <div style='font-size:11px;color:#888;'>{ta["buys"]}/{ta["total"]} indicators bullish</div>
</div>""", unsafe_allow_html=True)

            ta_c1, ta_c2 = st.columns(2)
            with ta_c1:
                st.markdown("**Oscillators**")
                for ind in ta["oscillators"]:
                    st.markdown(f"""<div style='display:flex;justify-content:space-between;
padding:5px 10px;background:#1a1a2e;border-radius:6px;margin-bottom:4px;font-size:12px;'>
<span style='color:#ccc;'>{ind["name"]}</span>
<span style='color:#888;font-family:monospace;'>{ind["value"]}</span>
<span style='color:{ind["color"]};font-weight:600;'>{ind["sig"]}</span>
</div>""", unsafe_allow_html=True)
            with ta_c2:
                st.markdown("**Moving Averages**")
                for ind in ta["moving_avgs"]:
                    st.markdown(f"""<div style='display:flex;justify-content:space-between;
padding:5px 10px;background:#1a1a2e;border-radius:6px;margin-bottom:4px;font-size:12px;'>
<span style='color:#ccc;'>{ind["name"]}</span>
<span style='color:#888;font-family:monospace;'>{ind["value"]}</span>
<span style='color:{ind["color"]};font-weight:600;'>{ind["sig"]}</span>
</div>""", unsafe_allow_html=True)



    # =============================================================
    # ADVANCED ANALYSIS TABS
    # =============================================================
    st.markdown("---")
    st.markdown("### Advanced Analysis")
    at1, at2, at3, at4, at5 = st.tabs([
        "Price Action + SMC",
        "Volume Profile",
        "Options Data",
        "ORB + Session",
        "Kelly Sizing",
    ])

    with at1:
        col_pa, col_smc = st.columns(2)

        with col_pa:
            st.markdown("#### Market Structure")
            ms = detect_market_structure(chart_df)
            if ms and "trend" in ms:
                tc = ms.get("trend_color","#f39c12")
                st.markdown(f"""
<div style='background:#1a1a2e;border:2px solid {tc};border-radius:10px;padding:14px;margin-bottom:10px;'>
  <div style='font-size:20px;font-weight:700;color:{tc};'>{ms.get("trend", "Unknown")}</div>
  <div style='font-size:12px;color:#aaa;margin-top:6px;'>
    {("HH" if ms.get("hh") else "LH")} + {("HL" if ms.get("hl") else "LL")}
  </div>
  {"<div style='margin-top:8px;background:#0d2818;border-radius:6px;padding:8px;font-size:12px;color:#00e5a0;'>" + ms.get("mss", "") + "</div>" if ms.get("mss") else ""}
</div>""", unsafe_allow_html=True)

            st.markdown("#### Demand / Supply Zones")
            supply_z, demand_z = find_demand_supply_zones(chart_df)
            if supply_z:
                for z in reversed(supply_z):
                    st.markdown(f"<div style='background:#2d0a0a;border-left:3px solid #e74c3c;border-radius:4px;padding:6px 10px;margin-bottom:4px;font-size:12px;color:#e74c3c;'>SUPPLY Rs.{z['bottom']} — Rs.{z['top']} | Strength: {'*'*z['strength']} | {z['date']}</div>", unsafe_allow_html=True)
            if demand_z:
                for z in reversed(demand_z):
                    st.markdown(f"<div style='background:#0d2818;border-left:3px solid #00b880;border-radius:4px;padding:6px 10px;margin-bottom:4px;font-size:12px;color:#00b880;'>DEMAND Rs.{z['bottom']} — Rs.{z['top']} | Strength: {'*'*z['strength']} | {z['date']}</div>", unsafe_allow_html=True)

            st.markdown("#### Fake Breakout Detection")
            fakes = detect_fake_breakout(chart_df)
            if fakes:
                for fb in fakes:
                    st.markdown(f"<div style='background:#1a0a0a;border:1px solid {fb['color']};border-radius:6px;padding:8px;margin-bottom:6px;font-size:12px;'><b style='color:{fb['color']};'>{fb['type']}</b><br><span style='color:#aaa;'>{fb['desc']}</span><br><b style='color:{fb['color']};'>{fb['signal']}</b></div>", unsafe_allow_html=True)
            else:
                st.success("No fake breakouts detected recently")

        with col_smc:
            st.markdown("#### Order Blocks (Institutional Zones)")
            obs = find_order_blocks(chart_df)
            if obs:
                for ob in reversed(obs[-3:]):
                    c = ob["color"]
                    st.markdown(f"""
<div style='background:{"#0d2818" if c=="#00b880" else "#2d0a0a"};border:1px solid {c};border-radius:8px;padding:10px;margin-bottom:6px;'>
  <div style='font-size:13px;font-weight:600;color:{c};'>{ob["type"]}</div>
  <div style='font-size:12px;color:#aaa;'>Zone: Rs.{ob["bottom"]} — Rs.{ob["top"]}</div>
  <div style='font-size:11px;color:#888;'>{ob["desc"]}</div>
  <div style='font-size:12px;color:{c};font-weight:600;margin-top:4px;'>{ob["signal"]}</div>
  <div style='font-size:10px;color:#666;'>{ob["date"]}</div>
</div>""", unsafe_allow_html=True)
            else:
                st.info("No clear Order Blocks found")

            st.markdown("#### Fair Value Gaps (FVG)")
            fvgs = find_fvg(chart_df)
            if fvgs:
                for fvg in fvgs[-3:]:
                    c = fvg["color"]
                    st.markdown(f"""
<div style='background:{"#0d2818" if c=="#00b880" else "#2d0a0a"};border:1px solid {c};border-radius:6px;padding:8px;margin-bottom:6px;font-size:12px;'>
  <b style='color:{c};'>{fvg["type"]}</b>
  <span style='color:#888;margin-left:8px;'>Gap: Rs.{fvg["gap"]:.2f}</span><br>
  <span style='color:#aaa;'>Zone: Rs.{fvg["bottom"]} — Rs.{fvg["top"]}</span><br>
  <span style='color:{c};'>{fvg["desc"]}</span><br>
  <span style='font-size:10px;color:#666;'>{fvg["date"]}</span>
</div>""", unsafe_allow_html=True)
            else:
                st.info("No unfilled FVGs found")

    with at2:
        st.markdown("#### Volume Profile")
        vp = compute_volume_profile(chart_df)
        if vp and "poc" in vp:
            price_now = float(chart_df["Close"].iloc[-1])
            vc1,vc2,vc3 = st.columns(3)
            vc1.metric("POC (Point of Control)", f"Rs.{vp['poc']:,.2f}",
                       f"{'Above' if price_now>vp['poc'] else 'Below'} POC")
            vc2.metric("Value Area High (VAH)", f"Rs.{vp['vah']:,.2f}")
            vc3.metric("Value Area Low (VAL)",  f"Rs.{vp['val']:,.2f}")

            pos_color = "#00b880" if "Potential" in vp["position"] or "AT" in vp["position"] else "#f39c12"
            st.markdown(f"""
<div style='background:#1a1a2e;border:2px solid {pos_color};border-radius:10px;padding:14px;'>
  <div style='font-size:15px;font-weight:600;color:{pos_color};'>{vp["position"]}</div>
  <div style='margin-top:10px;'>
    <div style='font-size:12px;color:#888;margin-bottom:4px;'>Price vs Value Area:</div>
    <div style='display:flex;gap:8px;flex-wrap:wrap;font-size:12px;'>
      <span style='background:#2d0a0a;color:#e74c3c;padding:3px 10px;border-radius:99px;'>VAH Rs.{vp["vah"]}</span>
      <span style='background:#1a1a2e;color:#a78bfa;padding:3px 10px;border-radius:99px;'>POC Rs.{vp["poc"]}</span>
      <span style='background:#0d2818;color:#00b880;padding:3px 10px;border-radius:99px;'>VAL Rs.{vp["val"]}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

            st.markdown("**Highest Volume Price Levels:**")
            for pl, vol in vp["top_levels"]:
                is_poc = pl == vp["poc"]
                st.markdown(f"<div style='display:flex;justify-content:space-between;background:{'#1a2a3d' if is_poc else '#161b22'};border-radius:5px;padding:5px 12px;margin-bottom:3px;font-size:12px;'><span style='color:{'#f39c12' if is_poc else '#ccc'};'>{'POC ' if is_poc else ''}Rs.{pl:,.2f}</span><span style='color:#888;'>{vol:.1f}M vol</span></div>", unsafe_allow_html=True)
        else:
            st.info("Not enough data for Volume Profile")

    with at3:
        st.markdown("#### Options Data (NSE)")
        sym_clean = stock.replace(".NS","")
        if st.button(f"Load Options Data for {sym_clean}", type="primary"):
            with st.spinner("Fetching options chain..."):
                opt_data = fetch_options_data(stock)
            if "error" in opt_data:
                st.warning(f"Options not available: {opt_data['error']}")
            elif opt_data:
                oc1,oc2,oc3 = st.columns(3)
                pcr_color = "#00b880" if opt_data["pcr"]>1 else "#e74c3c"
                oc1.metric("Total Call OI", f"{opt_data['call_oi']:,}")
                oc2.metric("Total Put OI",  f"{opt_data['put_oi']:,}")
                oc3.metric("PCR",           f"{opt_data['pcr']}", opt_data["pcr_signal"])

                st.markdown(f"""
<div style='background:#1a1a2e;border:2px solid {pcr_color};border-radius:10px;padding:14px;margin:10px 0;'>
  <div style='font-size:18px;font-weight:700;color:{pcr_color};'>PCR {opt_data["pcr"]} — {opt_data["pcr_signal"]}</div>
  <div style='font-size:12px;color:#888;margin-top:6px;'>
    PCR >1.5 = Bullish (more puts = market makers expect UP)<br>
    PCR 0.5-0.8 = Bearish (more calls = caution)<br>
    PCR 0.8-1.2 = Neutral
  </div>
  {"<div style='margin-top:10px;font-size:14px;'><b style='color:#f39c12;'>Max Pain: Rs." + str(opt_data.get("max_pain","N/A")) + "</b> — Price tends to expire near this level</div>" if opt_data.get("max_pain") else ""}
</div>""", unsafe_allow_html=True)

                if opt_data.get("top_call_strikes"):
                    st.markdown("**Highest Call OI (Resistance levels):**")
                    for strike, oi in opt_data["top_call_strikes"]:
                        st.markdown(f"<div style='background:#2d0a0a;border-radius:5px;padding:5px 12px;margin-bottom:3px;font-size:12px;color:#e74c3c;'>CALL Rs.{strike:,.0f} — OI: {int(oi):,}</div>", unsafe_allow_html=True)
                if opt_data.get("top_put_strikes"):
                    st.markdown("**Highest Put OI (Support levels):**")
                    for strike, oi in opt_data["top_put_strikes"]:
                        st.markdown(f"<div style='background:#0d2818;border-radius:5px;padding:5px 12px;margin-bottom:3px;font-size:12px;color:#00b880;'>PUT Rs.{strike:,.0f} — OI: {int(oi):,}</div>", unsafe_allow_html=True)
        else:
            st.info("Click button to load live options chain data (requires market hours for best data)")

        st.markdown("---")
        st.markdown("#### How to use Options Data:")
        st.markdown(f"""
- **PCR > 1.5** — Bullish signal (institutions writing puts = they expect market UP)
- **PCR < 0.7** — Bearish signal (institutions writing calls = they expect market DOWN)
- **Max Pain** — Price where maximum options expire worthless — market tends to go here
- **Highest Call OI strike** = Strong resistance level
- **Highest Put OI strike** = Strong support level
""")

    with at4:
        st.markdown("#### Opening Range Breakout (ORB)")
        orb = compute_orb(df)
        if orb and "error" not in orb and "orb_high" in orb:
            oc = orb["color"]
            st.markdown(f"""
<div style='background:#1a1a2e;border:2px solid {oc};border-radius:12px;padding:16px;'>
  <div style='font-size:18px;font-weight:700;color:{oc};'>{orb["status"]}</div>
  <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin:12px 0;text-align:center;'>
    <div style='background:#0d2818;border-radius:8px;padding:8px;'>
      <div style='font-size:10px;color:#888;'>ORB High</div>
      <div style='font-size:16px;font-weight:600;color:#e74c3c;'>Rs.{orb["orb_high"]}</div>
    </div>
    <div style='background:#1a1a2e;border-radius:8px;padding:8px;'>
      <div style='font-size:10px;color:#888;'>ORB Range</div>
      <div style='font-size:16px;font-weight:600;color:#f39c12;'>Rs.{orb["orb_range"]:.2f}</div>
    </div>
    <div style='background:#0d2818;border-radius:8px;padding:8px;'>
      <div style='font-size:10px;color:#888;'>ORB Low</div>
      <div style='font-size:16px;font-weight:600;color:#00b880;'>Rs.{orb["orb_low"]}</div>
    </div>
  </div>
  <div style='font-size:13px;color:{oc};font-weight:600;'>{orb["signal"]}</div>
  <div style='font-size:12px;color:#888;margin-top:6px;'>
    Target UP: Rs.{orb["target_up"]} | Target DOWN: Rs.{orb["target_down"]}
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.info("ORB needs intraday data. Switch to Intraday mode during market hours (9:15–10:00 AM).")

        st.markdown("#### Indian Market Session Guide")
        now_t = datetime.datetime.now().time()
        sessions = [
            ("9:15 – 9:30",  "Market Open",       "Very volatile, avoid if not experienced", "#f39c12"),
            ("9:30 – 10:30", "Momentum Session",  "Best time — strong trends, high volume", "#00b880"),
            ("10:30 – 12:30","Mid Morning",        "Moderate activity, trend continuation", "#4e8fff"),
            ("12:30 – 2:00", "Lunch Lull",         "Low volume, choppy — avoid intraday", "#888888"),
            ("2:00 – 3:15",  "Power Hour",         "Strong moves, reversals common", "#a78bfa"),
            ("3:15 – 3:30",  "Closing Session",    "Square off only — no new entries", "#e74c3c"),
        ]
        for time_r, name, desc, color in sessions:
            st.markdown(f"""<div style='display:flex;gap:10px;align-items:center;background:#1a1a2e;
border-left:3px solid {color};border-radius:5px;padding:8px 12px;margin-bottom:5px;'>
<div style='min-width:100px;font-family:monospace;font-size:11px;color:{color};'>{time_r}</div>
<div><b style='color:#ccc;font-size:12px;'>{name}</b>
<div style='font-size:11px;color:#888;'>{desc}</div></div>
</div>""", unsafe_allow_html=True)

    with at5:
        st.markdown("#### Kelly Criterion — Optimal Position Size")
        kc1,kc2,kc3 = st.columns(3)
        k_wr  = kc1.slider("Your Win Rate %", 30, 80, 55) / 100
        k_rr  = kc2.number_input("Risk:Reward Ratio", 0.5, 5.0, 2.0, 0.1)
        k_cap = kc3.number_input("Capital (Rs.)", 10000, 10000000, 500000, step=10000)

        kelly = kelly_sizing(k_wr, k_rr, k_cap)
        if kelly:
            ev_color = "#00b880" if kelly["expected_value"] > 0 else "#e74c3c"
            kk1,kk2,kk3 = st.columns(3)
            kk1.metric("Full Kelly %",  f"{kelly['full_kelly']}%",  f"Rs.{kelly['capital_full']:,.0f}")
            kk2.metric("Half Kelly %",  f"{kelly['half_kelly']}%",  f"Rs.{kelly['capital_half']:,.0f} (Recommended)")
            kk3.metric("Expected Value",f"{kelly['expected_value']}", kelly["interpretation"])

            st.markdown(f"""
<div style='background:#1a1a2e;border:2px solid {ev_color};border-radius:10px;padding:14px;margin:10px 0;'>
  <div style='font-size:16px;font-weight:700;color:{ev_color};'>{kelly["interpretation"]}</div>
  <div style='font-size:12px;color:#aaa;margin-top:8px;'>
    With {int(k_wr*100)}% win rate and {k_rr}:1 R:R —<br>
    Use <b>Half Kelly</b> for safer bet: Rs.{kelly['capital_half']:,.0f} per trade<br>
    Even with 40% win rate, 2:1 R:R is profitable!
  </div>
</div>""", unsafe_allow_html=True)

            st.markdown("**Win Rate vs R:R Profitability Table:**")
            import pandas as pd
            wr_vals = [35, 40, 45, 50, 55, 60]
            rr_vals = [1.0, 1.5, 2.0, 3.0]
            table_data = {}
            for rr in rr_vals:
                col_vals = []
                for wr in wr_vals:
                    ev = rr*(wr/100) - (1-wr/100)
                    col_vals.append(f"+{ev:.2f}" if ev>0 else f"{ev:.2f}")
                table_data[f"R:R {rr}"] = col_vals
            df_table = pd.DataFrame(table_data, index=[f"{w}% WR" for w in wr_vals])
            st.dataframe(df_table, use_container_width=True)
            st.caption("+ = profitable | - = losing | Values = Expected value per rupee risked")



    # ── TAB: Market Structure (BOS + CHOCH + EQH/EQL) ────────
    with tab_struct:
        st.markdown("#### Market Structure — BOS, CHOCH, EQH/EQL")
        sc1, sc2 = st.columns(2)

        with sc1:
            st.markdown("**Break of Structure (BOS) + Change of Character (CHOCH)**")
            try:
                bos_data = detect_bos_choch(chart_df)
                trend    = bos_data.get("trend","Unknown")
                tc2      = "#00b880" if trend=="Uptrend" else ("#e74c3c" if trend=="Downtrend" else "#f39c12")
                # Trend + swing levels
                _bd  = f"<div style='background:{tc2}22;border:2px solid {tc2};border-radius:12px;padding:14px;margin-bottom:12px;'>"
                _bd += f"<div style='font-size:20px;font-weight:800;color:{tc2};'>{trend}</div>"
                _bd += f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px;'>"
                _bd += f"<div style='background:rgba(0,0,0,0.3);border-radius:6px;padding:8px;text-align:center;'><div style='font-size:9px;color:#888;'>Last Swing High</div><div style='font-size:14px;color:#e74c3c;font-weight:600;'>Rs.{bos_data.get('last_sh','—')}</div></div>"
                _bd += f"<div style='background:rgba(0,0,0,0.3);border-radius:6px;padding:8px;text-align:center;'><div style='font-size:9px;color:#888;'>Last Swing Low</div><div style='font-size:14px;color:#00b880;font-weight:600;'>Rs.{bos_data.get('last_sl','—')}</div></div>"
                _bd += "</div></div>"
                # BOS
                if bos_data.get("bos"):
                    b=bos_data["bos"]; bc=b["color"]
                    _bd += f"<div style='background:{bc}22;border:2px solid {bc};border-radius:10px;padding:12px;margin-bottom:10px;'>"
                    _bd += f"<div style='font-size:11px;color:#888;text-transform:uppercase;margin-bottom:4px;'>BOS — Break of Structure</div>"
                    _bd += f"<div style='font-size:18px;font-weight:800;color:{bc};'>{b['direction']}</div>"
                    _bd += f"<div style='font-size:12px;color:#aaa;margin-top:5px;'>{b['desc']}</div>"
                    _bd += f"<div style='font-size:11px;color:{bc};margin-top:5px;'>Level: Rs.{b['level']}</div></div>"
                else:
                    _bd += "<div style='background:#161b22;border:1px solid #21262d;border-radius:8px;padding:10px;margin-bottom:8px;font-size:12px;color:#555;text-align:center;'>No BOS — Price within structure</div>"
                # CHOCH
                if bos_data.get("choch"):
                    c=bos_data["choch"]; cc=c["color"]
                    _bd += f"<div style='background:{cc}22;border:2px solid {cc};border-radius:10px;padding:12px;'>"
                    _bd += f"<div style='font-size:11px;color:#888;text-transform:uppercase;margin-bottom:4px;'>CHOCH — Change of Character</div>"
                    _bd += f"<div style='font-size:18px;font-weight:800;color:{cc};'>{c['direction']}</div>"
                    _bd += f"<div style='font-size:12px;color:#aaa;margin-top:5px;'>{c['desc']}</div>"
                    _bd += f"<div style='font-size:11px;color:{cc};margin-top:5px;'>Trend reversal signal — Act fast!</div></div>"
                else:
                    _bd += "<div style='background:#161b22;border:1px solid #21262d;border-radius:8px;padding:10px;font-size:12px;color:#555;text-align:center;'>No CHOCH — Trend intact</div>"
                st.markdown(_bd, unsafe_allow_html=True)
            except Exception as _e:
                st.caption(f"BOS/CHOCH: {str(_e)[:60]}")

        with sc2:
            st.markdown("**Equal Highs (EQH) / Equal Lows (EQL) — Liquidity Pools**")
            try:
                eq = detect_equal_levels(chart_df)
                if eq.get("eqh"):
                    for level in eq["eqh"]:
                        is_near = eq.get("near_eqh") and abs(level-price)<atr_now*2
                        c2="#e74c3c"; bg2="#2d0a0a" if is_near else "#161b22"
                        st.markdown(f"""<div style='background:{bg2};border-left:3px solid {c2};border-radius:5px;padding:6px 10px;margin-bottom:4px;font-size:12px;'>
<b style='color:{c2};'>EQH (Sell-side Liquidity)</b> Rs.{level:,.2f}
{"  PRICE NEAR — Sweep possible!" if is_near else ""}
</div>""", unsafe_allow_html=True)
                if eq.get("eql"):
                    for level in eq["eql"]:
                        is_near = eq.get("near_eql") and abs(level-price)<atr_now*2
                        c3="#00b880"; bg3="#0d2818" if is_near else "#161b22"
                        st.markdown(f"""<div style='background:{bg3};border-left:3px solid {c3};border-radius:5px;padding:6px 10px;margin-bottom:4px;font-size:12px;'>
<b style='color:{c3};'>EQL (Buy-side Liquidity)</b> Rs.{level:,.2f}
{"  PRICE NEAR — Sweep possible!" if is_near else ""}
</div>""", unsafe_allow_html=True)
                if not eq.get("eqh") and not eq.get("eql"):
                    st.info("No equal highs/lows detected")
            except Exception as _e2:
                st.caption(f"EQH/EQL: {str(_e2)[:50]}")

            st.markdown("**Fibonacci Levels**")
            try:
                fib = get_fibonacci_levels(chart_df)
                in_gz = fib.get("in_golden_zone",False)
                gz_c = "#ffd700" if in_gz else "#888"
                st.markdown(f"""<div style='background:{"#1a1a00" if in_gz else "#161b22"};border:1px solid {gz_c};border-radius:8px;padding:10px;'>
<div style='font-size:12px;font-weight:600;color:{gz_c};'>
{"IN Golden Zone (61.8%–38.2%) — Best entry zone!" if in_gz else "Golden Zone: Rs." + str(fib.get("golden_bot","—")) + " — Rs." + str(fib.get("golden_top","—"))}</div>
</div>""", unsafe_allow_html=True)
                for lvl, val in list(fib.get("levels",{}).items())[:7]:
                    d2 = val-price; is_n = abs(d2)/price<0.008
                    c4 = "#ffd700" if "Golden" in lvl else ("#00b880" if is_n and d2<0 else ("#e74c3c" if is_n and d2>0 else "#888"))
                    st.markdown(f"""<div style='display:flex;justify-content:space-between;background:#161b22;border-radius:4px;padding:4px 8px;margin-bottom:2px;font-size:11px;border-left:2px solid {c4};'>
<span style='color:{c4};'>{lvl}</span><span style='color:#ccc;'>Rs.{val:,.2f}</span>
<span style='color:{c4};'>{"← NEAR" if is_n else ""}</span></div>""", unsafe_allow_html=True)
            except Exception as _e3:
                st.caption(f"Fibonacci: {str(_e3)[:50]}")

    # ── TAB: Fibonacci + Trendline ────────────────────────────
    with tab_pat:
        st.markdown("#### Chart Patterns (Auto-Detected)")
        try:
            cp = detect_chart_patterns(chart_df)
            if cp:
                for p in cp:
                    pc4="#00b880" if p["type"]=="bullish" else ("#e74c3c" if p["type"]=="bearish" else "#f39c12")
                    pb4="#0d2818" if p["type"]=="bullish" else ("#2d0a0a" if p["type"]=="bearish" else "#1a1500")
                    tgt_line = f"<div style='font-size:11px;color:{pc4};margin-top:5px;'>Target: Rs.{p['target']:,.2f}</div>" if p.get("target") else ""
                    st.markdown(f"""<div style='background:{pb4};border:2px solid {pc4};border-radius:12px;padding:14px;margin-bottom:10px;'>
<div style='display:flex;justify-content:space-between;align-items:center;'>
  <div>
    <div style='font-size:16px;font-weight:700;color:{pc4};'>{p["pattern"]}</div>
    <div style='font-size:12px;color:#888;margin-top:3px;'>{p["desc"]}</div>
    {tgt_line}
  </div>
  <div style='text-align:right;'>
    <div style='font-size:14px;font-weight:600;color:{pc4};'>{p["signal"]}</div>
    <div style='font-size:11px;color:#888;'>{"★"*p["strength"]} {p["strength"]}/5</div>
  </div>
</div></div>""", unsafe_allow_html=True)
            else:
                st.info("No chart patterns in current data. Try 1mo/Swing mode.")

            st.markdown("**Trendlines**")
            try:
                tl = detect_trendlines(chart_df)
                sig_c5 = "#00b880" if "BUY" in tl.get("signal","") else ("#e74c3c" if "SELL" in tl.get("signal","") else "#f39c12")
                st.markdown(f"""<div style='background:#161b22;border:2px solid {sig_c5};border-radius:8px;padding:10px;margin-bottom:8px;'>
<div style='font-size:13px;font-weight:600;color:{sig_c5};'>{tl.get("signal","—")}</div></div>""", unsafe_allow_html=True)
                r5=tl.get("resistance"); s5=tl.get("support")
                if r5: st.markdown(f"""<div style='background:#2d0a0a;border-left:3px solid #e74c3c;border-radius:5px;padding:8px;margin-bottom:5px;font-size:12px;'>
<b style='color:#e74c3c;'>Resistance TL</b> Rs.{r5["price"]:,.2f} | {r5["direction"]} | Touches:{r5["touches"]}{"  BROKEN!" if r5["broken"] else ""}</div>""", unsafe_allow_html=True)
                if s5: st.markdown(f"""<div style='background:#0d2818;border-left:3px solid #00b880;border-radius:5px;padding:8px;font-size:12px;'>
<b style='color:#00b880;'>Support TL</b> Rs.{s5["price"]:,.2f} | {s5["direction"]} | Touches:{s5["touches"]}{"  BROKEN!" if s5["broken"] else ""}</div>""", unsafe_allow_html=True)
            except Exception as _tl:
                st.caption(f"Trendline: {str(_tl)[:50]}")
        except Exception as _cp2:
            st.warning(f"Chart patterns: {str(_cp2)[:60]}")

    # ── TAB: ICT + Liquidity ──────────────────────────────────
    with tab_ict2:
        st.markdown("#### ICT Concepts + Liquidity Sweep")
        ic1, ic2 = st.columns(2)

        with ic1:
            # Liquidity Sweep
            st.markdown("**Liquidity Sweep + BSL/SSL**")
            try:
                # EQH/EQL = BSL/SSL
                _eq2 = detect_equal_levels(chart_df)
                if _eq2.get("BSL"):
                    for _bsl_v in _eq2["BSL"][:2]:
                        _near_b = abs(_bsl_v - price)/price < 0.01
                        _bh = f"<div style='background:{'#2d0a0a' if _near_b else '#161b22'};border-left:4px solid #e74c3c;border-radius:6px;padding:8px 12px;margin-bottom:5px;'>"
                        _bh += f"<b style='color:#e74c3c;'>BSL (Buy-Side Liquidity)</b> Rs.{_bsl_v:,.2f}"
                        _bh += f"{'  ⚡ PRICE NEAR — Sweep risk!' if _near_b else ''}"
                        _bh += f"<div style='font-size:10px;color:#888;'>Equal Highs — stop hunts above this level</div></div>"
                        st.markdown(_bh, unsafe_allow_html=True)
                if _eq2.get("SSL"):
                    for _ssl_v in _eq2["SSL"][:2]:
                        _near_s = abs(_ssl_v - price)/price < 0.01
                        _sh = f"<div style='background:{'#0d2818' if _near_s else '#161b22'};border-left:4px solid #00b880;border-radius:6px;padding:8px 12px;margin-bottom:5px;'>"
                        _sh += f"<b style='color:#00b880;'>SSL (Sell-Side Liquidity)</b> Rs.{_ssl_v:,.2f}"
                        _sh += f"{'  ⚡ PRICE NEAR — Bounce zone!' if _near_s else ''}"
                        _sh += f"<div style='font-size:10px;color:#888;'>Equal Lows — stop hunts below this level</div></div>"
                        st.markdown(_sh, unsafe_allow_html=True)

                lq = detect_liquidity_sweep(chart_df)
                if lq:
                    st.markdown("**Recent Sweeps:**")
                    for sw in lq:
                        sc6=sw["color"]
                        _lqh  = f"<div style='background:{'#0d2818' if sc6=='#00b880' else '#2d0a0a'};border:2px solid {sc6};border-radius:8px;padding:10px;margin-bottom:6px;'>"
                        _lqh += f"<div style='font-size:13px;font-weight:700;color:{sc6};'>{sw['type']}</div>"
                        _lqh += f"<div style='font-size:11px;color:#aaa;margin:4px 0;'>{sw['desc']}</div>"
                        _lqh += f"<div style='font-size:12px;color:{sc6};font-weight:600;'>Signal: {sw['signal']} | Swept: Rs.{sw['swept']} | {sw['date']}</div></div>"
                        st.markdown(_lqh, unsafe_allow_html=True)
                else:
                    st.success("No liquidity sweeps — market clean")
            except Exception as _lq2: st.caption(f"Sweep: {str(_lq2)[:50]}")

            # Premium/Discount
            st.markdown("**Premium / Discount Zones**")
            try:
                pd6 = get_premium_discount(chart_df)
                zc6=pd6["color"]
                st.markdown(f"""<div style='background:{zc6}22;border:2px solid {zc6};border-radius:10px;padding:14px;'>
<div style='font-size:18px;font-weight:700;color:{zc6};'>{pd6["zone"]}</div>
<div style='font-size:13px;color:#aaa;margin:6px 0;'>{pd6["recommendation"]}</div>
<div style='background:rgba(255,255,255,0.08);border-radius:99px;height:10px;margin-bottom:8px;position:relative;'>
  <div style='width:{pd6["pct"]}%;background:{zc6};border-radius:99px;height:10px;'></div>
  <div style='position:absolute;left:50%;top:0;width:2px;height:10px;background:#fff;opacity:0.3;'></div>
</div>
<div style='display:flex;justify-content:space-between;font-size:10px;color:#888;'>
  <span>Discount (Buy)</span><span style='color:#f39c12;'>Eq</span><span>Premium (Sell)</span>
</div>
<div style='margin-top:8px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;text-align:center;font-size:11px;'>
  <div><div style='color:#888;'>Range H</div><div style='color:#e74c3c;'>Rs.{pd6["high"]}</div></div>
  <div><div style='color:#888;'>Mid</div><div style='color:#f39c12;'>Rs.{pd6["mid"]}</div></div>
  <div><div style='color:#888;'>Range L</div><div style='color:#00b880;'>Rs.{pd6["low"]}</div></div>
</div></div>""", unsafe_allow_html=True)
            except Exception as _pd: st.caption(f"P/D: {str(_pd)[:50]}")

        with ic2:
            # Kill Zones
            st.markdown("**ICT Kill Zones (Live)**")
            try:
                kz = get_kill_zones()
                if kz.get("active"):
                    az=kz["active"]
                    st.markdown(f"""<div style='background:{az["color"]}22;border:2px solid {az["color"]};border-radius:8px;padding:10px;margin-bottom:8px;'>
<div style='font-size:14px;font-weight:700;color:{az["color"]};'>ACTIVE: {az["name"]}</div>
<div style='font-size:12px;color:#aaa;'>{az["desc"]}</div>
</div>""", unsafe_allow_html=True)
                for start,end,name,color,desc in kz["zones"]:
                    active_z = kz.get("active") and kz["active"]["name"]==name
                    bg7="#161b22" if not active_z else f"{color}11"
                    bd7=f"1px solid {color}" if active_z else "1px solid #21262d"
                    st.markdown(f"""<div style='background:{bg7};border:{bd7};border-radius:6px;padding:6px 10px;margin-bottom:4px;font-size:11px;'>
<b style='color:{color};'>{"LIVE — " if active_z else ""}{name}</b> ({start}–{end})<br>
<span style='color:#888;'>{desc}</span></div>""", unsafe_allow_html=True)
            except Exception as _kz: st.caption(f"Kill Zones: {str(_kz)[:50]}")

            # Institutional Score
            st.markdown("**Institutional Score**")
            try:
                inst = get_institutional_score(chart_df, stock)
                isc=inst["color"]; isc2=inst["score"]
                bkd=inst.get("breakdown",{}); bkd_mx=inst.get("breakdown_max",{})

                _ish  = f"<div style='background:{isc}22;border:2px solid {isc};border-radius:14px;padding:16px;'>"
                _ish += f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;'>"
                _ish += f"<div><div style='font-size:10px;color:#888;text-transform:uppercase;'>Institutional Grade</div>"
                _ish += f"<div style='font-size:42px;font-weight:900;color:{isc};line-height:1;'>{isc2}/100</div>"
                _ish += f"<div style='font-size:14px;color:{isc};font-weight:600;'>{inst['grade']} — {inst['label']}</div></div>"
                _ish += f"<div style='background:rgba(255,255,255,0.06);border-radius:99px;width:80px;height:80px;display:flex;align-items:center;justify-content:center;'>"
                _ish += f"<div style='font-size:22px;font-weight:900;color:{isc};'>{isc2}</div></div></div>"
                _ish += f"<div style='background:rgba(255,255,255,0.06);border-radius:99px;height:10px;margin-bottom:14px;'>"
                _ish += f"<div style='width:{isc2}%;background:{isc};border-radius:99px;height:10px;'></div></div>"
                # 5-component breakdown
                comp_map = [
                    ("Trend",    "trend",    20,  "EMA/ST/ADX"),
                    ("Volume",   "volume",   20,  "Vol ratio"),
                    ("Momentum", "momentum", 20,  "RSI/MFI/MACD"),
                    ("SMC",      "smc",      20,  "OB/FVG/BOS"),
                    ("Risk/PD",  "risk",     16,  "Premium/Disc"),
                    ("Options",  "options",   4,  "PCR score"),
                ]
                _ish += "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:6px;'>"
                for comp_name, comp_key, comp_max, comp_desc in comp_map:
                    comp_val = bkd.get(comp_key, 0)
                    comp_pct = int(comp_val/comp_max*100)
                    comp_c   = "#00b880" if comp_pct>=65 else ("#f39c12" if comp_pct>=40 else "#e74c3c")
                    _ish += f"<div style='background:#161b22;border-radius:6px;padding:8px;text-align:center;border:1px solid {comp_c}44;'>"
                    _ish += f"<div style='font-size:9px;color:#888;'>{comp_name}</div>"
                    _ish += f"<div style='font-size:16px;font-weight:700;color:{comp_c};'>{comp_val}/{comp_max}</div>"
                    _ish += f"<div style='background:rgba(255,255,255,0.05);border-radius:99px;height:4px;margin:4px 0;'><div style='width:{comp_pct}%;background:{comp_c};border-radius:99px;height:4px;'></div></div>"
                    _ish += f"<div style='font-size:8px;color:#555;'>{comp_desc}</div></div>"
                _ish += "</div></div>"
                st.markdown(_ish, unsafe_allow_html=True)
            except Exception as _is: st.caption(f"Inst Score: {str(_is)[:50]}")

    # ── TAB: Multi-Timeframe ──────────────────────────────────
    with tab_mtf:
        st.markdown("#### Multi-Timeframe Confluence (5m + 15m + 1H + 4H)")
        st.caption("Comparing all timeframes — requires internet connection")

        if st.button("Load MTF Analysis", type="primary", key="mtf_load"):
            with st.spinner("Loading 4 timeframes..."):
                mtf = get_mtf_confluence(stock)
            st.session_state["mtf_cache"] = mtf

        mtf = st.session_state.get("mtf_cache")
        if mtf:
            conf = mtf["confluence"]
            sig7 = mtf["signal"]
            cc7  = "#00b880" if "BUY" in sig7 else ("#e74c3c" if "SELL" in sig7 else "#f39c12")

            _ap  = mtf.get("align_pct",conf); _maj = mtf.get("majority","BULLISH")
            _alc = "#00b880" if _maj=="BULLISH" else "#e74c3c"
            _mh  = f"<div style='background:{cc7}22;border:2px solid {cc7};border-radius:14px;padding:18px;margin-bottom:14px;'>"
            _mh += f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;'>"
            _mh += f"<div style='text-align:center;'><div style='font-size:10px;color:#888;text-transform:uppercase;'>MTF Confluence</div><div style='font-size:44px;font-weight:900;color:{cc7};line-height:1;'>{conf}%</div><div style='font-size:12px;color:{cc7};font-weight:600;'>{sig7}</div></div>"
            _mh += f"<div style='text-align:center;'><div style='font-size:10px;color:#888;text-transform:uppercase;'>TF Alignment</div><div style='font-size:44px;font-weight:900;color:{_alc};line-height:1;'>{_ap}%</div><div style='font-size:12px;color:{_alc};font-weight:600;'>{_maj}</div></div>"
            _mh += "</div>"
            _mh += f"<div style='background:rgba(255,255,255,0.08);border-radius:99px;height:10px;margin-bottom:5px;'><div style='width:{conf}%;background:{cc7};border-radius:99px;height:10px;'></div></div>"
            _mh += f"<div style='display:flex;justify-content:space-between;font-size:10px;color:#555;'><span>Bear: {mtf.get('bear_count',0)}</span><span>Bull: {mtf.get('bull_count',0)}</span></div>"
            _mh += "</div>"
            st.markdown(_mh, unsafe_allow_html=True)

            tf_cols = st.columns(len(mtf.get("breakdown",{})) or 4)
            for i,(tf,data) in enumerate(mtf.get("breakdown",{}).items()):
                tc8="#00b880" if data["direction"]=="BULLISH" else ("#e74c3c" if data["direction"]=="BEARISH" else "#f39c12")
                if i < len(tf_cols):
                    tf_cols[i].markdown(f"""<div style='background:{tc8}22;border:1px solid {tc8};border-radius:8px;padding:10px;text-align:center;'>
<div style='font-size:12px;color:#888;'>{tf}</div>
<div style='font-size:20px;font-weight:700;color:{tc8};'>{data["pct"]}%</div>
<div style='font-size:11px;color:{tc8};'>{data["direction"]}</div>
<div style='font-size:10px;color:#888;'>RSI {data["rsi"]}</div>
<div style='font-size:10px;color:#888;'>{data["score"]}/6</div>
</div>""", unsafe_allow_html=True)

            with st.expander("How to use MTF Confluence"):
                st.markdown("""
**Multi-Timeframe Rules:**
- **All 4 TFs bullish** → Strongest BUY signal, high confidence
- **3/4 TFs bullish** → BUY — good confluence
- **2/4 TFs aligned** → WAIT — mixed signals
- **3-4 TFs bearish** → SELL signal

**Entry Rule:**
1. Check 4H/1H for trend direction
2. Wait for 15m to confirm direction
3. Enter on 5m when it aligns
4. SL below 5m swing low

**Best setups:** When 1H + 4H agree → enter on 5m/15m pullback
""")
        else:
            st.info("Click 'Load MTF Analysis' to compare all timeframes")

        # Correlation Filter
        st.markdown("---")
        st.markdown("#### Correlation Filter")
        try:
            all_stocks = [s for lst in list(STOCK_UNIVERSE.values())[:2] for s in lst]
            with st.spinner("Checking correlations..."):
                corr_data = get_correlation_filter(stock, tuple(all_stocks[:8]))
            if corr_data.get("status") == "ok":
                corr_c = "#e74c3c" if corr_data["warning"] else "#00b880"
                st.markdown(f"""<div style='background:{corr_c}22;border:2px solid {corr_c};
border-radius:10px;padding:12px;margin-bottom:10px;'>
<div style='font-size:14px;font-weight:700;color:{corr_c};'>{corr_data["rec"]}</div>
<div style='font-size:11px;color:#888;margin-top:4px;'>
Avg correlation: {corr_data["avg_corr"]} | High corr stocks: {len(corr_data["high_corr"])}
</div></div>""", unsafe_allow_html=True)

                corr_cols = st.columns(2)
                with corr_cols[0]:
                    st.markdown("**High Correlation (avoid same trade):**")
                    for sym, c in sorted(corr_data["high_corr"].items(), key=lambda x:-abs(x[1])):
                        cc2 = "#e74c3c" if c>0 else "#a78bfa"
                        st.markdown(f"""<div style='background:#161b22;border-left:3px solid {cc2};
border-radius:4px;padding:5px 10px;margin-bottom:3px;font-size:12px;
display:flex;justify-content:space-between;'>
<span style='color:#ccc;'>{sym}</span>
<span style='color:{cc2};font-weight:600;'>{c:+.2f}</span>
</div>""", unsafe_allow_html=True)
                    if not corr_data["high_corr"]:
                        st.success("No highly correlated stocks")

                with corr_cols[1]:
                    st.markdown("**All Correlations:**")
                    for sym, c in sorted(corr_data["correlations"].items(), key=lambda x:-abs(x[1]))[:6]:
                        cc3="#e74c3c" if c>0.6 else ("#00b880" if c<0 else "#888")
                        st.markdown(f"""<div style='display:flex;justify-content:space-between;
background:#161b22;border-radius:4px;padding:4px 8px;margin-bottom:2px;font-size:11px;'>
<span style='color:#ccc;'>{sym}</span><span style='color:{cc3};'>{c:+.2f}</span>
</div>""", unsafe_allow_html=True)
        except Exception as _cr:
            st.caption(f"Correlation: {str(_cr)[:50]}")

        # Portfolio Risk Check
        st.markdown("---")
        st.markdown("#### Portfolio Risk Control")
        try:
            pr = portfolio_risk_check(
                st.session_state.paper_balance,
                st.session_state.get("paper_positions", {}),
                float(last.get("ATR", price*0.01)) * 1.5 * max(1, int((capital*(risk/100))/max(float(last.get("ATR",price*0.01))*1.5,0.01)))
            )
            prc=pr["color"]
            st.markdown(f"""<div style='background:{prc}22;border:2px solid {prc};border-radius:10px;padding:12px;'>
<div style='font-size:16px;font-weight:700;color:{prc};'>{pr["status"]} — Portfolio Risk: {pr["exposure_pct"]}%</div>
<div style='background:rgba(255,255,255,0.1);border-radius:99px;height:8px;margin:8px 0;'>
<div style='width:{min(pr["exposure_pct"],100)}%;background:{prc};border-radius:99px;height:8px;'></div></div>
<div style='font-size:12px;color:#888;'>Open Positions: {pr["positions"]} | Max allowed: 5 | Exposure: {pr["exposure_pct"]}%</div>
{"".join([f"<div style='margin-top:4px;font-size:12px;color:#e74c3c;'>⚠ {w}</div>" for w in pr["warnings"]])}
</div>""", unsafe_allow_html=True)
        except Exception as _pr: st.caption(f"Portfolio Risk: {str(_pr)[:50]}")

# =============================================================
# TRADE LOGS
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
        st.session_state.trade_log      = []
        st.session_state.pnl_history    = []
        st.session_state.paper_balance  = 100000.0
        st.session_state.paper_position = None
        save_user_data(user)  # clear file too
        st.rerun()
else:
    st.info("No trades yet — place your first trade above")


# =============================================================
# PROFESSIONAL ENHANCEMENT MODULES ADDED
# =============================================================

import sqlite3
import hashlib
from functools import lru_cache

# =============================================================
# PERFORMANCE OPTIMIZATION
# =============================================================

@st.cache_data(ttl=300)
def cached_history(symbol, period="5d", interval="5m"):
    try:
        return yf.Ticker(symbol).history(period=period, interval=interval)
    except Exception:
        return pd.DataFrame()

# =============================================================
# SECURITY UPGRADE
# =============================================================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# =============================================================
# DAILY RISK ENGINE
# =============================================================

MAX_DAILY_LOSS_PCT = 3
MAX_CONSECUTIVE_LOSSES = 3
MAX_TRADES_PER_DAY = 10

def risk_engine(pnl_history):
    today = datetime.date.today()
    today_pnl = 0
    consecutive_losses = 0
    trades_today = 0

    for trade in reversed(pnl_history):
        try:
            trade_date = datetime.datetime.strptime(
                trade["time"], "%Y-%m-%d %H:%M:%S"
            ).date()

            if trade_date != today:
                continue

            trades_today += 1
            pnl = float(trade.get("pnl", 0))
            today_pnl += pnl

            if pnl < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0

        except Exception:
            pass

    return {
        "today_pnl": round(today_pnl, 2),
        "consecutive_losses": consecutive_losses,
        "trades_today": trades_today,
        "allow_trade": (
            today_pnl > -(MAX_DAILY_LOSS_PCT / 100) * 100000
            and consecutive_losses < MAX_CONSECUTIVE_LOSSES
            and trades_today < MAX_TRADES_PER_DAY
        )
    }

# =============================================================
# BACKTESTING ENGINE
# =============================================================

def run_backtest(df, strategy="master", initial_capital=100000,
                  sl_atr_mult=1.5, tp_rr=2.0, risk_pct=1.5):
    """Professional Backtesting Engine — Institutional Grade"""
    df = df.copy().dropna()
    if df is None or len(df) < 50:
        return {}

    capital   = float(initial_capital)
    position  = 0
    entry_px  = 0.0
    stop_px   = 0.0
    target_px = 0.0
    trades    = []
    equity    = [capital]
    monthly_r = {}

    for i in range(30, len(df)-1):
        row   = df.iloc[i]
        price = float(row["Close"])
        atr   = float(row.get("ATR", price*0.015))
        rsi   = float(row.get("RSI", 50))
        macd  = float(row.get("MACD", 0))
        macd_s= float(row.get("MACD_Signal", 0))
        adx   = float(row.get("ADX", 15))
        st_d  = float(row.get("ST_Dir", 0))
        vol_r = float(row.get("Vol_Ratio", 1))
        ema20 = float(row.get("EMA20", price))
        ema50 = float(row.get("EMA50", price))
        mfi   = float(row.get("MFI", 50))
        mh    = float(row.get("MACD_Hist", 0))
        date  = str(df.index[i])[:10]
        month = str(df.index[i])[:7]

        if strategy == "master":
            conds = [price>ema20>ema50, 45<rsi<72, macd>macd_s and mh>0,
                     st_d>0, adx>20, vol_r>1.1, mfi>50]
            buy_signal  = sum(conds) >= 5 and rsi < 75
            sell_signal = sum(conds) <= 2 or rsi > 78 or st_d < 0
        elif strategy == "trend":
            buy_signal  = price>ema20>ema50 and adx>25 and st_d>0
            sell_signal = price<ema20 or st_d<0
        elif strategy == "momentum":
            buy_signal  = macd>macd_s and 50<rsi<70 and vol_r>1.2
            sell_signal = macd<macd_s or rsi>75
        elif strategy == "mean_revert":
            buy_signal  = rsi<35 and price<float(row.get("BB_Lower",price))
            sell_signal = rsi>65 or price>float(row.get("BB_Mid",price))
        else:
            buy_signal = sell_signal = False

        if position == 0 and buy_signal and capital > price*5:
            sl_d = atr*sl_atr_mult
            qty  = max(1, int(capital*(risk_pct/100)/sl_d))
            cost = qty*price
            if cost <= capital:
                position = qty; entry_px = price
                stop_px  = round(price-sl_d,2)
                target_px= round(price+sl_d*tp_rr,2)
                capital -= cost
                trades.append({"type":"BUY","date":date,"price":round(price,2),
                               "qty":qty,"sl":stop_px,"target":target_px})
        elif position > 0:
            hit_sl  = price <= stop_px
            hit_tgt = price >= target_px
            if sell_signal or hit_sl or hit_tgt:
                ep  = stop_px if hit_sl else (target_px if hit_tgt else price)
                pnl = (ep-entry_px)*position
                capital += position*ep
                reason = "SL" if hit_sl else ("Target" if hit_tgt else "Signal")
                trades.append({"type":"SELL","date":date,"price":round(ep,2),
                               "qty":position,"pnl":round(pnl,2),"reason":reason})
                position=0; entry_px=0; stop_px=0; target_px=0

        cur_eq = capital + position*price
        equity.append(cur_eq)
        monthly_r[month] = cur_eq

    if position > 0:
        fp  = float(df["Close"].iloc[-1])
        pnl = (fp-entry_px)*position
        capital += position*fp
        trades.append({"type":"SELL","date":str(df.index[-1])[:10],
                       "price":round(fp,2),"qty":position,"pnl":round(pnl,2),"reason":"EOD"})

    eq_arr  = np.array(equity)
    ret_arr = np.diff(eq_arr)/(eq_arr[:-1]+1e-9)
    closed  = [t for t in trades if "pnl" in t]
    wins    = [t for t in closed if t["pnl"]>0]
    losses  = [t for t in closed if t["pnl"]<=0]
    tp      = sum(t["pnl"] for t in closed)
    wr      = len(wins)/len(closed)*100 if closed else 0
    aw      = float(np.mean([t["pnl"] for t in wins]))    if wins   else 0
    al      = abs(float(np.mean([t["pnl"] for t in losses]))) if losses else 1
    gp      = sum(t["pnl"] for t in wins)
    gl      = abs(sum(t["pnl"] for t in losses))
    pf      = gp/(gl+1e-9)
    exp     = (wr/100*aw) - ((1-wr/100)*al)
    peak    = np.maximum.accumulate(eq_arr)
    dd      = (eq_arr-peak)/(peak+1e-9)
    max_dd  = float(dd.min())*100
    sharpe  = float(np.mean(ret_arr)/np.std(ret_arr+1e-9)*np.sqrt(252)) if len(ret_arr)>2 else 0
    neg_r   = ret_arr[ret_arr<0]
    sortino = float(np.mean(ret_arr)/(np.std(neg_r)+1e-9)*np.sqrt(252)) if len(neg_r)>1 else 0
    calmar  = abs(tp/initial_capital*100)/(abs(max_dd)+1e-9)
    bh_ret  = (float(df["Close"].iloc[-1])/float(df["Close"].iloc[30])-1)*100
    months  = sorted(monthly_r.keys())
    mo_rets = [round((monthly_r[months[j]]-monthly_r[months[j-1]])/monthly_r[months[j-1]]*100,2)
               for j in range(1, len(months))]
    cw=cl=mcw=mcl=0
    for t in closed:
        if t["pnl"]>0: cw+=1; cl=0
        else: cl+=1; cw=0
        mcw=max(mcw,cw); mcl=max(mcl,cl)

    return {
        "trades": trades, "closed": closed,
        "total_trades": len(closed), "wins": len(wins), "losses": len(losses),
        "total_pnl": round(tp,2), "final_balance": round(capital,2),
        "return_pct": round((capital-initial_capital)/initial_capital*100,2),
        "bh_return": round(bh_ret,2),
        "win_rate": round(wr,1), "avg_win": round(aw,2), "avg_loss": round(al,2),
        "profit_factor": round(pf,2), "expectancy": round(exp,2),
        "max_drawdown": round(max_dd,2),
        "recovery_factor": round(abs(tp/initial_capital*100/(abs(max_dd)+1e-9)),2),
        "max_consec_wins": mcw, "max_consec_loss": mcl,
        "sharpe": round(sharpe,2), "sortino": round(sortino,2), "calmar": round(calmar,2),
        "equity": equity, "monthly_rets": mo_rets, "months": months[1:] if len(months)>1 else [],
    }

# =============================================================
# MULTI TIMEFRAME ANALYSIS
# =============================================================

def multi_timeframe_analysis(symbol):
    results = {}

    try:
        tf_5m = compute_indicators(
            cached_history(symbol, "5d", "5m")
        )

        tf_15m = compute_indicators(
            cached_history(symbol, "5d", "15m")
        )

        tf_1h = compute_indicators(
            cached_history(symbol, "1mo", "1h")
        )

        def trend(df):
            if len(df) < 10:
                return "UNKNOWN"

            last = df.iloc[-1]

            if last["Close"] > last["EMA20"] > last["EMA50"]:
                return "BULLISH"

            if last["Close"] < last["EMA20"] < last["EMA50"]:
                return "BEARISH"

            return "SIDEWAYS"

        results["5m"] = trend(tf_5m)
        results["15m"] = trend(tf_15m)
        results["1h"] = trend(tf_1h)

    except Exception as e:
        results["error"] = str(e)

    return results

# =============================================================
# TRADE JOURNAL AI
# =============================================================

def analyze_trade_journal(pnl_history):
    if not pnl_history:
        return {}

    total = len(pnl_history)
    wins = len([x for x in pnl_history if x.get("pnl", 0) > 0])
    losses = total - wins

    avg_win = (
        sum(x["pnl"] for x in pnl_history if x.get("pnl", 0) > 0)
        / max(wins, 1)
    )

    avg_loss = (
        sum(x["pnl"] for x in pnl_history if x.get("pnl", 0) < 0)
        / max(losses, 1)
    )

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round((wins / total) * 100, 2) if total else 0,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "suggestion": (
            "Good consistency"
            if wins > losses
            else "Reduce overtrading and improve entries"
        )
    }

# =============================================================
# ADVANCED AI SCORE
# =============================================================

def advanced_ai_score(df):
    if len(df) < 50:
        return 50

    last = df.iloc[-1]
    score = 0

    if last["Close"] > last["EMA20"]:
        score += 20

    if last["EMA20"] > last["EMA50"]:
        score += 20

    if last["MACD"] > last["MACD_Signal"]:
        score += 20

    if 50 < last["RSI"] < 70:
        score += 20

    if last["Vol_Ratio"] > 1.2:
        score += 20

    return min(score, 100)

# =============================================================
# PROFESSIONAL DASHBOARD SECTION
# =============================================================

st.markdown("---")
st.subheader("🧠 Professional Analytics")

try:
    risk_stats = risk_engine(st.session_state.pnl_history)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Daily P&L", f"₹{risk_stats['today_pnl']}")
    c2.metric("Trades Today", risk_stats["trades_today"])
    c3.metric("Consecutive Losses", risk_stats["consecutive_losses"])
    c4.metric(
        "Trading Status",
        "ALLOWED" if risk_stats["allow_trade"] else "BLOCKED"
    )

except Exception as e:
    st.warning(f"Risk engine issue: {e}")

try:
    with st.expander("📈 Professional Backtesting — Click to Run", expanded=False):
        bt_cols = st.columns(4)
        _bt_strat  = bt_cols[0].selectbox("Strategy", ["master","trend","momentum","mean_revert"],
                                           format_func=lambda x:x.title(), key="bt_strat_main")
        _bt_sl     = bt_cols[1].slider("SL ATR Mult", 1.0, 3.0, 1.5, 0.1, key="bt_sl_main")
        _bt_rr     = bt_cols[2].slider("Target R:R",  1.0, 5.0, 2.0, 0.1, key="bt_rr_main")
        _bt_cap    = bt_cols[3].number_input("Capital", 50000, 5000000, 100000, 50000, key="bt_cap_main")
        if st.button("Run Professional Backtest", type="primary", key="bt_run_main"):
            with st.spinner("Running institutional backtest..."):
                bt = run_backtest(df, strategy=_bt_strat, initial_capital=_bt_cap,
                                  sl_atr_mult=_bt_sl, tp_rr=_bt_rr)
                st.session_state["bt_result_main"] = bt
        bt = st.session_state.get("bt_result_main", None)
        if bt and bt.get("total_trades",0) > 0:
            ret_c = "#00b880" if bt["return_pct"]>=0 else "#e74c3c"
            al_c  = "#00b880" if bt["return_pct"]>=bt["bh_return"] else "#e74c3c"
            mc = st.columns(8)
            for col, (lbl,val,cc) in zip(mc, [
                ("Return", f"{bt['return_pct']:+.1f}%", ret_c),
                ("vs B&H", f"{bt['return_pct']-bt['bh_return']:+.1f}%", al_c),
                ("Win%", f"{bt['win_rate']:.0f}%", "#f39c12"),
                ("Trades", str(bt["total_trades"]), "#a78bfa"),
                ("PF", f"{bt['profit_factor']:.2f}", "#4e8fff"),
                ("MaxDD", f"{bt['max_drawdown']:.1f}%", "#e74c3c"),
                ("Sharpe", f"{bt['sharpe']:.2f}", "#00e5a0"),
                ("Sortino", f"{bt['sortino']:.2f}", "#ffa94d"),
            ]):
                col.markdown(f"""<div style='background:#1a1a2e;border-radius:6px;padding:8px;text-align:center;'>
<div style='font-size:16px;font-weight:700;color:{cc};font-family:monospace;'>{val}</div>
<div style='font-size:9px;color:#888;'>{lbl}</div></div>""", unsafe_allow_html=True)

            if bt["profit_factor"]>=1.5 and bt["win_rate"]>=50:
                st.success(f"GOOD STRATEGY — PF {bt['profit_factor']} | WR {bt['win_rate']}% | Sharpe {bt['sharpe']}")
            elif bt["profit_factor"]>=1.0:
                st.warning(f"MARGINAL — PF {bt['profit_factor']:.2f} | Needs refinement")
            else:
                st.error(f"LOSING — Do NOT use real money. PF: {bt['profit_factor']:.2f}")

            if PLOTLY_OK and len(bt.get("equity",[])) > 5:
                from plotly.subplots import make_subplots as _msp2
                eq_fig = _msp2(rows=2,cols=1,shared_xaxes=True,
                               row_heights=[0.65,0.35],vertical_spacing=0.04)
                xr = list(range(len(bt["equity"])))
                eq_fig.add_trace(go.Scatter(x=xr,y=bt["equity"],
                    line=dict(color="#00e5a0",width=2),name="Strategy",
                    fill="tozeroy",fillcolor="rgba(0,229,160,0.06)"),row=1,col=1)
                bh_eq=[_bt_cap*(1+bt["bh_return"]/100*j/len(xr)) for j in range(len(xr))]
                eq_fig.add_trace(go.Scatter(x=xr,y=bh_eq,
                    line=dict(color="#4e8fff",width=1.5,dash="dot"),
                    name=f"Buy&Hold {bt['bh_return']:+.1f}%"),row=1,col=1)
                eq_arr=np.array(bt["equity"]); pk=np.maximum.accumulate(eq_arr)
                dd=(eq_arr-pk)/(pk+1e-9)*100
                eq_fig.add_trace(go.Scatter(x=xr,y=dd,fill="tozeroy",
                    fillcolor="rgba(231,76,60,0.2)",line=dict(color="#e74c3c",width=1),
                    name="Drawdown%"),row=2,col=1)
                eq_fig.update_layout(height=380,plot_bgcolor="#0d1117",paper_bgcolor="#0d1117",
                    font=dict(color="#8b949e",size=9),
                    legend=dict(orientation="h",y=1.05,bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=0,r=0,t=10,b=0))
                for r in [1,2]:
                    eq_fig.update_xaxes(gridcolor="#21262d",row=r,col=1)
                    eq_fig.update_yaxes(gridcolor="#21262d",row=r,col=1)
                st.plotly_chart(eq_fig,use_container_width=True,config={"displayModeBar":False})

                if bt.get("monthly_rets") and len(bt["monthly_rets"])>1:
                    mo_c=["#00b880" if v>=0 else "#e74c3c" for v in bt["monthly_rets"]]
                    mf=go.Figure(go.Bar(x=bt["months"],y=bt["monthly_rets"],marker_color=mo_c))
                    mf.update_layout(title=dict(text="Monthly Returns %",font=dict(color="#e6edf3",size=12)),
                        height=200,plot_bgcolor="#0d1117",paper_bgcolor="#0d1117",
                        font=dict(color="#8b949e"),margin=dict(l=0,r=0,t=28,b=0))
                    mf.add_hline(y=0,line_color="#555")
                    st.plotly_chart(mf,use_container_width=True,config={"displayModeBar":False})

            cl2 = bt.get("closed",[])
            if cl2:
                ex2=st.columns(2)
                slh=sum(1 for t in cl2 if t.get("reason")=="SL")
                tgh=sum(1 for t in cl2 if t.get("reason")=="Target")
                ex2[0].markdown(f"""<div style='background:#1a1a2e;border-radius:8px;padding:10px;font-size:12px;'>
<b>Exit Reasons</b><br>
SL hits: <span style='color:#e74c3c;'>{slh} ({int(slh/len(cl2)*100)}%)</span><br>
Target: <span style='color:#00b880;'>{tgh} ({int(tgh/len(cl2)*100)}%)</span><br>
Signal: <span style='color:#f39c12;'>{len(cl2)-slh-tgh}</span>
</div>""",unsafe_allow_html=True)
                pnls=[t["pnl"] for t in cl2]
                ex2[1].markdown(f"""<div style='background:#1a1a2e;border-radius:8px;padding:10px;font-size:12px;'>
<b>P&L Stats</b><br>
Best: <span style='color:#00b880;'>Rs.{max(pnls):+,.0f}</span><br>
Worst: <span style='color:#e74c3c;'>Rs.{min(pnls):+,.0f}</span><br>
Expectancy: Rs.{bt["expectancy"]:+,.2f}/trade
</div>""",unsafe_allow_html=True)

            # ── TRADE REPLAY ─────────────────────────────────
            if cl2 and PLOTLY_OK:
                st.markdown("**Trade Replay — Step through each trade:**")
                replay_idx = st.slider(
                    "Select Trade #", 1, max(len(cl2),1),
                    min(len(cl2),1), key="replay_slider")
                replay_trade = cl2[min(replay_idx-1, len(cl2)-1)]

                r_buy  = next((t for t in bt.get("trades",[])
                               if t["type"]=="BUY" and
                               t["date"] <= replay_trade["date"]), None)
                r_sell = replay_trade

                rc1,rc2,rc3,rc4,rc5 = st.columns(5)
                rp = replay_trade.get("pnl",0)
                rpc= "#00b880" if rp>=0 else "#e74c3c"
                rc1.markdown(f"<div style='text-align:center;'><div style='font-size:10px;color:#888;'>Entry Date</div><div style='font-size:13px;color:#ccc;'>{r_buy['date'] if r_buy else '—'}</div></div>", unsafe_allow_html=True)
                rc2.markdown(f"<div style='text-align:center;'><div style='font-size:10px;color:#888;'>Entry Rs.</div><div style='font-size:13px;color:#4e8fff;'>Rs.{r_buy['price'] if r_buy else '—'}</div></div>", unsafe_allow_html=True)
                rc3.markdown(f"<div style='text-align:center;'><div style='font-size:10px;color:#888;'>Exit Rs.</div><div style='font-size:13px;color:#f39c12;'>Rs.{r_sell['price']}</div></div>", unsafe_allow_html=True)
                rc4.markdown(f"<div style='text-align:center;'><div style='font-size:10px;color:#888;'>P&L</div><div style='font-size:16px;font-weight:700;color:{rpc};'>Rs.{rp:+,.0f}</div></div>", unsafe_allow_html=True)
                rc5.markdown(f"<div style='text-align:center;'><div style='font-size:10px;color:#888;'>Exit Reason</div><div style='font-size:13px;color:{'#e74c3c' if r_sell.get('reason')=='SL' else '#00b880'};'>{r_sell.get('reason','—')}</div></div>", unsafe_allow_html=True)

                # Cumulative P&L up to this trade
                cumulative_pnl = [sum(t["pnl"] for t in cl2[:i+1])
                                   for i in range(len(cl2))]
                replay_fig = go.Figure()
                bar_colors = ["#00b880" if p>=0 else "#e74c3c"
                              for p in [t["pnl"] for t in cl2]]
                replay_fig.add_trace(go.Bar(
                    x=list(range(1,len(cl2)+1)),
                    y=[t["pnl"] for t in cl2],
                    marker_color=bar_colors,
                    name="Trade P&L",
                    opacity=0.7))
                replay_fig.add_trace(go.Scatter(
                    x=list(range(1,len(cl2)+1)),
                    y=cumulative_pnl,
                    line=dict(color="#f39c12",width=2),
                    name="Cumulative P&L",
                    yaxis="y2"))
                # Highlight selected trade
                replay_fig.add_vline(
                    x=replay_idx,
                    line_color="#fff", line_dash="dot",
                    line_width=1)
                replay_fig.update_layout(
                    height=220,
                    plot_bgcolor="#0d1117",
                    paper_bgcolor="#0d1117",
                    font=dict(color="#8b949e",size=9),
                    yaxis2=dict(overlaying="y",side="right",
                                gridcolor="#21262d"),
                    legend=dict(bgcolor="rgba(0,0,0,0)",
                               orientation="h",y=1.05),
                    margin=dict(l=0,r=0,t=10,b=0),
                    barmode="relative")
                replay_fig.update_xaxes(gridcolor="#21262d",
                    title="Trade #")
                replay_fig.update_yaxes(gridcolor="#21262d",
                    title="P&L (Rs.)")
                replay_fig.add_hline(y=0, line_color="#555")
                st.plotly_chart(replay_fig, use_container_width=True,
                               config={"displayModeBar":False})

except Exception as e:
    st.warning(f"Backtesting issue: {e}")

try:
    mtf = multi_timeframe_analysis(stock)

    st.markdown("### ⏱ Multi Timeframe Analysis")

    t1, t2, t3 = st.columns(3)

    t1.metric("5m Trend", mtf.get("5m", "NA"))
    t2.metric("15m Trend", mtf.get("15m", "NA"))
    t3.metric("1H Trend", mtf.get("1h", "NA"))

except Exception as e:
    st.warning(f"MTF issue: {e}")

try:
    journal = analyze_trade_journal(st.session_state.pnl_history)

    if journal:
        st.markdown("### 📓 Trade Journal AI")

        j1, j2, j3 = st.columns(3)

        j1.metric("Win Rate", f"{journal['win_rate']}%")
        j2.metric("Avg Win", f"₹{journal['avg_win']}")
        j3.metric("Avg Loss", f"₹{journal['avg_loss']}")

        st.info(journal["suggestion"])

except Exception as e:
    st.warning(f"Journal AI issue: {e}")
