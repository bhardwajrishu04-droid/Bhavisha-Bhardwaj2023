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


# =============================================================
# GLOBAL UI COLOR VARIABLES FIX
# =============================================================
meter_bg = "#001a12"
meter_color = "#00e5a0"



# =============================================================
# GLOBAL UI LABEL FIXES
# =============================================================
meter_label = "AI SIGNAL"
meter_bg = "#001a12"
meter_color = "#00e5a0"

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
}

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

    # ── ULTRA ADVANCED AI ENGINE ─────────────────────────────
    # Models: XGBoost + LightGBM + RF + GB + AdaBoost
    # Walk-Forward TimeSeriesSplit Validation
    # ─────────────────────────────────────────────────────────
    # Core features — always available after compute_indicators
    FEAT_COLS_CORE = [
        "EMA9","EMA20","EMA50","RSI","MACD","MACD_Hist",
        "Stoch_K","BB_Pct","BB_Width","Vol_Ratio",
        "Return_1","Return_3","Price_Pos"
    ]
    # Extended features — only if enough data
    FEAT_COLS_EXT = [
        "RSI_MA","Stoch_D","ADX","MFI","CCI",
        "Williams_R","OBV"
    ]
    # Use extended only if data is sufficient
    _all_feat = FEAT_COLS_CORE + [f for f in FEAT_COLS_EXT if f in df.columns]
    feat_cols  = [c for c in _all_feat if c in df.columns]

    df["T1"] = (df["Close"].shift(-1) > df["Close"] * 1.002).astype(int)
    df["T3"] = (df["Close"].shift(-3) > df["Close"] * 1.005).astype(int)

    # Fill NaN with forward-fill then 0 — don't drop rows
    _fd_raw = df[feat_cols].copy()
    _fd_raw = _fd_raw.ffill().fillna(0)
    fd = _fd_raw.iloc[20:]   # skip first 20 (indicators warming up)

    ai_prob       = 0.5
    _data_count   = len(fd)
    ai_model_name = f"Training... ({_data_count} samples)"
    ai_confidence = "Low"
    ai_accuracy   = 0.0
    feature_importance = {}
    wf_results    = []   # walk-forward fold accuracies

    if len(fd) >= 40:
        try:
            scaler   = StandardScaler()
            # Use T1 if tight data, T3 if enough
            _target_col = "T3" if len(fd) >= 60 else "T1"
            td       = df[_target_col].loc[fd.index]
            td       = td.fillna(0)          # fill any NaN targets
            td       = td.iloc[:-3]          # remove last 3 (future unknown)
            fd_clean = fd.loc[td.index]
            X_all    = scaler.fit_transform(fd_clean.values)
            y_all    = td.values

            # ── WALK-FORWARD VALIDATION (3-5 folds based on data) ───
            n_folds = 5 if len(fd_clean) >= 80 else 3
            tscv = TimeSeriesSplit(n_splits=n_folds)
            fold_accs = []
            for fold_tr, fold_val in tscv.split(X_all):
                Xf_tr, Xf_val = X_all[fold_tr], X_all[fold_val]
                yf_tr, yf_val = y_all[fold_tr], y_all[fold_val]
                if len(set(yf_tr)) < 2 or len(Xf_val) < 2: continue
                if len(set(yf_val)) < 1: continue
                try:
                    _rf = RandomForestClassifier(n_estimators=50,
                            max_depth=4, random_state=42)
                    _rf.fit(Xf_tr, yf_tr)
                    fold_acc = accuracy_score(yf_val, _rf.predict(Xf_val))*100
                    fold_accs.append(round(fold_acc, 1))
                except Exception:
                    continue
            wf_results = fold_accs if fold_accs else []

            # Final train/val split (last fold — 80/20 fallback for small data)
            splits = list(tscv.split(X_all))
            if splits:
                tr_idx, val_idx = splits[-1]
            else:
                cut = int(len(X_all)*0.8)
                tr_idx = list(range(cut))
                val_idx = list(range(cut, len(X_all)))
            X_tr, X_val = X_all[tr_idx], X_all[val_idx]
            y_tr, y_val = y_all[tr_idx], y_all[val_idx]

            # Need at least 2 classes in train set
            if len(set(y_tr)) < 2 or len(X_val) < 3:
                raise ValueError("Insufficient class distribution")

            model_probs  = []
            model_labels = []
            all_importances = []

            # ── MODEL 1: XGBoost ────────────────────────────
            if XGB_OK:
                try:
                    _n_est = 200 if len(X_tr) >= 100 else 100
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=_n_est, max_depth=4,
                        learning_rate=0.08, subsample=0.8,
                        colsample_bytree=0.8, reg_alpha=0.1,
                        eval_metric="logloss", random_state=42,
                        verbosity=0
                    )
                    xgb_model.fit(X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        verbose=False)
                    p = xgb_model.predict_proba(X_all[-1:].reshape(1,-1))[0][1]
                    a = accuracy_score(y_val, xgb_model.predict(X_val)) * 100
                    model_probs.append(p)
                    model_labels.append(f"XGBoost {a:.1f}%")
                    all_importances.append(dict(zip(feat_cols, xgb_model.feature_importances_)))
                except Exception:
                    pass

            # ── MODEL 2: LightGBM ───────────────────────────
            if LGB_OK:
                try:
                    _lgb_n = 200 if len(X_tr) >= 100 else 80
                    lgb_model = lgb.LGBMClassifier(
                        n_estimators=_lgb_n, num_leaves=15,
                        learning_rate=0.08, subsample=0.8,
                        colsample_bytree=0.8, reg_alpha=0.1,
                        min_child_samples=5,
                        random_state=42, verbose=-1
                    )
                    lgb_model.fit(X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(20, verbose=False),
                                   lgb.log_evaluation(-1)])
                    p = lgb_model.predict_proba(X_all[-1:].reshape(1,-1))[0][1]
                    a = accuracy_score(y_val, lgb_model.predict(X_val)) * 100
                    model_probs.append(p)
                    model_labels.append(f"LightGBM {a:.1f}%")
                    all_importances.append(dict(zip(feat_cols,
                        lgb_model.feature_importances_ / (lgb_model.feature_importances_.sum()+1e-9))))
                except Exception:
                    pass

            # ── MODEL 3: Gradient Boosting ──────────────────
            try:
                gb = GradientBoostingClassifier(
                    n_estimators=200, max_depth=4,
                    learning_rate=0.05, subsample=0.8, random_state=42)
                gb.fit(X_tr, y_tr)
                p = gb.predict_proba(X_all[-1:].reshape(1,-1))[0][1]
                a = accuracy_score(y_val, gb.predict(X_val)) * 100
                model_probs.append(p)
                model_labels.append(f"GradBoost {a:.1f}%")
                all_importances.append(dict(zip(feat_cols, gb.feature_importances_)))
            except Exception:
                pass

            # ── MODEL 4: RandomForest ───────────────────────
            try:
                rf = RandomForestClassifier(
                    n_estimators=300, max_depth=6,
                    min_samples_leaf=5, random_state=42)
                rf.fit(X_tr, y_tr)
                p = rf.predict_proba(X_all[-1:].reshape(1,-1))[0][1]
                a = accuracy_score(y_val, rf.predict(X_val)) * 100
                model_probs.append(p)
                model_labels.append(f"RandomForest {a:.1f}%")
                all_importances.append(dict(zip(feat_cols, rf.feature_importances_)))
            except Exception:
                pass

            # ── MODEL 5: AdaBoost ───────────────────────────
            try:
                ada = AdaBoostClassifier(
                    n_estimators=100, learning_rate=0.1, random_state=42)
                ada.fit(X_tr, y_tr)
                p = ada.predict_proba(X_all[-1:].reshape(1,-1))[0][1]
                a = accuracy_score(y_val, ada.predict(X_val)) * 100
                model_probs.append(p)
                model_labels.append(f"AdaBoost {a:.1f}%")
            except Exception:
                pass

            # ── ENSEMBLE AVERAGE ─────────────────────────────
            if model_probs:
                ai_prob      = float(np.mean(model_probs))
                ai_accuracy  = round(float(np.mean([
                    float(l.split()[-1].replace("%",""))
                    for l in model_labels])), 1)
                model_count  = len(model_probs)
                ai_model_name = (
                    f"{'XGB+' if XGB_OK else ''}{'LGB+' if LGB_OK else ''}"
                    f"GB+RF+Ada ({model_count} models) | "
                    f"Avg Acc: {ai_accuracy}%"
                )
                # Average feature importance
                if all_importances:
                    combined = {}
                    for imp_dict in all_importances:
                        for k,v in imp_dict.items():
                            combined[k] = combined.get(k,0) + v/len(all_importances)
                    feature_importance = dict(sorted(
                        combined.items(), key=lambda x:-x[1])[:8])

            if   ai_accuracy >= 65: ai_confidence = "High"
            elif ai_accuracy >= 55: ai_confidence = "Medium"
            else:                   ai_confidence = "Low"

        except Exception as _ae:
            try:
                td2 = df["T1"].loc[fd.index].dropna()
                fd2 = fd.loc[td2.index]
                rf2 = RandomForestClassifier(n_estimators=200, random_state=42)
                rf2.fit(fd2.values[:-1], td2.values[:-1])
                ai_prob = rf2.predict_proba(fd2.iloc[-1:].values)[0][1]
                ai_model_name = "RandomForest (fallback)"
            except Exception:
                ai_prob = 0.5
    else:
        ai_prob = 0.5

    # ── MASTER SIGNAL ENGINE ─────────────────────────────────
    st.markdown("---")

    # LAYER 1: Technical (30%)
    c_trend  = last["Close"] > last["EMA20"] > last["EMA50"]
    c_ema9   = last["Close"] > last.get("EMA9", last["Close"])
    c_rsi    = 45 < last["RSI"] < 68
    c_macd   = last["MACD"] > last["MACD_Signal"]
    c_macd_h = float(last.get("MACD_Hist", 0)) > 0
    c_vol    = float(last["Vol_Ratio"]) > 1.1
    c_bb     = last["Close"] > float(last.get("BB_Mid", last["Close"]))
    c_stoch  = float(last.get("Stoch_K", 50)) < 70
    rsi_ob   = float(last["RSI"]) > 75
    rsi_os   = float(last["RSI"]) < 30
    tech_checks = {
        "Trend: Price > EMA20 > EMA50": c_trend,
        "Price above EMA9":             c_ema9,
        "RSI in zone (45-68)":          c_rsi,
        "MACD above Signal line":       c_macd,
        "MACD Histogram positive":      c_macd_h,
        "Volume surge (>1.1x)":         c_vol,
        "Price above BB Midline":       c_bb,
        "Stochastic not overbought":    c_stoch,
    }
    tech_score = sum(tech_checks.values())
    tech_pct   = round(tech_score / len(tech_checks) * 100)

    # LAYER 2: AI (25%)
    ai_pct = round(ai_prob * 100)
    c_ai   = ai_prob > 0.55
    score  = sum([c_trend, c_rsi, c_macd, c_vol, c_ai])

    # LAYER 3: Candlestick (15%)
    candle_pct = 50; candle_top = "None"
    try:
        _cd = df.dropna(subset=["Close","EMA20"]).tail(50).copy()
        _cd.index = pd.to_datetime(_cd.index)
        if "ATR" not in _cd.columns:
            _cd["ATR"] = (_cd["High"]-_cd["Low"]).rolling(14).mean()
        _pts = detect_candlestick_patterns(_cd)
        if _pts:
            _tp = sorted(_pts, key=lambda x: -x["strength"])[0]
            candle_top = _tp["pattern"]
            if _tp["type"]=="bullish":   candle_pct = min(100, 50+_tp["strength"]*10)
            elif _tp["type"]=="bearish": candle_pct = max(0,   50-_tp["strength"]*10)
    except Exception:
        pass

    # LAYER 4: Market Structure (15%)
    struct_pct = 50; struct_label = "Unknown"
    try:
        _sd = df.tail(60).copy()
        if "ATR" not in _sd.columns:
            _sd["ATR"] = (_sd["High"]-_sd["Low"]).rolling(14).mean()
        _ms = detect_market_structure(_sd)
        if _ms and "hh" in _ms:
            if   _ms["hh"] and _ms["hl"]: struct_pct=85; struct_label="Uptrend HH+HL"
            elif _ms["lh"] and _ms["ll"]: struct_pct=20; struct_label="Downtrend LH+LL"
            else:                          struct_pct=50; struct_label=_ms.get("trend","Ranging")[:18]
            if _ms.get("mss") and "BULLISH" in _ms.get("mss",""):
                struct_pct = min(100, struct_pct+15)
    except Exception:
        pass

    # LAYER 5: SMC — Order Block + FVG (10%)
    smc_pct=50; smc_label="No clear OB/FVG"
    try:
        _ob_df = df.tail(80).copy()
        if "ATR" not in _ob_df.columns:
            _ob_df["ATR"] = (_ob_df["High"]-_ob_df["Low"]).rolling(14).mean()
        _obs  = find_order_blocks(_ob_df)
        _fvgs = find_fvg(_ob_df)
        _b_ob = [o for o in _obs if o["type"]=="Bullish Order Block"]
        _b_fvg= [f for f in _fvgs if f["type"]=="Bullish FVG"]
        if _b_ob and price < _b_ob[-1]["top"]*1.02:
            smc_pct=80; smc_label="Near Bullish OB"
        elif _b_fvg:
            smc_pct=70; smc_label="Bullish FVG"
    except Exception:
        pass

    # LAYER 6: Volume (5%)
    _vr = float(last.get("Vol_Ratio",1))
    if _vr>2.0:   vol_pct=90; vol_label="Very High Vol"
    elif _vr>1.5: vol_pct=75; vol_label="Above Avg Vol"
    elif _vr>1.0: vol_pct=55; vol_label="Normal Vol"
    else:         vol_pct=30; vol_label="Low Vol"

    # MASTER WEIGHTED SCORE
    master = round(
        tech_pct   * 0.30 +
        ai_pct     * 0.25 +
        candle_pct * 0.15 +
        struct_pct * 0.15 +
        smc_pct    * 0.10 +
        vol_pct    * 0.05
    )

    # RISK PENALTIES
    _risks=[]; _pen=0
    if rsi_ob:
        _pen+=15; _risks.append(("RSI Overbought","Pull back expected","#e74c3c"))
    if _vr<0.7:
        _pen+=10; _risks.append(("Low Volume","Weak signal — skip","#f39c12"))
    if float(last.get("MACD_Hist",0))<0 and float(last.get("MACD",0))>float(last.get("MACD_Signal",0)):
        _pen+=5;  _risks.append(("MACD Divergence","Momentum weakening","#f39c12"))
    try:
        _sz,_ = find_demand_supply_zones(df.tail(80))
        for _z in _sz:
            if abs(price-_z["top"])/price < 0.01:
                _pen+=10; _risks.append(("Near Supply Zone",f"Resistance Rs.{_z['top']}","#e74c3c"))
    except Exception:
        pass
    master = max(0, master - _pen)

    # FINAL VERDICT
    if force_trade:          direction="TRADE";       v_color="#00b880"; v_bg="#003d2a"
    elif rsi_ob and master<70:direction="NO TRADE";   v_color="#e74c3c"; v_bg="#2d0a0a"
    elif master>=78:          direction="STRONG BUY"; v_color="#00b880"; v_bg="#003d2a"
    elif master>=65:          direction="BUY";        v_color="#27ae60"; v_bg="#0a1f10"
    elif master>=50:          direction="WAIT";       v_color="#f39c12"; v_bg="#1a1200"
    elif master>=35:          direction="AVOID";      v_color="#e07b39"; v_bg="#2d1800"
    else:                     direction="NO TRADE";   v_color="#e74c3c"; v_bg="#2d0a0a"

    signal    = direction in ["STRONG BUY","BUY","TRADE"]
    combined  = master  # keep for compatibility

    # TRADE PLAN CALC
    atr_now      = float(last["ATR"])
    _sl_d        = atr_now * mcfg.get("sl_mult",1.5)
    _tg_d        = _sl_d   * mcfg.get("rr",2.0)
    stop_loss_m  = round(price - _sl_d, 2)
    target_m     = round(price + _tg_d, 2)
    qty_m        = max(1, int((capital*(risk/100)) / _sl_d))
    max_loss_rs  = round(_sl_d * qty_m, 2)
    max_gain_rs  = round(_tg_d * qty_m, 2)
    win_prob     = min(82, round(master*0.7+15))
    _rr_m        = round(_tg_d/_sl_d, 1)

    # RISK WARNINGS HTML
    _risk_html = "".join([
        "<div style='background:rgba(231,76,60,0.1);border-left:3px solid " + r[2] + ";border-radius:5px;"
        "padding:6px 12px;margin-top:6px;font-size:12px;'>"
        "<span style=\'color:" + r[2] + ";font-weight:600;\'>Warning: " + r[0] + "</span> — " + r[1] + "</div>"
        for r in _risks
    ]) if _risks else (
        "<div style='background:rgba(0,184,128,0.08);border-left:3px solid #00b880;"
        "border-radius:5px;padding:6px 12px;margin-top:6px;font-size:12px;color:#00b880;'>"
        "No major risk factors detected</div>"
    )

    st.markdown(f"""
<div style='background:{v_bg};border:3px solid {v_color};border-radius:16px;padding:22px 26px;margin-bottom:16px;'>

  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;'>
    <div>
      <div style='font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px;'>
        MASTER SIGNAL — {stock.replace(".NS","")} {selected_mode}
      </div>
      <div style='font-size:40px;font-weight:800;color:{v_color};line-height:1;'>{direction}</div>
      <div style='font-size:12px;color:#aaa;margin-top:4px;'>
        6-layer analysis: Technical + AI + Candles + Structure + SMC + Volume
      </div>
    </div>
    <div style='text-align:right;'>
      <div style='font-size:52px;font-weight:800;color:{v_color};line-height:1;'>{master}%</div>
      <div style='font-size:12px;color:#888;'>Overall Confidence</div>
    </div>
  </div>

  <div style='background:rgba(255,255,255,0.08);border-radius:99px;height:18px;margin-bottom:6px;position:relative;'>
    <div style='width:{master}%;background:linear-gradient(90deg,{v_color}88,{v_color});
    border-radius:99px;height:18px;box-shadow:0 0 14px {v_color}55;'></div>
    <div style='position:absolute;left:35%;top:-6px;width:2px;height:30px;background:#e07b39;opacity:0.5;'></div>
    <div style='position:absolute;left:50%;top:-6px;width:2px;height:30px;background:#f39c12;opacity:0.5;'></div>
    <div style='position:absolute;left:65%;top:-6px;width:2px;height:30px;background:#27ae60;opacity:0.5;'></div>
    <div style='position:absolute;left:78%;top:-6px;width:2px;height:30px;background:#00b880;opacity:0.7;'></div>
  </div>
  <div style='display:flex;justify-content:space-between;font-size:10px;color:#555;margin-bottom:16px;'>
    <span>0 NO TRADE</span>
    <span style='color:#e07b39;'>35 AVOID</span>
    <span style='color:#f39c12;'>50 WAIT</span>
    <span style='color:#27ae60;'>65 BUY</span>
    <span style='color:#00b880;'>78 STRONG</span>
    <span>100</span>
  </div>

  <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px;'>
    {"".join([
      "<div style='background:rgba(0,0,0,0.35);border-radius:8px;padding:10px;text-align:center;border:1px solid " +
      ("#00b880" if v>=65 else ("#f39c12" if v>=45 else "#e74c3c")) + "33;'>" +
      "<div style='font-size:10px;color:#888;margin-bottom:3px;'>" + lbl + "</div>" +
      "<div style='font-size:22px;font-weight:700;color:" +
      ("#00b880" if v>=65 else ("#f39c12" if v>=45 else "#e74c3c")) + ";'>" + str(v) + "%</div>" +
      "<div style='font-size:10px;color:#666;'>" + sub + "</div></div>"
      for lbl,v,sub in [
        ("Technical 30%",  tech_pct,    str(tech_score)+"/8 checks"),
        ("AI Model 25%",   ai_pct,      "Bullish" if ai_pct>=60 else ("Neutral" if ai_pct>=40 else "Bearish")),
        ("Candlestick 15%",candle_pct,  candle_top[:14]),
        ("Structure 15%",  struct_pct,  struct_label[:17]),
        ("SMC 10%",        smc_pct,     smc_label[:15]),
        ("Volume 5%",      vol_pct,     vol_label[:14]),
      ]
    ])}
  </div>

  <div style='background:rgba(0,0,0,0.3);border-radius:10px;padding:12px 16px;margin-bottom:10px;'>
    <div style='font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;'>Trade Plan</div>
    <div style='display:grid;grid-template-columns:repeat(5,1fr);gap:6px;text-align:center;'>
      <div><div style='font-size:10px;color:#888;'>Entry</div><div style='font-size:14px;font-weight:600;color:#e6edf3;'>Rs.{price:.2f}</div></div>
      <div><div style='font-size:10px;color:#e74c3c;'>Stop Loss</div><div style='font-size:14px;font-weight:600;color:#e74c3c;'>Rs.{stop_loss_m}</div></div>
      <div><div style='font-size:10px;color:#00b880;'>Target</div><div style='font-size:14px;font-weight:600;color:#00b880;'>Rs.{target_m}</div></div>
      <div><div style='font-size:10px;color:#f39c12;'>R:R</div><div style='font-size:14px;font-weight:600;color:#f39c12;'>{_rr_m}:1</div></div>
      <div><div style='font-size:10px;color:#a78bfa;'>Qty</div><div style='font-size:14px;font-weight:600;color:#a78bfa;'>{qty_m} sh</div></div>
    </div>
  </div>

  <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:10px;'>
    <div style='background:rgba(0,184,128,0.12);border:1px solid rgba(0,184,128,0.3);border-radius:8px;padding:10px;text-align:center;'>
      <div style='font-size:10px;color:#888;'>Win Probability</div>
      <div style='font-size:24px;font-weight:700;color:#00b880;'>{win_prob}%</div>
      <div style='font-size:10px;color:#666;'>Score-based estimate</div>
    </div>
    <div style='background:rgba(231,76,60,0.1);border:1px solid rgba(231,76,60,0.25);border-radius:8px;padding:10px;text-align:center;'>
      <div style='font-size:10px;color:#888;'>Max Loss if SL hits</div>
      <div style='font-size:24px;font-weight:700;color:#e74c3c;'>Rs.{max_loss_rs:,.0f}</div>
      <div style='font-size:10px;color:#666;'>SL at Rs.{stop_loss_m}</div>
    </div>
    <div style='background:rgba(0,184,128,0.1);border:1px solid rgba(0,184,128,0.25);border-radius:8px;padding:10px;text-align:center;'>
      <div style='font-size:10px;color:#888;'>Max Gain if Target hits</div>
      <div style='font-size:24px;font-weight:700;color:#00b880;'>Rs.{max_gain_rs:,.0f}</div>
      <div style='font-size:10px;color:#666;'>Target Rs.{target_m}</div>
    </div>
  </div>

  {_risk_html}

  <div style='margin-top:10px;font-size:10px;color:#555;text-align:center;'>
    Penalty: -{_pen}pts applied | Score = Tech30%+AI25%+Candle15%+Structure15%+SMC10%+Vol5%
  </div>
</div>""", unsafe_allow_html=True)

    if signal and ALERT_ON_SIGNAL:
        fire_alert(f"{direction} [{selected_mode}]", stock, price,
                   qty_m, stop_loss_m, target_m, master, mode)

    col_sig, col_pos = st.columns(2)
    with col_sig:
        st.markdown("#### Detailed Layer Checks")
        all_checks_display = {
            **tech_checks,
            f"AI Model ({ai_pct}%)": c_ai,
            "Market Structure Bullish": struct_pct >= 65,
            "Candle Pattern Bullish":   candle_pct >= 65,
            "Volume Confirmation":      vol_pct >= 55,
            "SMC / OB Zone":            smc_pct >= 65,
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
                f"AI: {'Training' if _data_count>=40 else 'Need more data'} ({_data_count} samples)"
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

    st.markdown(f"""
<div style="background:{meter_bg};border:2px solid {meter_color};border-radius:12px;padding:16px 20px;margin-bottom:14px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <div style="font-size:13px;font-weight:600;color:#ccc;">🧠 Combined AI + Technical Score</div>
    <div style="font-size:22px;font-weight:800;color:{meter_color};">{combined}%
      <span style="font-size:13px;font-weight:500;margin-left:6px;">{meter_label}</span>
    </div>
  </div>
  <div style="background:rgba(255,255,255,0.1);border-radius:99px;height:14px;margin-bottom:12px;position:relative;">
    <div style="width:{combined}%;background:{meter_color};border-radius:99px;height:14px;box-shadow:0 0 8px {meter_color}66;"></div>
    <div style="position:absolute;left:80%;top:-4px;width:2px;height:22px;background:#fff;opacity:0.4;"></div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;text-align:center;">
    <div style="background:rgba(0,0,0,0.3);border-radius:8px;padding:8px;">
      <div style="font-size:11px;color:#999;margin-bottom:3px;">🤖 AI Score</div>
      <div style="font-size:22px;font-weight:700;color:{'#27ae60' if ai_pct>=60 else ('#f39c12' if ai_pct>=40 else '#e74c3c')};">{ai_pct}%</div>
      <div style="font-size:10px;color:#888;">{'Bullish' if ai_pct>=60 else ('Neutral' if ai_pct>=40 else 'Bearish')}</div>
    </div>
    <div style="background:rgba(0,0,0,0.3);border-radius:8px;padding:8px;">
      <div style="font-size:11px;color:#999;margin-bottom:3px;">📊 Technical</div>
      <div style="font-size:22px;font-weight:700;color:{'#27ae60' if tech_pct>=75 else ('#f39c12' if tech_pct>=50 else '#e74c3c')};">{tech_pct}%</div>
      <div style="font-size:10px;color:#888;">{tech_score}/4 checks</div>
    </div>
    <div style="background:rgba(0,0,0,0.3);border-radius:8px;padding:8px;">
      <div style="font-size:11px;color:#999;margin-bottom:3px;">🎯 RSI</div>
      <div style="font-size:22px;font-weight:700;color:{'#e74c3c' if rsi_ob else ('#4e8fff' if rsi_os else '#27ae60')};">{last['RSI']:.0f}</div>
      <div style="font-size:10px;color:#888;">{'Overbought' if rsi_ob else ('Oversold' if rsi_os else 'Normal')}</div>
    </div>
  </div>
  <div style="margin-top:8px;font-size:11px;color:#888;text-align:center;">80%+ = STRONG BUY · Combined = AI×50% + Technical×50%</div>
</div>""", unsafe_allow_html=True)

    col_sig, col_pos = st.columns(2)
    with col_sig:
        st.markdown(f"#### 🎯 {selected_mode} Signal")
        checks = {
            "Trend (Price > EMA20 > EMA50)":   c_trend,
            "RSI Bullish (45–68)":              c_rsi,
            "MACD > Signal Line":               c_macd,
            "Volume Surge (> 1.1x)":            c_vol,
            f"AI Bullish ({ai_pct}% > 55%)":   c_ai,
        }
        atr_now = float(last["ATR"])
        if direction == "STRONG BUY":
            st.success(f"🚀 STRONG BUY | Combined {combined}% | AI {ai_pct}% | Tech {tech_pct}%")
            if ALERT_ON_SIGNAL:
                fire_alert(f"STRONG BUY [{selected_mode}]", stock, price,
                           max(1,int((capital*risk/100)/max(atr_now*1.5,0.01))),
                           round(price-atr_now*1.5,2), round(price+atr_now*3,2), score, mode)
        elif direction == "BUY":
            st.success(f"🟢 BUY SIGNAL | Combined {combined}% | AI {ai_pct}% | Tech {tech_pct}%")
            if ALERT_ON_SIGNAL:
                fire_alert(f"BUY SIGNAL [{selected_mode}]", stock, price,
                           max(1,int((capital*risk/100)/max(atr_now*1.5,0.01))),
                           round(price-atr_now*1.5,2), round(price+atr_now*3,2), score, mode)
        elif direction == "SELL":
            st.error(f"🔴 SELL / AVOID | Combined {combined}% | AI {ai_pct}% | Tech {tech_pct}%")
        else:
            if rsi_ob:
                st.warning(f"🟡 WAIT — RSI Overbought ({last['RSI']:.0f}) | Pull back expected")
            elif combined < 62:
                st.warning(f"🟡 WAIT — Combined {combined}% (need 62%+ for BUY)")
            else:
                st.warning(f"🟡 WAIT | Combined {combined}% | Need stronger confirmation")

        chk_df = pd.DataFrame([{"Check":k,"Pass":"✅" if v else "❌"} for k,v in checks.items()])
        st.dataframe(chk_df, hide_index=True, height=205)

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
                    fire_alert(f"BUY EXECUTED [{selected_mode}]",stock,live_px,qty,stop_loss,target_price,score,"Paper")
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
                               pos["stop_loss"],pos["target"],score,"Paper",pnl=pnl)
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
                fire_alert(f"{txn} LIVE [{selected_mode}]",stock,price,fo_qty,stop_loss,target_price,score,"Live")
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
        open_pnl = (price-pos["price"])*pos["qty"]
        color = "#2d8a4e" if open_pnl>=0 else "#c0392b"
        st.markdown(f"""<div style="background:#f0fff4;border:1px solid #2d8a4e;border-radius:6px;
padding:10px 16px;margin:8px 0;">
<b>📦 Open [{pos.get('strategy','—')}]</b><br>
{pos['stock'].replace('.NS','')} | Entry ₹{pos['price']:.2f} | Qty {pos['qty']}
| SL ₹{pos['stop_loss']} | TGT ₹{pos['target']}<br>
Unrealised: <b style="color:{color}">₹{open_pnl:+.2f}</b>
</div>""", unsafe_allow_html=True)

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
    tab_candle, tab_ta = st.tabs(["Candlestick Patterns", "Technical Analysis Summary"])

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
  <div style='font-size:20px;font-weight:700;color:{tc};'>{ms["trend"]}</div>
  <div style='font-size:12px;color:#aaa;margin-top:6px;'>
    {"HH" if ms["hh"] else "LH"} + {"HL" if ms["hl"] else "LL"}
  </div>
  {"<div style='margin-top:8px;background:#0d2818;border-radius:6px;padding:8px;font-size:12px;color:#00e5a0;'>" + ms["mss"] + "</div>" if ms.get("mss") else ""}
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
