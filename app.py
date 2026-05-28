# =============================================================
# AI Trading PRO+ v1.3 — FINAL FIXED VERSION
# Fixes applied:
#   ✅ Graph/chart not showing — fixed with data cleaning
#   ✅ Chart data info bar added
#   ✅ Extended fallback periods (6mo added)
#   ✅ plotly config for Streamlit Cloud
#   ✅ Admin panel CallMeBot uses global var not config import
#   ✅ HTML email app link fixed
#   ✅ All features from v1.3 preserved
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
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
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
# MAIN DASHBOARD
# =============================================================
st.title("📊 AI Trading PRO+ v1.3")
st.caption(f"👤 {user}  |  {datetime.datetime.now().strftime('%d %b %Y  %H:%M:%S')}")

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

    # ── AI MODEL ─────────────────────────────────────────────
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    feat_cols = ["EMA20","EMA50","RSI","MACD","Stoch_K","Vol_Ratio"]
    fd = df[feat_cols].dropna()
    td = df["Target"].loc[fd.index]
    if len(fd) >= 30:
        mdl = RandomForestClassifier(n_estimators=200, random_state=42)
        mdl.fit(fd, td)
        ai_prob = mdl.predict_proba(fd.iloc[-1:].values)[0][1]
    else:
        ai_prob = 0.5

    # ── SIGNAL + COMBINED SCORE ───────────────────────────────
    st.markdown("---")
    c_trend = last["Close"] > last["EMA20"] > last["EMA50"]
    c_rsi   = 45 < last["RSI"] < 68
    c_macd  = last["MACD"] > last["MACD_Signal"]
    c_vol   = last["Vol_Ratio"] > 1.1
    c_ai    = ai_prob > 0.55
    rsi_ob  = last["RSI"] > 75
    rsi_os  = last["RSI"] < 30
    score   = sum([c_trend, c_rsi, c_macd, c_vol, c_ai])
    tech_score = sum([c_trend, c_rsi, c_macd, c_vol])
    tech_pct   = round((tech_score / 4) * 100)
    ai_pct     = round(ai_prob * 100)
    combined   = round(tech_pct * 0.5 + ai_pct * 0.5)

    if force_trade:        direction = "STRONG BUY"
    elif rsi_ob:           direction = "WAIT"
    elif combined >= 80:   direction = "STRONG BUY"
    elif combined >= 62 and tech_score >= 3 and ai_pct >= 40: direction = "BUY"
    elif combined <= 35:   direction = "SELL"
    elif combined <= 45 and tech_score <= 1: direction = "SELL"
    else:                  direction = "WAIT"

    signal = direction in ["BUY", "STRONG BUY"]

    if combined >= 80:   meter_color="#00b880"; meter_bg="#003d2a"; meter_label="STRONG BUY ✅"
    elif combined >= 62: meter_color="#27ae60"; meter_bg="#1a3d20"; meter_label="BUY"
    elif combined >= 45: meter_color="#f39c12"; meter_bg="#3d2a00"; meter_label="WAIT"
    else:                meter_color="#e74c3c"; meter_bg="#3d0a0a"; meter_label="SELL"

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
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
                    "time":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
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
                    "time":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
                st.markdown("""
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
