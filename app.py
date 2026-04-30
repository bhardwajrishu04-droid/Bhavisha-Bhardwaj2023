# =========================
# IMPORTS
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

# STATUS
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
# AUTO CREATE ADMIN (IMPORTANT)
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

    # LOGIN
    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if u in users and users[u]["password"] == p:

                # CHECK STATUS
                if users[u].get("status") != "active":
                    st.error("❌ Not Approved by Admin")
                    st.stop()

                # CHECK EXPIRY
                exp = users[u].get("expiry", "2000-01-01")
                if datetime.date.today() > datetime.datetime.strptime(exp, "%Y-%m-%d").date():
                    st.error("❌ Subscription Expired")
                    st.stop()

                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid Login")

    # SIGNUP
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

# LOGOUT BUTTON
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
    active = [u for u in users if users[u].get("status") == "active"]

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
            users[u]["expiry"] = str(datetime.date.today() + datetime.timedelta(days=30))
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

mode=st.radio("Mode",["Paper","Live"])
capital=st.number_input("Capital",10000)
risk=st.number_input("Risk %",1.5)

if mode=="Live":
    st.warning("⚠️ LIVE TRADING ENABLED (REAL MONEY)")

# =========================
# STOCK LIST
# =========================
stocks=["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS"]

# =========================
# SCANNER
# =========================
scan=[]
for s in stocks:
    try:
        d=yf.Ticker(s).history(period="5d",interval="15m")
        if d.empty: continue

        d["EMA20"]=d["Close"].ewm(span=20).mean()
        d["EMA50"]=d["Close"].ewm(span=50).mean()

        last=d.iloc[-1]
        score=2 if last["Close"]>last["EMA20"]>last["EMA50"] else 0

        scan.append({"Stock":s,"Score":score,"Price":last["Close"]})
    except:
        pass

df_scan=pd.DataFrame(scan)
st.dataframe(df_scan)

best = stocks[0]
if not df_scan.empty:
    best=df_scan.sort_values("Score",ascending=False).iloc[0]["Stock"]

stock=st.selectbox("Stock",stocks,index=stocks.index(best))

# =========================
# DATA
# =========================
df=yf.Ticker(stock).history(period="5d",interval="5m")

if df is None or df.empty:
    st.error("No Market Data")
    st.stop()

price=float(df["Close"].iloc[-1])

# =========================
# INDICATORS
# =========================
df["EMA20"]=df["Close"].ewm(span=20).mean()
df["EMA50"]=df["Close"].ewm(span=50).mean()

delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

df["EMA12"]=df["Close"].ewm(span=12).mean()
df["EMA26"]=df["Close"].ewm(span=26).mean()
df["MACD"]=df["EMA12"]-df["EMA26"]

df["ATR"]=(df["High"]-df["Low"]).rolling(14).mean()

# =========================
# AI MODEL
# =========================
df["Target"]=(df["Close"].shift(-1)>df["Close"]).astype(int)

features=df[["EMA20","EMA50","RSI","MACD"]].dropna()
target=df["Target"].loc[features.index]

model=RandomForestClassifier(n_estimators=200)
model.fit(features,target)

pred=model.predict(features.iloc[-1:].values)[0]

# =========================
# SIGNAL ENGINE
# =========================
force_trade = st.checkbox("🔥 FORCE TRADE (TEST MODE)", value=False)

trend = price > df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]
rsi_ok = df["RSI"].iloc[-1] > 50
macd_ok = df["MACD"].iloc[-1] > 0

score = sum([trend, rsi_ok, macd_ok, pred])

signal = True if force_trade else score >= 3

if signal:
    st.success(f"🟢 BUY SIGNAL | Score {score}/4")
else:
    st.warning(f"🟡 NO TRADE | Score {score}/4")

# =========================
# TRADE ENGINE
# =========================
if "pos" not in st.session_state:
    st.session_state.pos=None

atr=float(df["ATR"].iloc[-1])
qty=max(1,int((capital*(risk/100))/max(atr,0.01)))

def log_trade(action):
    st.session_state.trade_log.append({
        "time":datetime.datetime.now(),
        "stock":stock,
        "action":action,
        "price":price
    })

def order(txn):
    if not kite_ok():
        st.error("❌ Kite Not Connected")
        return

    try:
        kite.place_order(
            variety="regular",
            exchange="NSE",
            tradingsymbol=stock.replace(".NS",""),
            transaction_type=txn,
            quantity=qty,
            order_type="MARKET",
            product="CNC"
        )
        st.success(f"{txn} Order Placed")
    except Exception as e:
        st.error(e)
# =========================
# 🔘 MANUAL BUTTONS
# =========================
col1, col2 = st.columns(2)

if col1.button("🚀 BUY NOW"):
    if mode=="Live" and kite_ok():
        order("BUY")

if col2.button("🛑 SELL NOW"):
    if mode=="Live" and kite_ok():
        order("SELL")

# =========================
# AUTO LOOP
# =========================
if auto_trade:
    if mode=="Live" and kite_ok() and signal and can_trade():
        order("BUY")
        st.session_state.last_trade=datetime.datetime.now()
    time.sleep(interval)
    st.rerun()

# =========================
# LOGS
# =========================
st.subheader("Trade Logs")
st.dataframe(pd.DataFrame(st.session_state.trade_log))