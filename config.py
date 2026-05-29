# modules/config.py
# =============================================================
# Central configuration — secrets, constants, stock universe
# =============================================================
import streamlit as st

def get(key: str, default=""):
    try: return st.secrets[key]
    except Exception: pass
    try:
        import config as _c; return getattr(_c, key, default)
    except Exception: pass
    return default

# Kite
API_KEY    = get("API_KEY",    "")
API_SECRET = get("API_SECRET", "")

# Alerts
EMAIL_ON    = get("EMAIL_ALERTS_ON",    False)
EMAIL_TO    = get("ALERT_EMAIL_TO",     "")
SMTP_USER   = get("SMTP_USER",          "")
SMTP_PASS   = get("SMTP_PASS",          "")
SMTP_HOST   = get("SMTP_SERVER",        "smtp.gmail.com")
SMTP_PORT   = get("SMTP_PORT",          587)
WA_ON       = get("CALLMEBOT_ALERTS_ON",False)
WA_PHONE    = get("CALLMEBOT_PHONE",    "")
WA_KEY      = get("CALLMEBOT_APIKEY",   "")
APP_URL     = get("APP_URL", "https://bhavisha-ai-trading-pro.streamlit.app")

# Alert thresholds
ALERT_MIN_SCORE    = get("ALERT_MIN_SCORE",    3)
ALERT_COOLDOWN_MIN = get("ALERT_COOLDOWN_MIN", 15)
ALERT_ON_SIGNAL    = get("ALERT_ON_SIGNAL",    True)
ALERT_ON_EXEC      = get("ALERT_ON_EXECUTION", True)

# Stock universe
STOCKS = {
    "⭐ Nifty 50 Top 20": [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS",
        "HINDUNILVR.NS","SBIN.NS","BAJFINANCE.NS","BHARTIARTL.NS","KOTAKBANK.NS",
        "WIPRO.NS","AXISBANK.NS","LTIM.NS","HCLTECH.NS","ASIANPAINT.NS",
        "MARUTI.NS","TITAN.NS","SUNPHARMA.NS","ULTRACEMCO.NS","NESTLEIND.NS",
    ],
    "🏦 Bank Nifty": [
        "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS",
        "BANKBARODA.NS","CANBK.NS","INDUSINDBK.NS","FEDERALBNK.NS","IDFCFIRSTB.NS",
    ],
    "💻 IT Sector": [
        "TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","LTIM.NS",
        "TECHM.NS","MPHASIS.NS","COFORGE.NS","PERSISTENT.NS","OFSS.NS",
    ],
    "🚗 Auto Sector": [
        "MARUTI.NS","TATAMOTORS.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","EICHERMOT.NS",
        "ASHOKLEY.NS","TVSMOTOR.NS","BALKRISIND.NS","BOSCHLTD.NS","MOTHERSON.NS",
    ],
    "🛒 FMCG Sector": [
        "HINDUNILVR.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS","MARICO.NS",
        "COLPAL.NS","GODREJCP.NS","ITC.NS","TATACONSUM.NS","EMAMILTD.NS",
    ],
    "💊 Pharma Sector": [
        "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","BIOCON.NS",
        "AUROPHARMA.NS","TORNTPHARM.NS","LUPIN.NS","IPCALAB.NS","ALKEM.NS",
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
    "📈 Intraday": {"period":"1d", "interval":"5m",  "sl_mult":1.0,"rr":1.5,"product":"MIS","color":"#4e8fff"},
    "🌊 Swing":    {"period":"1mo","interval":"1d",  "sl_mult":2.0,"rr":3.0,"product":"CNC","color":"#a78bfa"},
    "📊 Futures":  {"period":"5d", "interval":"15m", "sl_mult":1.5,"rr":2.0,"product":"NRML","color":"#ffa94d"},
    "🎯 Options":  {"period":"5d", "interval":"15m", "sl_mult":1.5,"rr":2.5,"product":"NRML","color":"#00e5a0"},
}
