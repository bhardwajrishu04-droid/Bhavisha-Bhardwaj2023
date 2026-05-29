# =============================================================
# config.py — AI Trading PRO+ v1.3
# LOCAL PC ke liye — Streamlit Cloud pe Secrets use karo
# WARNING: Apni real keys yahan mat daalo GitHub pe!
# =============================================================

# ── Kite Connect ──────────────────────────────────────────────
API_KEY      = "YOUR_KITE_API_KEY_HERE"
API_SECRET   = "YOUR_KITE_API_SECRET_HERE"
ACCESS_TOKEN = ""   # har din generate hota hai

# =============================================================
# EMAIL ALERTS
# =============================================================
EMAIL_ALERTS_ON = False

ALERT_EMAIL_TO = "your@email.com"
SMTP_USER      = "your@gmail.com"
SMTP_PASS      = "your_gmail_app_password"
SMTP_SERVER    = "smtp.gmail.com"
SMTP_PORT      = 587

# =============================================================
# WHATSAPP — CallMeBot (FREE)
# =============================================================
CALLMEBOT_ALERTS_ON = False
CALLMEBOT_PHONE     = "+91XXXXXXXXXX"
CALLMEBOT_APIKEY    = ""

# =============================================================
# WHATSAPP — Twilio (optional)
# =============================================================
TWILIO_ALERTS_ON = False
TWILIO_SID       = ""
TWILIO_TOKEN     = ""
TWILIO_FROM      = "whatsapp:+14155238886"
TWILIO_TO        = "whatsapp:+91XXXXXXXXXX"

# =============================================================
# ALERT BEHAVIOUR
# =============================================================
ALERT_ON_SIGNAL    = True
ALERT_ON_EXECUTION = True
ALERT_MIN_SCORE    = 3
ALERT_COOLDOWN_MIN = 15

# =============================================================
# APP PUBLIC URL
# =============================================================
APP_URL = "https://bhavisha-ai-trading-pro.streamlit.app"
