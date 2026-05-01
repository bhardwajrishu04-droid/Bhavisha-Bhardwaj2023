# =============================================================
# config.py — AI Trading PRO+ v1.3
# Pre-filled for: Rishu Bhardwaj (Deepak Bhardwaj)
# =============================================================

# ── Kite Connect credentials ──────────────────────────────────
API_KEY      = "6c3uhkm1yw56fd8u"
API_SECRET   = "fi60nvnc50gjyjl5rmsyq7jmrvl1yyyl"
ACCESS_TOKEN = ""   # filled automatically after Kite login

# =============================================================
# EMAIL ALERTS — Gmail SMTP
# =============================================================
EMAIL_ALERTS_ON = True    # ✅ ENABLED

ALERT_EMAIL_TO = "bhardwaj.rishu04@gmail.com"
SMTP_USER      = "bhardwaj.rishu04@gmail.com"
SMTP_PASS      = "xezpjztkqqtjwnjw"    # ✅ App Password set
SMTP_SERVER    = "smtp.gmail.com"
SMTP_PORT      = 587

# =============================================================
# WHATSAPP ALERTS — CallMeBot (FREE)
# =============================================================
# TO ENABLE:
#   1. Save +34 644 60 42 96 in contacts as "CallMeBot"
#   2. Send this WhatsApp message to that number:
#          I allow callmebot to send me messages
#   3. You'll get API key in reply — paste below
#   4. Change CALLMEBOT_ALERTS_ON = True

CALLMEBOT_ALERTS_ON = False
CALLMEBOT_PHONE     = "+919805184822"
CALLMEBOT_APIKEY    = ""   # ← paste your CallMeBot API key here

# =============================================================
# WHATSAPP ALERTS — Twilio (optional)
# =============================================================
TWILIO_ALERTS_ON = False
TWILIO_SID       = ""
TWILIO_TOKEN     = ""
TWILIO_FROM      = "whatsapp:+14155238886"
TWILIO_TO        = "whatsapp:+919805184822"

# =============================================================
# ALERT BEHAVIOUR
# =============================================================
ALERT_ON_SIGNAL    = True    # alert when BUY signal detected
ALERT_ON_EXECUTION = True    # alert when BUY/SELL order placed
ALERT_MIN_SCORE    = 3       # minimum score to trigger alert
ALERT_COOLDOWN_MIN = 15      # minutes gap between repeat alerts

# =============================================================
# APP PUBLIC URL — ✅ UPDATED
# =============================================================
APP_URL = "https://bhavisha-ai-trading-pro.streamlit.app"
