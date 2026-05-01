# =============================================================
# config.py — AI Trading PRO+ v1.2
# Pre-filled for: Rishu Bhardwaj
# =============================================================

# ── Kite Connect credentials ──────────────────────────────────
API_KEY      = "6c3uhkm1yw56fd8u"
API_SECRET   = "fi60nvnc50gjyjl5rmsyq7jmrvl1yyyl"
ACCESS_TOKEN = ""   # filled automatically after Kite login

# =============================================================
# EMAIL ALERTS — Gmail SMTP
# =============================================================
# Your email is pre-filled ✓
# ONLY THING LEFT TO DO:
#   1. Open: myaccount.google.com/apppasswords
#   2. Create App Password → name it anything (e.g. "Trading")
#   3. Copy the 16-char password Google shows you
#   4. Paste it into SMTP_PASS below (no spaces)
#   5. Change EMAIL_ALERTS_ON = True

EMAIL_ALERTS_ON = True    # ✅ ENABLED

ALERT_EMAIL_TO = "bhardwaj.rishu04@gmail.com"   # ✓
SMTP_USER      = "bhardwaj.rishu04@gmail.com"   # ✓
SMTP_PASS      = "xezpjztkqqtjwnjw"             # ✅ App Password set
SMTP_SERVER    = "smtp.gmail.com"
SMTP_PORT      = 587

# =============================================================
# WHATSAPP ALERTS — CallMeBot (FREE)
# =============================================================
# Your number is pre-filled ✓
# ONLY THING LEFT TO DO:
#   1. Save +34 644 60 42 96 in contacts as "CallMeBot"
#   2. Open WhatsApp → send EXACTLY this to CallMeBot:
#          I allow callmebot to send me messages
#   3. You'll get a reply with your API key (e.g. 1234567)
#   4. Paste key into CALLMEBOT_APIKEY below
#   5. Change CALLMEBOT_ALERTS_ON = True

CALLMEBOT_ALERTS_ON = False   # ← change to True after step above

CALLMEBOT_PHONE  = "+919805184822"   # ✓ pre-filled
CALLMEBOT_APIKEY = ""                # ← PASTE CALLMEBOT API KEY HERE

# =============================================================
# WHATSAPP ALERTS — Twilio (optional, skip for now)
# =============================================================
TWILIO_ALERTS_ON = False
TWILIO_SID       = ""
TWILIO_TOKEN     = ""
TWILIO_FROM      = "whatsapp:+14155238886"
TWILIO_TO        = "whatsapp:+919805184822"   # ✓ pre-filled

# =============================================================
# ALERT BEHAVIOUR — no changes needed
# =============================================================
ALERT_ON_SIGNAL    = True    # alert when BUY signal detected
ALERT_ON_EXECUTION = True    # alert when BUY/SELL order placed
ALERT_MIN_SCORE    = 3       # minimum score to trigger alert
ALERT_COOLDOWN_MIN = 15      # minutes gap between repeat alerts
