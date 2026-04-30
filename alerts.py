# =============================================================
# alerts.py — AI Trading PRO+ v1.2
# Email + WhatsApp alert system
#
# TWO WhatsApp methods (pick one or both):
#   A) CallMeBot — FREE, no account needed, setup in 2 min
#   B) Twilio    — paid, enterprise-grade, very reliable
#
# Email method:
#   Gmail SMTP with App Password — free, works instantly
# =============================================================

import smtplib
import requests
import datetime
import urllib.parse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    from config import (
        EMAIL_ALERTS_ON, ALERT_EMAIL_TO,
        SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASS,
        CALLMEBOT_ALERTS_ON, CALLMEBOT_PHONE, CALLMEBOT_APIKEY,
        TWILIO_ALERTS_ON, TWILIO_SID, TWILIO_TOKEN,
        TWILIO_FROM, TWILIO_TO,
    )
except ImportError as e:
    raise ImportError(f"alerts.py: missing config key → {e}")


# =============================================================
# MAIN DISPATCHER — called from app.py
# =============================================================
def send_alert(action, stock, price, qty, stop_loss,
               target, score, mode="Paper", pnl=None):
    """
    action : "BUY SIGNAL" | "SELL SIGNAL" |
             "BUY EXECUTED" | "SELL EXECUTED"
    Returns list of result strings for st.toast()
    """
    subject = f"[AI Trading PRO] {action} — {stock} @ ₹{price:.2f}"
    body    = _build_body(action, stock, price, qty,
                          stop_loss, target, score, mode, pnl)
    results = []

    if EMAIL_ALERTS_ON:
        ok, err = _send_email(subject, body)
        results.append(f"📧 Email: {'✅ Sent' if ok else '❌ ' + err}")

    if CALLMEBOT_ALERTS_ON:
        ok, err = _send_callmebot(body)
        results.append(f"📱 WhatsApp (CallMeBot): {'✅ Sent' if ok else '❌ ' + err}")

    if TWILIO_ALERTS_ON:
        ok, err = _send_twilio(body)
        results.append(f"📱 WhatsApp (Twilio): {'✅ Sent' if ok else '❌ ' + err}")

    return results  # show with st.toast() in app.py


# =============================================================
# MESSAGE BUILDER
# =============================================================
def _build_body(action, stock, price, qty, stop_loss,
                target, score, mode, pnl):
    now  = datetime.datetime.now().strftime("%d %b %Y  %H:%M:%S")
    rr   = round((target - price) / max(price - stop_loss, 0.01), 2)
    emoji = "🟢" if "BUY" in action else "🔴"
    pnl_line = ""
    if pnl is not None:
        p_emoji = "🟢" if pnl >= 0 else "🔴"
        pnl_line = f"\nP&L         : {p_emoji} ₹{pnl:+.2f}"

    return (
        f"{emoji} *AI Trading PRO+ Alert*\n"
        f"{'─'*32}\n"
        f"Action      : {action}\n"
        f"Stock       : {stock}\n"
        f"Price       : ₹{price:.2f}\n"
        f"Qty         : {qty} shares\n"
        f"Mode        : {mode}\n"
        f"{'─'*32}\n"
        f"Stop Loss   : ₹{stop_loss:.2f}\n"
        f"Target      : ₹{target:.2f}\n"
        f"R:R Ratio   : {rr} : 1\n"
        f"Signal Score: {score}/4"
        f"{pnl_line}\n"
        f"{'─'*32}\n"
        f"Time        : {now}"
    )


# =============================================================
# 1. EMAIL — Gmail SMTP
# =============================================================
# SETUP (one-time, 3 minutes):
#   1. Google Account → Security → 2-Step Verification → ON
#   2. Google Account → Security → App Passwords
#      → Select app: Mail  → Select device: Windows PC → Generate
#   3. Copy the 16-character password into config.py as SMTP_PASS
#   4. Set SMTP_USER = your Gmail address (e.g. yourname@gmail.com)
#   5. Set ALERT_EMAIL_TO = where you want alerts sent
# =============================================================
def _send_email(subject, body):
    try:
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_USER
        msg["To"]      = ALERT_EMAIL_TO

        msg.attach(MIMEText(body.replace("*", ""), "plain"))
        msg.attach(MIMEText(_html_email(body), "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as srv:
            srv.ehlo()
            srv.starttls()
            srv.login(SMTP_USER, SMTP_PASS)
            srv.sendmail(SMTP_USER, ALERT_EMAIL_TO, msg.as_string())
        return True, ""
    except smtplib.SMTPAuthenticationError:
        return False, "Auth failed — check Gmail App Password in config.py"
    except Exception as e:
        return False, str(e)


def _html_email(body):
    rows = ""
    for line in body.split("\n"):
        clean = line.replace("*", "")
        if "─" in clean:
            rows += "<tr><td colspan='2'><hr style='border:0;border-top:1px solid #e8e8e8;margin:4px 0'></td></tr>"
        elif ":" in clean:
            k, _, v = clean.partition(":")
            color = "#27ae60" if "BUY" in v or "🟢" in v else ("#e74c3c" if "SELL" in v or "🔴" in v else "#1a1a2e")
            rows += (
                f"<tr>"
                f"<td style='padding:5px 14px;color:#888;font-size:13px;white-space:nowrap'>{k.strip()}</td>"
                f"<td style='padding:5px 14px;color:{color};font-size:13px;font-weight:500'>{v.strip()}</td>"
                f"</tr>"
            )
        else:
            rows += f"<tr><td colspan='2' style='padding:8px 14px;font-weight:700;font-size:15px;color:#1a1a2e'>{clean}</td></tr>"

    return f"""<html><body>
<div style='font-family:Inter,Arial,sans-serif;max-width:460px;margin:24px auto;
            border:1px solid #e0e0e0;border-radius:10px;overflow:hidden;'>
  <div style='background:#0a0c10;padding:14px 20px;display:flex;align-items:center;gap:10px'>
    <span style='color:#00e5a0;font-size:17px;font-weight:700;letter-spacing:.05em'>📈 AI Trading PRO+</span>
  </div>
  <table width='100%' cellpadding='0' cellspacing='0'
         style='background:#fff;padding:8px 0'>{rows}</table>
  <div style='background:#f8f9fa;padding:10px 20px;font-size:11px;
              color:#aaa;text-align:center;border-top:1px solid #f0f0f0'>
    AI Trading PRO+ · Automated alert · Do not reply to this email
  </div>
</div></body></html>"""


# =============================================================
# 2. WHATSAPP — CallMeBot (FREE)
# =============================================================
# SETUP (one-time, 2 minutes):
#   1. Save this number in your phone contacts:
#        +34 644 60 42 96   (name it "CallMeBot")
#   2. Open WhatsApp and send EXACTLY this message to that number:
#        I allow callmebot to send me messages
#   3. Within 1 minute you'll receive a reply with your API key
#      e.g.  "Your API key is: 1234567"
#   4. In config.py:
#        CALLMEBOT_PHONE  = "+91XXXXXXXXXX"   ← your WhatsApp number with country code
#        CALLMEBOT_APIKEY = "1234567"          ← key from the reply
#   5. Set CALLMEBOT_ALERTS_ON = True
#
#   Docs: https://www.callmebot.com/blog/free-api-whatsapp-messages/
# =============================================================
def _send_callmebot(message):
    if not CALLMEBOT_PHONE or not CALLMEBOT_APIKEY:
        return False, "CALLMEBOT_PHONE / CALLMEBOT_APIKEY not set in config.py"
    try:
        encoded = urllib.parse.quote(message)
        url = (
            f"https://api.callmebot.com/whatsapp.php"
            f"?phone={CALLMEBOT_PHONE}"
            f"&text={encoded}"
            f"&apikey={CALLMEBOT_APIKEY}"
        )
        r = requests.get(url, timeout=12)
        if r.status_code == 200 and ("Message queued" in r.text or "OK" in r.text):
            return True, ""
        return False, f"HTTP {r.status_code}: {r.text[:100]}"
    except requests.exceptions.Timeout:
        return False, "Request timed out — check internet connection"
    except Exception as e:
        return False, str(e)


# =============================================================
# 3. WHATSAPP — Twilio (paid, enterprise-grade)
# =============================================================
# SETUP:
#   1. Sign up at https://www.twilio.com (free trial gives ~$15 credit)
#   2. Go to: console.twilio.com → Messaging → Try it out → Send a WhatsApp message
#   3. From YOUR phone, send "join <sandbox-word>" to +1 415 523 8886
#   4. In config.py:
#        TWILIO_SID   = "ACxxxxxxxxxxxxxxxxx"     ← Account SID from console
#        TWILIO_TOKEN = "your_auth_token"          ← Auth Token from console
#        TWILIO_FROM  = "whatsapp:+14155238886"    ← Twilio sandbox number
#        TWILIO_TO    = "whatsapp:+91XXXXXXXXXX"   ← your number with country code
#   5. Set TWILIO_ALERTS_ON = True
# =============================================================
def _send_twilio(message):
    if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, TWILIO_TO]):
        return False, "Twilio credentials not set in config.py"
    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=TWILIO_FROM,
            to=TWILIO_TO
        )
        return (True, "") if msg.sid else (False, "No SID returned")
    except ImportError:
        return False, "Twilio not installed — run: pip install twilio"
    except Exception as e:
        return False, str(e)
