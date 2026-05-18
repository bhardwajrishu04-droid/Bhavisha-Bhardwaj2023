# =============================================================
# alerts.py — AI Trading PRO+ v1.3
# FIXED: Works on Streamlit Cloud (Secrets) + local (config.py)
# Email + WhatsApp (CallMeBot + Twilio)
# =============================================================

import smtplib
import requests
import datetime
import urllib.parse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── Safe config reader — Streamlit Secrets first, then config.py ──
def _get(key, default=""):
    # Try Streamlit Secrets (Streamlit Cloud)
    try:
        import streamlit as st
        val = st.secrets.get(key, None)
        if val is not None:
            return val
    except Exception:
        pass
    # Try local config.py (local PC)
    try:
        import config as _cfg
        return getattr(_cfg, key, default)
    except Exception:
        pass
    return default

# Load all config values safely
EMAIL_ALERTS_ON    = _get("EMAIL_ALERTS_ON",    False)
ALERT_EMAIL_TO     = _get("ALERT_EMAIL_TO",     "")
SMTP_SERVER        = _get("SMTP_SERVER",        "smtp.gmail.com")
SMTP_PORT          = _get("SMTP_PORT",          587)
SMTP_USER          = _get("SMTP_USER",          "")
SMTP_PASS          = _get("SMTP_PASS",          "")

CALLMEBOT_ALERTS_ON = _get("CALLMEBOT_ALERTS_ON", False)
CALLMEBOT_PHONE     = _get("CALLMEBOT_PHONE",     "")
CALLMEBOT_APIKEY    = _get("CALLMEBOT_APIKEY",    "")

TWILIO_ALERTS_ON = _get("TWILIO_ALERTS_ON", False)
TWILIO_SID       = _get("TWILIO_SID",       "")
TWILIO_TOKEN     = _get("TWILIO_TOKEN",     "")
TWILIO_FROM      = _get("TWILIO_FROM",      "whatsapp:+14155238886")
TWILIO_TO        = _get("TWILIO_TO",        "")


# =============================================================
# MAIN DISPATCHER
# =============================================================
def send_alert(action, stock, price, qty, stop_loss,
               target, score, mode="Paper", pnl=None):
    subject = f"[AI Trading PRO] {action} - {stock} @ Rs.{price:.2f}"
    body    = _build_body(action, stock, price, qty,
                          stop_loss, target, score, mode, pnl)
    results = []

    if EMAIL_ALERTS_ON:
        ok, err = _send_email(subject, body)
        results.append(f"Email: {'Sent' if ok else 'Failed - ' + err}")

    if CALLMEBOT_ALERTS_ON:
        ok, err = _send_callmebot(body)
        results.append(f"WhatsApp: {'Sent' if ok else 'Failed - ' + err}")

    if TWILIO_ALERTS_ON:
        ok, err = _send_twilio(body)
        results.append(f"Twilio WA: {'Sent' if ok else 'Failed - ' + err}")

    return results


# =============================================================
# MESSAGE BUILDER
# =============================================================
def _build_body(action, stock, price, qty, stop_loss,
                target, score, mode, pnl):
    now = datetime.datetime.now().strftime("%d %b %Y  %H:%M:%S")
    try:
        rr = round((target - price) / max(abs(price - stop_loss), 0.01), 2)
    except Exception:
        rr = 0

    pnl_line = ""
    if pnl is not None:
        p_sign = "+" if pnl >= 0 else ""
        pnl_line = "\nP&L         : Rs." + p_sign + f"{pnl:.2f}"

    direction = "BUY" if "BUY" in str(action).upper() else "SELL"

    lines = [
        "AI Trading PRO+ Alert",
        "─" * 30,
        "Action      : " + str(action),
        "Stock       : " + str(stock),
        "Price       : Rs." + f"{price:.2f}",
        "Qty         : " + str(qty) + " shares",
        "Mode        : " + str(mode),
        "─" * 30,
        "Stop Loss   : Rs." + f"{stop_loss:.2f}",
        "Target      : Rs." + f"{target:.2f}",
        "R:R Ratio   : " + str(rr) + " : 1",
        "Score       : " + str(score) + "/5",
    ]
    if pnl_line:
        lines.append(pnl_line)
    lines += ["─" * 30, "Time        : " + now]
    return "\n".join(lines)


# =============================================================
# EMAIL — Gmail SMTP
# =============================================================
def _send_email(subject, body):
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_USER
        msg["To"]      = ALERT_EMAIL_TO
        msg.attach(MIMEText(body, "plain"))
        msg.attach(MIMEText(_html_email(body), "html"))

        with smtplib.SMTP(SMTP_SERVER, int(SMTP_PORT), timeout=10) as srv:
            srv.ehlo()
            srv.starttls()
            srv.login(SMTP_USER, SMTP_PASS)
            srv.sendmail(SMTP_USER, ALERT_EMAIL_TO, msg.as_string())
        return True, ""
    except smtplib.SMTPAuthenticationError:
        return False, "Gmail auth failed - check App Password"
    except Exception as e:
        return False, str(e)


def _html_email(body):
    rows = ""
    for line in body.split("\n"):
        if "─" in line:
            rows += "<tr><td colspan='2'><hr style='border:0;border-top:1px solid #e8e8e8;margin:4px 0'></td></tr>"
        elif ":" in line:
            k, _, v = line.partition(":")
            color = "#27ae60" if "BUY" in v else ("#e74c3c" if "SELL" in v else "#1a1a2e")
            rows += (f"<tr><td style='padding:5px 14px;color:#888;font-size:13px'>{k.strip()}</td>"
                     f"<td style='padding:5px 14px;color:{color};font-size:13px;font-weight:500'>{v.strip()}</td></tr>")
        else:
            rows += f"<tr><td colspan='2' style='padding:8px 14px;font-weight:700;font-size:15px;color:#1a1a2e'>{line}</td></tr>"

    return f"""<html><body>
<div style='font-family:Arial,sans-serif;max-width:460px;margin:24px auto;
border:1px solid #e0e0e0;border-radius:10px;overflow:hidden;'>
<div style='background:#0a0c10;padding:14px 20px;'>
<span style='color:#00e5a0;font-size:17px;font-weight:700;'>AI Trading PRO+</span>
</div>
<table width='100%' cellpadding='0' cellspacing='0'
style='background:#fff;padding:8px 0;'>{rows}</table>
<div style='background:#f8f9fa;padding:10px 20px;font-size:11px;color:#aaa;text-align:center;'>
AI Trading PRO+ - Automated alert - Do not reply
</div></div></body></html>"""


# =============================================================
# WHATSAPP — CallMeBot (FREE)
# =============================================================
def _send_callmebot(message):
    if not CALLMEBOT_PHONE or not CALLMEBOT_APIKEY:
        return False, "Phone or API key not set"
    try:
        encoded = urllib.parse.quote(message)
        url = (f"https://api.callmebot.com/whatsapp.php"
               f"?phone={CALLMEBOT_PHONE}&text={encoded}&apikey={CALLMEBOT_APIKEY}")
        r = requests.get(url, timeout=12)
        if r.status_code == 200 and ("Message queued" in r.text or "OK" in r.text):
            return True, ""
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


# =============================================================
# WHATSAPP — Twilio
# =============================================================
def _send_twilio(message):
    if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, TWILIO_TO]):
        return False, "Twilio credentials not set"
    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(body=message, from_=TWILIO_FROM, to=TWILIO_TO)
        return (True, "") if msg.sid else (False, "No SID")
    except ImportError:
        return False, "pip install twilio"
    except Exception as e:
        return False, str(e)
