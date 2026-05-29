# modules/alerts.py
# =============================================================
# Alerts module — Email (Gmail SMTP) + WhatsApp (CallMeBot)
# =============================================================
import smtplib, requests, urllib.parse, datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from modules import config


def send(action: str, stock: str, price: float, qty: int,
         stop_loss: float, target: float, score: float,
         mode: str = "Paper", pnl: float = None) -> list[str]:
    """Send alert via all configured channels."""
    now  = datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")
    subj = f"[AI Trading] {action} — {stock} @ Rs.{price:.2f}"
    body = _build_body(action, stock, price, qty, stop_loss,
                       target, score, mode, pnl, now)
    results = []
    if config.EMAIL_ON:
        ok, err = _send_email(subj, body)
        results.append(f"Email: {'Sent' if ok else 'Failed: '+err}")
    if config.WA_ON:
        ok, err = _send_wa(body)
        results.append(f"WhatsApp: {'Sent' if ok else 'Failed: '+err}")
    return results


def _build_body(action, stock, price, qty, sl, tgt, score, mode, pnl, now):
    try:    rr = round((tgt-price)/max(abs(price-sl),0.01),2)
    except: rr = 0
    lines = [
        "AI Trading PRO+ Alert",
        "-"*28,
        f"Action    : {action}",
        f"Stock     : {stock}",
        f"Price     : Rs.{price:.2f}",
        f"Qty       : {qty} shares",
        f"Mode      : {mode}",
        "-"*28,
        f"Stop Loss : Rs.{sl:.2f}",
        f"Target    : Rs.{tgt:.2f}",
        f"R:R Ratio : {rr}:1",
        f"Score     : {score}",
    ]
    if pnl is not None:
        lines.append(f"P&L       : Rs.{pnl:+.2f}")
    lines += ["-"*28, f"Time      : {now}"]
    return "\n".join(lines)


def _send_email(subject: str, body: str):
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = config.SMTP_USER
        msg["To"]      = config.EMAIL_TO
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(config.SMTP_HOST, int(config.SMTP_PORT),
                          timeout=10) as srv:
            srv.ehlo(); srv.starttls()
            srv.login(config.SMTP_USER, config.SMTP_PASS)
            srv.sendmail(config.SMTP_USER, config.EMAIL_TO, msg.as_string())
        return True, ""
    except smtplib.SMTPAuthenticationError:
        return False, "Gmail auth failed"
    except Exception as e:
        return False, str(e)[:60]


def _send_wa(message: str):
    if not config.WA_PHONE or not config.WA_KEY:
        return False, "Phone or API key not set"
    try:
        enc = urllib.parse.quote(message)
        url = (f"https://api.callmebot.com/whatsapp.php"
               f"?phone={config.WA_PHONE}&text={enc}&apikey={config.WA_KEY}")
        r = requests.get(url, timeout=12)
        if r.status_code == 200 and ("queued" in r.text.lower() or
                                      "ok" in r.text.lower()):
            return True, ""
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)[:60]
