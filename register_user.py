# =============================================================
# register_user.py — AI Trading PRO+ Quick Registration Tool
# Admin use karta hai jab user payment complete kar leta hai
#
# Usage:
#   python register_user.py
#   (interactive prompts chalenge)
#
# Ya direct:
#   python register_user.py Bhavisha2023 pass123 monthly
# =============================================================

import json, os, sys, datetime, smtplib, requests, urllib.parse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

DB = "users.json"

# Plan durations
PLANS = {
    "monthly":   {"days": 30,  "label": "Monthly Plan (30 days)",   "price": "₹499"},
    "quarterly": {"days": 90,  "label": "Quarterly Plan (90 days)", "price": "₹999"},
    "annual":    {"days": 365, "label": "Annual Plan (365 days)",    "price": "₹2,999"},
}

# Load config for alerts
try:
    from config import (
        SMTP_USER, SMTP_PASS, SMTP_SERVER, SMTP_PORT, EMAIL_ALERTS_ON,
        CALLMEBOT_PHONE, CALLMEBOT_APIKEY, CALLMEBOT_ALERTS_ON,
    )
except Exception as e:
    print(f"⚠️ Config import warning: {e}")
    EMAIL_ALERTS_ON = False
    CALLMEBOT_ALERTS_ON = False


def load_users():
    if not os.path.exists(DB):
        json.dump({}, open(DB, "w"))
    try:
        return json.load(open(DB))
    except:
        return {}


def save_users(users):
    json.dump(users, open(DB, "w"), indent=2)


def send_welcome_email(email, username, password, plan_label, expiry):
    """Send login credentials to new user via email."""
    if not EMAIL_ALERTS_ON or not email:
        return False, "Email alerts OFF"
    try:
        subject = "🎉 AI Trading PRO+ — Your Account is Active!"
        body_html = f"""
        <html><body>
        <div style='font-family:Inter,Arial,sans-serif;max-width:480px;margin:20px auto;
                    border:1px solid #e0e0e0;border-radius:10px;overflow:hidden;'>
          <div style='background:#0a0c10;padding:16px 24px;'>
            <span style='color:#00e5a0;font-size:18px;font-weight:700;'>📈 AI Trading PRO+</span>
          </div>
          <div style='padding:24px;'>
            <h2 style='color:#1a1a2e;margin-bottom:8px;'>Welcome! Your account is ready 🎉</h2>
            <p style='color:#555;font-size:14px;margin-bottom:20px;'>
              Your payment has been verified and your account is now active.
            </p>
            <div style='background:#f8f9fa;border-radius:8px;padding:16px;margin-bottom:20px;'>
              <table style='width:100%;font-size:14px;'>
                <tr><td style='color:#888;padding:5px 0;'>🔑 Username</td>
                    <td style='font-weight:600;color:#1a1a2e;'>{username}</td></tr>
                <tr><td style='color:#888;padding:5px 0;'>🔒 Password</td>
                    <td style='font-weight:600;color:#1a1a2e;'>{password}</td></tr>
                <tr><td style='color:#888;padding:5px 0;'>📦 Plan</td>
                    <td style='font-weight:600;color:#1a1a2e;'>{plan_label}</td></tr>
                <tr><td style='color:#888;padding:5px 0;'>📅 Valid Until</td>
                    <td style='font-weight:600;color:#27ae60;'>{expiry}</td></tr>
              </table>
            </div>
            <div style='background:#003d2a;border:1px solid #00b880;border-radius:8px;
                        padding:14px;margin-bottom:20px;'>
              <p style='color:#00e5a0;font-size:14px;font-weight:600;margin-bottom:6px;'>
                📱 How to access the app:
              </p>
              <p style='color:#aaa;font-size:13px;margin:0;'>
                Contact your admin on WhatsApp for the app link.<br>
                Use the username and password above to login.
              </p>
            </div>
            <p style='color:#888;font-size:12px;'>
              Questions? WhatsApp us at <strong>+91 98051 84822</strong>
            </p>
          </div>
        </div>
        </body></html>"""

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_USER
        msg["To"]      = email
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as srv:
            srv.ehlo(); srv.starttls()
            srv.login(SMTP_USER, SMTP_PASS)
            srv.sendmail(SMTP_USER, email, msg.as_string())
        return True, "Sent"
    except Exception as e:
        return False, str(e)


def send_welcome_whatsapp(phone, username, password, plan_label, expiry):
    """Send credentials to user via WhatsApp CallMeBot."""
    if not CALLMEBOT_ALERTS_ON or not phone:
        return False, "WhatsApp OFF"
    try:
        msg = (
            f"🎉 *AI Trading PRO+ — Account Active!*\n"
            f"{'─'*30}\n"
            f"✅ Your payment has been verified!\n\n"
            f"🔑 Username : {username}\n"
            f"🔒 Password : {password}\n"
            f"📦 Plan     : {plan_label}\n"
            f"📅 Valid    : {expiry}\n"
            f"{'─'*30}\n"
            f"📱 Contact admin for app link\n"
            f"📞 +91 98051 84822"
        )
        encoded = urllib.parse.quote(msg)
        url = f"https://api.callmebot.com/whatsapp.php?phone={phone}&text={encoded}&apikey={CALLMEBOT_APIKEY}"
        r = requests.get(url, timeout=12)
        return True, "Sent"
    except Exception as e:
        return False, str(e)


def register_user(username, password, plan_key, email="", phone="", txn_id=""):
    """Main registration function."""
    users = load_users()

    if username in users:
        return False, f"User '{username}' already exists"

    plan = PLANS.get(plan_key, PLANS["monthly"])
    expiry = str(datetime.date.today() + datetime.timedelta(days=plan["days"]))

    users[username] = {
        "password":  password,
        "role":      "user",
        "status":    "active",
        "expiry":    expiry,
        "plan":      plan_key,
        "email":     email,
        "phone":     phone,
        "txn_id":    txn_id,
        "joined":    str(datetime.date.today()),
    }
    save_users(users)

    print(f"\n✅ User '{username}' created successfully!")
    print(f"   Plan   : {plan['label']}")
    print(f"   Expiry : {expiry}")

    # Send welcome email
    if email:
        ok, msg = send_welcome_email(email, username, password, plan["label"], expiry)
        print(f"   📧 Email  : {'✅ Sent to ' + email if ok else '❌ ' + msg}")

    # Send welcome WhatsApp
    if phone:
        ok, msg = send_welcome_whatsapp(phone, username, password, plan["label"], expiry)
        print(f"   📱 WA     : {'✅ Sent to ' + phone if ok else '❌ ' + msg}")

    return True, expiry


def list_users():
    users = load_users()
    print("\n" + "="*60)
    print("  ALL USERS — AI Trading PRO+")
    print("="*60)
    active  = [u for u in users if users[u].get("status")=="active" and u!="admin"]
    pending = [u for u in users if users[u].get("status")=="pending"]
    print(f"\n✅ Active ({len(active)}):")
    for u in active:
        print(f"   {u:20} | {users[u].get('plan','?'):10} | Expiry: {users[u].get('expiry','?')}")
    print(f"\n⏳ Pending ({len(pending)}):")
    for u in pending:
        print(f"   {u:20} | {users[u].get('plan','?'):10} | Joined: {users[u].get('joined','?')}")
    print()


def extend_user(username, days=30):
    users = load_users()
    if username not in users:
        print(f"❌ User '{username}' not found")
        return
    cur = datetime.datetime.strptime(users[username]["expiry"], "%Y-%m-%d").date()
    new_exp = max(cur, datetime.date.today()) + datetime.timedelta(days=days)
    users[username]["expiry"] = str(new_exp)
    save_users(users)
    print(f"✅ Extended '{username}' → {new_exp}")


def delete_user(username):
    users = load_users()
    if username not in users or username == "admin":
        print(f"❌ Cannot delete '{username}'")
        return
    del users[username]
    save_users(users)
    print(f"✅ Deleted '{username}'")


# =============================================================
# INTERACTIVE MENU
# =============================================================
def main():
    # Direct CLI: python register_user.py username password plan
    if len(sys.argv) >= 4:
        ok, result = register_user(sys.argv[1], sys.argv[2], sys.argv[3])
        sys.exit(0 if ok else 1)

    while True:
        print("\n" + "="*50)
        print("  AI Trading PRO+ — Admin Registration Tool")
        print("="*50)
        print("  1. ➕ Add New User (after payment)")
        print("  2. 📋 List All Users")
        print("  3. 🔄 Extend User Subscription")
        print("  4. 🗑  Delete User")
        print("  5. 🚪 Exit")
        print("="*50)

        choice = input("Choose (1-5): ").strip()

        if choice == "1":
            print("\n── NEW USER REGISTRATION ──")
            username = input("Username     : ").strip()
            password = input("Password     : ").strip()
            print("Plans: monthly / quarterly / annual")
            plan     = input("Plan         : ").strip().lower()
            email    = input("User Email   : ").strip()
            phone    = input("User Phone (+91...): ").strip()
            txn_id   = input("UPI Txn ID   : ").strip()

            if not username or not password:
                print("❌ Username and password required")
                continue

            if plan not in PLANS:
                plan = "monthly"
                print(f"⚠️ Invalid plan — defaulting to monthly")

            confirm = input(f"\nCreate user '{username}' on {plan} plan? (y/n): ")
            if confirm.lower() == "y":
                register_user(username, password, plan, email, phone, txn_id)

        elif choice == "2":
            list_users()

        elif choice == "3":
            username = input("Username to extend: ").strip()
            days_str = input("Extend by days (default 30): ").strip()
            days = int(days_str) if days_str.isdigit() else 30
            extend_user(username, days)

        elif choice == "4":
            username = input("Username to delete: ").strip()
            confirm  = input(f"Delete '{username}'? This cannot be undone (y/n): ")
            if confirm.lower() == "y":
                delete_user(username)

        elif choice == "5":
            print("Bye!")
            break

        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
