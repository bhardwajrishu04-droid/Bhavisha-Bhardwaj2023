# 📈 AI Trading PRO+ v1.3

> Professional AI-powered trading platform for NSE Indian markets

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-red)](https://bhavisha-ai-trading-pro.streamlit.app)

---

## ✨ Features

- **AI Signal Engine** — Combined score (Technical + RandomForest)
- **Master Signal** — 6-layer analysis with win probability
- **25+ Candlestick Patterns** — Auto-detected on every chart
- **SMC Analysis** — Order Blocks, FVG, Market Structure
- **Volume Profile** — POC, VAH, VAL
- **Options Data** — PCR, Max Pain, OI analysis
- **4 Trading Modes** — Intraday, Swing, Futures, Options
- **60+ Stocks** — Nifty 50, Bank Nifty, IT, Auto, FMCG, Pharma
- **Paper + Live Trading** — Kite Connect integration
- **Admin Panel** — User management, UPI payments
- **Email + WhatsApp Alerts** — Real-time notifications

---

## 🚀 Quick Start (Local PC)

```bash
# 1. Clone repo
git clone https://github.com/bhardwajrishu04-droid/Bhavisha-Bhardwaj2023
cd Bhavisha-Bhardwaj2023

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy config template and fill your keys
cp config.py config_local.py
# Edit config_local.py with your Kite API keys

# 4. Run app
streamlit run app.py
```

---

## ⚙️ Configuration

### Streamlit Cloud (Recommended)
Add these in **App Settings → Secrets**:
```toml
API_KEY = "your_kite_api_key"
API_SECRET = "your_kite_secret"
ACCESS_TOKEN = ""
EMAIL_ALERTS_ON = true
ALERT_EMAIL_TO = "your@email.com"
SMTP_USER = "your@gmail.com"
SMTP_PASS = "your_gmail_app_password"
APP_URL = "https://your-app.streamlit.app"
```

### Local PC
Edit `config.py` with your keys (never commit this file!)

---

## 📁 File Structure

```
├── app.py              ← Main Streamlit app
├── alerts.py           ← Email + WhatsApp alerts
├── bot.py              ← Auto trading bot (local only)
├── register_user.py    ← Admin CLI user registration
├── start_app.py        ← One-click app launcher
├── kite_login.py       ← Get Kite login URL
├── kite_token.py       ← Generate access token
├── config.py           ← Local config (DO NOT commit!)
├── requirements.txt    ← Python dependencies
└── .gitignore          ← Sensitive files excluded
```

---

## 🔐 Security

- **Never commit** `config.py` or `users.json` to GitHub
- Use Streamlit Secrets for production deployment
- Rotate Kite API keys regularly (they expire)
- Gmail App Password (not regular password)

---

## 💰 Plans

| Plan | Price | Duration |
|------|-------|----------|
| Monthly | ₹499 | 30 days |
| Quarterly | ₹999 | 90 days |
| Annual | ₹2,999 | 365 days |

UPI: `bhardwaj.rishu04@oksbi`

---

## 📞 Support

- WhatsApp: +91 98051 84822
- Email: bhardwaj.rishu04@gmail.com

---

## ⚠️ Disclaimer

This software is for educational purposes only.
Trading involves substantial risk. Past performance does not guarantee future results.
Always use Stop Loss. Never trade with money you can't afford to lose.

---

MIT License © 2026 bhardwajrishu04-droid
