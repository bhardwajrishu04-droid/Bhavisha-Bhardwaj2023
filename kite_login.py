# =============================================================
# kite_login.py — Login URL generator
# =============================================================

from kiteconnect import KiteConnect

try:
    from config import API_KEY
except ImportError:
    API_KEY = input("API Key: ").strip()

kite = KiteConnect(api_key=API_KEY)
print("Login URL:", kite.login_url())
