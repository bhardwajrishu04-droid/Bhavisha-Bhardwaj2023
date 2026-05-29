# =============================================================
# kite_token.py — Access Token Generator
# Usage: python kite_token.py
# WARNING: Apni real keys config.py mein daalo, yahan nahi
# =============================================================

from kiteconnect import KiteConnect

# config.py se keys lo
try:
    from config import API_KEY, API_SECRET
except ImportError:
    API_KEY    = input("API Key: ").strip()
    API_SECRET = input("API Secret: ").strip()

kite = KiteConnect(api_key=API_KEY)

print(f"\nStep 1: Yeh URL browser mein kholo:")
print(kite.login_url())
print("\nStep 2: Login karo → URL se request_token copy karo")
print("        (URL mein ?request_token=XXXXX dikhega)")

request_token = input("\nRequest Token paste karo: ").strip()

try:
    data         = kite.generate_session(request_token, api_secret=API_SECRET)
    access_token = data["access_token"]
    print(f"\nACCESS TOKEN: {access_token}")
    print("\nYeh token Streamlit Secrets mein daalo:")
    print(f'ACCESS_TOKEN = "{access_token}"')
except Exception as e:
    print(f"Error: {e}")
