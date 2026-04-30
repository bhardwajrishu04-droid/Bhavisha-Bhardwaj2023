from kiteconnect import KiteConnect

api_key = "6c3uhkm1yw56fd8u"

kite = KiteConnect(api_key=api_key)

print("Login URL:", kite.login_url())