from kiteconnect import KiteConnect

api_key = "6c3uhkm1yw56fd8u"
api_secret = "jnh8ms80eeluto2xg5ustm7khbxm8u7z"
request_token = "uSKwzut5pNrIvdg8Yw2Q19UF4VxSE35m"

kite = KiteConnect(api_key=api_key)

data = kite.generate_session(request_token, api_secret=api_secret)

access_token = data["access_token"]

print("ACCESS TOKEN:", access_token)