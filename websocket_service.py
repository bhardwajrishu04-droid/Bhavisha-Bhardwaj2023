
from kiteconnect import KiteTicker

class LiveMarketFeed:
    def __init__(self, api_key, access_token):
        self.kws = KiteTicker(api_key, access_token)

    def connect(self):
        self.kws.connect(threaded=True)
