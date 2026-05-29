# modules/data.py
# =============================================================
# Data layer — cached yfinance + KiteTicker websocket
# =============================================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import threading
import queue

# ── CACHED YFINANCE ──────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)   # 5-min cache
def fetch_ohlcv(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV with fallback chain. Cached 5 min."""
    fallbacks = [
        (period, interval),
        ("5d",  "15m"),
        ("5d",  "30m"),
        ("1mo", "1d"),
        ("3mo", "1d"),
        ("6mo", "1d"),
    ]
    seen = set()
    for fp, fi in fallbacks:
        key = f"{fp}_{fi}"
        if key in seen: continue
        seen.add(key)
        try:
            d = yf.Ticker(symbol).history(period=fp, interval=fi)
            if d is not None and not d.empty and len(d) >= 20:
                if (fp, fi) != (period, interval):
                    st.caption(f"Using {fp}/{fi} (primary timeframe unavailable)")
                return d
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)    # 1-min cache for live price
def fetch_live_price(symbol: str) -> float:
    """Latest close price. Cached 1 min."""
    try:
        d = yf.Ticker(symbol).history(period="1d", interval="1m")
        return float(d["Close"].iloc[-1]) if not d.empty else 0.0
    except Exception:
        return 0.0


@st.cache_data(ttl=600, show_spinner=False)   # 10-min cache
def fetch_options_chain(symbol: str) -> dict:
    """Options chain — PCR, Max Pain, top OI strikes. Cached 10 min."""
    try:
        ticker = yf.Ticker(symbol)
        exps   = ticker.options
        if not exps: return {}
        chain  = ticker.option_chain(exps[0])
        calls, puts = chain.calls, chain.puts
        call_oi = int(calls["openInterest"].sum()) if "openInterest" in calls else 0
        put_oi  = int(puts["openInterest"].sum())  if "openInterest" in puts  else 0
        pcr     = round(put_oi / max(call_oi, 1), 2)

        # Max Pain
        max_pain = 0
        try:
            strikes = sorted(set(
                calls["strike"].tolist() + puts["strike"].tolist()))
            pains = []
            for s in strikes:
                cp = sum(max(0, s-k)*oi for k,oi in
                         zip(calls["strike"], calls["openInterest"].fillna(0))
                         if s > k)
                pp = sum(max(0, k-s)*oi for k,oi in
                         zip(puts["strike"], puts["openInterest"].fillna(0))
                         if s < k)
                pains.append((s, cp+pp))
            if pains:
                max_pain = min(pains, key=lambda x:x[1])[0]
        except Exception:
            pass

        if   pcr > 1.5: pcr_sig = "VERY BULLISH"
        elif pcr > 1.2: pcr_sig = "BULLISH"
        elif pcr > 0.8: pcr_sig = "NEUTRAL"
        elif pcr > 0.5: pcr_sig = "BEARISH"
        else:           pcr_sig = "VERY BEARISH"

        return {
            "expiry": exps[0], "call_oi": call_oi, "put_oi": put_oi,
            "pcr": pcr, "pcr_signal": pcr_sig, "max_pain": max_pain,
            "top_calls": calls.nlargest(3,"openInterest")[
                ["strike","openInterest"]].values.tolist()
                if "openInterest" in calls else [],
            "top_puts":  puts.nlargest(3,"openInterest")[
                ["strike","openInterest"]].values.tolist()
                if "openInterest" in puts else [],
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_overview() -> dict:
    """Nifty, BankNifty, Sensex, VIX. Cached 5 min."""
    result = {}
    symbols = {
        "NIFTY 50":    "^NSEI",
        "BANK NIFTY":  "^NSEBANK",
        "SENSEX":      "^BSESN",
        "INDIA VIX":   "^INDIAVIX",
    }
    for name, sym in symbols.items():
        try:
            d = yf.Ticker(sym).history(period="2d", interval="1d")
            if not d.empty:
                price = float(d["Close"].iloc[-1])
                prev  = float(d["Close"].iloc[-2]) if len(d)>1 else price
                result[name] = {
                    "price": price,
                    "chg":   round((price-prev)/prev*100, 2),
                }
        except Exception:
            pass
    return result


# ── KITE TICKER (WEBSOCKET) ───────────────────────────────────
class KiteWebSocket:
    """
    Zerodha KiteTicker wrapper.
    Usage:
        ws = KiteWebSocket(api_key, access_token)
        ws.subscribe([738561, 895745])   # instrument tokens
        price = ws.get_ltp(738561)
    """
    def __init__(self, api_key: str, access_token: str):
        self.api_key      = api_key
        self.access_token = access_token
        self.ticks        = {}          # token → ltp
        self._ticker      = None
        self._connected   = False
        self._q           = queue.Queue()

    def connect(self) -> bool:
        try:
            from kiteconnect import KiteTicker
            self._ticker = KiteTicker(self.api_key, self.access_token)
            self._ticker.on_connect   = self._on_connect
            self._ticker.on_ticks     = self._on_ticks
            self._ticker.on_close     = self._on_close
            self._ticker.on_error     = self._on_error
            t = threading.Thread(target=self._ticker.connect, daemon=True)
            t.start()
            return True
        except Exception as e:
            st.warning(f"KiteTicker: {e}")
            return False

    def subscribe(self, tokens: list):
        if self._ticker and self._connected:
            self._ticker.subscribe(tokens)
            self._ticker.set_mode(self._ticker.MODE_LTP, tokens)

    def get_ltp(self, token: int) -> float:
        return self.ticks.get(token, 0.0)

    def is_connected(self) -> bool:
        return self._connected

    def _on_connect(self, ws, response):
        self._connected = True

    def _on_ticks(self, ws, ticks):
        for tick in ticks:
            self.ticks[tick["instrument_token"]] = tick.get("last_price", 0.0)

    def _on_close(self, ws, code, reason):
        self._connected = False

    def _on_error(self, ws, code, reason):
        self._connected = False


def get_ws_instance(api_key: str, access_token: str) -> KiteWebSocket:
    """Get or create WebSocket instance stored in session state."""
    if "kite_ws" not in st.session_state or not st.session_state.kite_ws:
        ws = KiteWebSocket(api_key, access_token)
        ws.connect()
        st.session_state.kite_ws = ws
    return st.session_state.kite_ws
