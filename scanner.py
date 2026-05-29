# modules/scanner.py
# =============================================================
# Stock scanner — fully cached, runs once per universe+mode
# =============================================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from modules import indicators


@st.cache_data(ttl=300, show_spinner=False)   # 5-min cache
def scan(universe_stocks: tuple, period: str, interval: str) -> list:
    """
    Scan all stocks in universe. Cached 5 min.
    Pass stocks as tuple (hashable) for caching.
    """
    results = []
    for sym in universe_stocks:
        try:
            d = yf.Ticker(sym).history(period=period, interval=interval)
            if d is None or d.empty or len(d) < 20: continue
            d = indicators.compute_direct(d)
            last = d.iloc[-1]
            p    = float(last["Close"])

            sc = 0
            if p > last.get("EMA20",p) > last.get("EMA50",p): sc += 2
            if 45 < float(last.get("RSI",50)) < 68:            sc += 1
            if last.get("MACD",0) > last.get("MACD_Signal",0): sc += 1
            if float(last.get("Vol_Ratio",1)) > 1.2:            sc += 1
            if float(last.get("ST_Dir",0)) > 0:                 sc += 1
            if float(last.get("ADX",0)) > 20:                   sc += 1
            if float(last.get("MFI",50)) > 50:                  sc += 1
            # Penalties
            if float(last.get("RSI",50)) > 78: sc -= 2
            if float(last.get("RSI",50)) < 25: sc -= 1
            sc = max(0, min(sc, 7))

            chg = (p - float(d["Close"].iloc[-2]))/float(d["Close"].iloc[-2])*100

            results.append({
                "Stock":  sym.replace(".NS",""),
                "Price":  round(p,2),
                "Chg%":   round(chg,2),
                "RSI":    round(float(last.get("RSI",50)),1),
                "MACD":   round(float(last.get("MACD",0)),2),
                "ADX":    round(float(last.get("ADX",0)),1),
                "MFI":    round(float(last.get("MFI",50)),1),
                "Vol":    round(float(last.get("Vol_Ratio",1)),2),
                "ST":     "Bull" if float(last.get("ST_Dir",0))>0 else "Bear",
                "Score":  sc,
                "Signal": "BUY" if sc>=5 else ("SELL" if sc<=1 else "HOLD"),
                "_sym":   sym,
                "_score": sc,
            })
        except Exception:
            continue

    return sorted(results, key=lambda x: -x["_score"])
