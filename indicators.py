# modules/indicators.py
# =============================================================
# Technical indicators — all cached, no recomputation
# =============================================================
import streamlit as st
import pandas as pd
import numpy as np


@st.cache_data(ttl=300, show_spinner=False)
def compute(_df_json: str) -> pd.DataFrame:
    """
    Compute 25+ indicators. Input as JSON string for caching.
    Usage:
        df_json = df.to_json()
        df_ind  = indicators.compute(df_json)
    """
    df = pd.read_json(_df_json)
    return _compute(df)


def compute_direct(df: pd.DataFrame) -> pd.DataFrame:
    """Direct (uncached) computation — use when df is already fresh."""
    return _compute(df)


def _compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if len(df) < 5: return df

    # ── EMAs ────────────────────────────────────────────────
    for s in [5, 9, 20, 50, 100, 200]:
        df[f"EMA{s}"] = df["Close"].ewm(span=s).mean()

    # ── RSI ─────────────────────────────────────────────────
    delta = df["Close"].diff()
    g = delta.where(delta>0,0).rolling(14).mean()
    l = (-delta.where(delta<0,0)).rolling(14).mean()
    df["RSI"]    = 100 - 100/(1+g/(l+1e-9))
    df["RSI_MA"] = df["RSI"].rolling(9).mean()

    # ── MACD ────────────────────────────────────────────────
    df["EMA12"]       = df["Close"].ewm(12).mean()
    df["EMA26"]       = df["Close"].ewm(26).mean()
    df["MACD"]        = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(9).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ── ATR ─────────────────────────────────────────────────
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    df["ATR"] = pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14).mean()

    # ── Bollinger Bands ─────────────────────────────────────
    df["BB_Mid"]   = df["Close"].rolling(20).mean()
    std            = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2*std
    df["BB_Lower"] = df["BB_Mid"] - 2*std
    df["BB_Width"] = (df["BB_Upper"]-df["BB_Lower"])/(df["BB_Mid"]+1e-9)
    df["BB_Pct"]   = (df["Close"]-df["BB_Lower"])/(df["BB_Upper"]-df["BB_Lower"]+1e-9)

    # ── Stochastic ──────────────────────────────────────────
    low14  = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100*(df["Close"]-low14)/(high14-low14+1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # ── VWAP ────────────────────────────────────────────────
    tp = (df["High"]+df["Low"]+df["Close"])/3
    df["VWAP"] = (tp*df["Volume"]).cumsum()/(df["Volume"].cumsum()+1e-9)

    # ── ADX ─────────────────────────────────────────────────
    pdm = df["High"].diff().clip(lower=0)
    ndm = (-df["Low"].diff()).clip(lower=0)
    pdm[pdm<ndm]=0; ndm[ndm<pdm]=0
    atr = df["ATR"]
    pdi = 100*pdm.rolling(14).mean()/(atr+1e-9)
    ndi = 100*ndm.rolling(14).mean()/(atr+1e-9)
    dx  = 100*(pdi-ndi).abs()/(pdi+ndi+1e-9)
    df["ADX"]     = dx.rolling(14).mean()
    df["Plus_DI"] = pdi; df["Minus_DI"] = ndi

    # ── OBV ─────────────────────────────────────────────────
    obv = [0]
    for i in range(1,len(df)):
        if df["Close"].iloc[i]>df["Close"].iloc[i-1]:
            obv.append(obv[-1]+df["Volume"].iloc[i])
        elif df["Close"].iloc[i]<df["Close"].iloc[i-1]:
            obv.append(obv[-1]-df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"]    = obv
    df["OBV_MA"] = pd.Series(obv,index=df.index).rolling(20).mean()

    # ── MFI ─────────────────────────────────────────────────
    tp2 = (df["High"]+df["Low"]+df["Close"])/3
    mf  = tp2*df["Volume"]
    pmf = mf.where(tp2>tp2.shift(),0).rolling(14).sum()
    nmf = mf.where(tp2<tp2.shift(),0).rolling(14).sum()
    df["MFI"] = 100-100/(1+pmf/(nmf+1e-9))

    # ── CCI ─────────────────────────────────────────────────
    tp3 = (df["High"]+df["Low"]+df["Close"])/3
    md  = (tp3-tp3.rolling(20).mean()).abs().rolling(20).mean()
    df["CCI"] = (tp3-tp3.rolling(20).mean())/(0.015*md+1e-9)

    # ── Williams %R ─────────────────────────────────────────
    df["Williams_R"] = -100*(high14-df["Close"])/(high14-low14+1e-9)

    # ── Supertrend ──────────────────────────────────────────
    hl2 = (df["High"]+df["Low"])/2
    up  = hl2+3*df["ATR"]; dn = hl2-3*df["ATR"]
    st_dir = pd.Series(0,index=df.index)
    st_val = pd.Series(0.0,index=df.index)
    for i in range(1,len(df)):
        if df["Close"].iloc[i] > up.iloc[i-1]:   st_dir.iloc[i] = 1
        elif df["Close"].iloc[i] < dn.iloc[i-1]: st_dir.iloc[i] = -1
        else: st_dir.iloc[i] = st_dir.iloc[i-1]
        st_val.iloc[i] = dn.iloc[i] if st_dir.iloc[i]==1 else up.iloc[i]
    df["Supertrend"] = st_val; df["ST_Dir"] = st_dir

    # ── Volume & Returns ────────────────────────────────────
    df["Vol_Ratio"] = df["Volume"]/(df["Volume"].rolling(20).mean()+1e-9)
    df["Vol_MA20"]  = df["Volume"].rolling(20).mean()
    df["Return_1"]  = df["Close"].pct_change(1)
    df["Return_3"]  = df["Close"].pct_change(3)
    df["Return_5"]  = df["Close"].pct_change(5)
    df["Volatility"]= df["Return_1"].rolling(20).std()*(252**0.5)
    df["Price_Pos"] = (df["Close"]-df["Low"].rolling(20).min())/(
        df["High"].rolling(20).max()-df["Low"].rolling(20).min()+1e-9)

    return df
