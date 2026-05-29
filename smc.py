# modules/smc.py
# =============================================================
# Price Action + SMC Analysis
# Candlestick patterns, Order Blocks, FVG,
# Market Structure, Demand/Supply, Fake Breakouts
# =============================================================
import streamlit as st
import pandas as pd
import numpy as np


@st.cache_data(ttl=300, show_spinner=False)
def full_analysis(_df_json: str) -> dict:
    """All price action & SMC analysis. Cached 5 min."""
    df = pd.read_json(_df_json)
    return {
        "patterns":   _candlestick_patterns(df),
        "structure":  _market_structure(df),
        "order_blocks": _order_blocks(df),
        "fvg":        _fvg(df),
        "fake_bo":    _fake_breakout(df),
        "demand_supply": _demand_supply(df),
        "volume_profile": _volume_profile(df),
    }


def _candlestick_patterns(df):
    df = df.copy().dropna(); found = []
    if len(df) < 5: return found
    for i in range(2, len(df)):
        o=float(df["Open"].iloc[i]); h=float(df["High"].iloc[i])
        l=float(df["Low"].iloc[i]);  c=float(df["Close"].iloc[i])
        o1=float(df["Open"].iloc[i-1]); h1=float(df["High"].iloc[i-1])
        l1=float(df["Low"].iloc[i-1]);  c1=float(df["Close"].iloc[i-1])
        o2=float(df["Open"].iloc[i-2]); c2=float(df["Close"].iloc[i-2])
        body=abs(c-o); body1=abs(c1-o1)
        rng=h-l+1e-9
        up=h-max(o,c); dn=min(o,c)-l
        atr=float(df["ATR"].iloc[i]) if "ATR" in df.columns else rng
        dt=str(df.index[i])[:10]

        def add(name,typ,st,desc,sig):
            found.append({"pattern":name,"type":typ,"strength":st,
                          "desc":desc,"signal":sig,"date":dt})

        if body<atr*0.1 and rng>atr*0.5: add("Doji","neutral",2,"Indecision","WAIT")
        if body<atr*0.05 and up>atr*0.6 and dn<atr*0.05:
            add("Gravestone Doji","bearish",3,"Bears pushed back down","SELL")
        if body<atr*0.05 and dn>atr*0.6 and up<atr*0.05:
            add("Dragonfly Doji","bullish",3,"Bulls pushed back up","BUY")
        if dn>2*body and up<body*0.5 and c>o and body>atr*0.1:
            add("Hammer","bullish",4,"Buyers rejected lower prices","BUY")
        if up>2*body and dn<body*0.3 and c<o and body>atr*0.1:
            add("Shooting Star","bearish",4,"Sellers rejected higher","SELL")
        if c>o and body>atr*0.8 and up<body*0.05 and dn<body*0.05:
            add("Bullish Marubozu","bullish",5,"Full bull control","STRONG BUY")
        if c<o and body>atr*0.8 and up<body*0.05 and dn<body*0.05:
            add("Bearish Marubozu","bearish",5,"Full bear control","STRONG SELL")
        if c1<o1 and c>o and c>o1 and o<c1 and body>body1*0.9:
            add("Bullish Engulfing","bullish",5,"Bulls overwhelm bears","STRONG BUY")
        if c1>o1 and c<o and c<o1 and o>c1 and body>body1*0.9:
            add("Bearish Engulfing","bearish",5,"Bears overwhelm bulls","STRONG SELL")
        if c1<o1 and c>o and c<o1 and o>c1:
            add("Bullish Harami","bullish",3,"Bearish momentum slowing","WATCH BUY")
        if c1>o1 and c<o and c>o1 and o<c1:
            add("Bearish Harami","bearish",3,"Bullish momentum slowing","WATCH SELL")
        if c1<o1 and c>o and o<l1 and c>(o1+c1)/2 and c<o1:
            add("Piercing Line","bullish",4,"Buyers pierce midpoint","BUY")
        if c1>o1 and c<o and o>h1 and c<(o1+c1)/2:
            add("Dark Cloud Cover","bearish",4,"Sellers pierce midpoint","SELL")
        if c1<o1 and c>o and abs(l-l1)<atr*0.05:
            add("Tweezer Bottom","bullish",3,"Support confirmed","BUY")
        if c1>o1 and c<o and abs(h-h1)<atr*0.05:
            add("Tweezer Top","bearish",3,"Resistance confirmed","SELL")
        if c2<o2 and abs(c1-o1)<body*0.3 and c>o and c>(o2+c2)/2:
            add("Morning Star","bullish",5,"3-candle bottom reversal","STRONG BUY")
        if c2>o2 and abs(c1-o1)<body*0.3 and c<o and c<(o2+c2)/2:
            add("Evening Star","bearish",5,"3-candle top reversal","STRONG SELL")
        if c>o and c1>o1 and c2>o2 and c>c1>c2 and o>o1>o2:
            add("Three White Soldiers","bullish",5,"3 green candles","STRONG BUY")
        if c<o and c1<o1 and c2<o2 and c<c1<c2 and o<o1<o2:
            add("Three Black Crows","bearish",5,"3 red candles","STRONG SELL")

    seen=set(); unique=[]
    for p in reversed(found):
        if p["pattern"] not in seen:
            seen.add(p["pattern"]); unique.append(p)
    return sorted(unique[:12], key=lambda x:-x["strength"])


def _market_structure(df, n=5):
    if len(df) < n*3: return {"trend":"Unknown","trend_color":"#888"}
    h=df["High"].values; l=df["Low"].values; c=df["Close"].values
    sh=[]; sl=[]
    for i in range(n,len(df)-n):
        if all(h[i]>=h[i-n:i]) and all(h[i]>=h[i+1:i+n+1]): sh.append((i,h[i]))
        if all(l[i]<=l[i-n:i]) and all(l[i]<=l[i+1:i+n+1]): sl.append((i,l[i]))
    if len(sh)<2 or len(sl)<2:
        return {"trend":"Insufficient data","trend_color":"#888",
                "hh":False,"hl":False,"lh":False,"ll":False}
    hh=sh[-1][1]>sh[-2][1]; lh=sh[-1][1]<sh[-2][1]
    hl=sl[-1][1]>sl[-2][1]; ll=sl[-1][1]<sl[-2][1]
    if hh and hl:   tr,tc="Uptrend (HH+HL)","#00b880"
    elif lh and ll: tr,tc="Downtrend (LH+LL)","#e74c3c"
    elif hh and ll: tr,tc="Choppy (HH+LL)","#f39c12"
    else:           tr,tc="Ranging (LH+HL)","#a78bfa"
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 1
    cur=float(df["Close"].iloc[-1]); rh=max(v for _,v in sh[-3:])
    rl=min(v for _,v in sl[-3:])
    mss=None
    if ll and cur>rh+atr*0.3: mss="BULLISH MSS — Break above recent high"
    if hh and cur<rl-atr*0.3: mss="BEARISH MSS — Break below recent low"
    return {"trend":tr,"trend_color":tc,"hh":hh,"hl":hl,"lh":lh,"ll":ll,
            "mss":mss,"recent_high":rh,"recent_low":rl,
            "swing_highs":sh[-5:],"swing_lows":sl[-5:]}


def _order_blocks(df, n=3):
    if len(df)<10: return []
    obs=[]; atr=float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 5
    for i in range(3,len(df)-2):
        o=float(df["Open"].iloc[i]); c=float(df["Close"].iloc[i])
        h=float(df["High"].iloc[i]); l=float(df["Low"].iloc[i])
        c2=float(df["Close"].iloc[i+1])
        c3=float(df["Close"].iloc[i+2]) if i+2<len(df) else c2
        mv=abs(c3-c)
        if c<o and c3>h and mv>atr*1.5:
            obs.append({"type":"Bullish OB","color":"#00b880",
                        "top":round(max(o,c),2),"bottom":round(min(o,c),2),
                        "desc":"Institutional BUY zone","date":str(df.index[i])[:10]})
        if c>o and c3<l and mv>atr*1.5:
            obs.append({"type":"Bearish OB","color":"#e74c3c",
                        "top":round(max(o,c),2),"bottom":round(min(o,c),2),
                        "desc":"Institutional SELL zone","date":str(df.index[i])[:10]})
    return obs[-4:]


def _fvg(df):
    if len(df)<5: return []
    fvgs=[]
    for i in range(1,len(df)-1):
        h1=float(df["High"].iloc[i-1]); l1=float(df["Low"].iloc[i-1])
        h2=float(df["High"].iloc[i+1]); l2=float(df["Low"].iloc[i+1])
        if l2>h1:
            fvgs.append({"type":"Bullish FVG","color":"#00b880",
                "top":round(l2,2),"bottom":round(h1,2),"gap":round(l2-h1,2),
                "desc":"Imbalance — likely to be filled (BUY zone)",
                "date":str(df.index[i])[:10],
                "filled":float(df["Low"].iloc[-1])<=l2})
        if h2<l1:
            fvgs.append({"type":"Bearish FVG","color":"#e74c3c",
                "top":round(l1,2),"bottom":round(h2,2),"gap":round(l1-h2,2),
                "desc":"Imbalance — likely to be filled (SELL zone)",
                "date":str(df.index[i])[:10],
                "filled":float(df["High"].iloc[-1])>=h2})
    return [f for f in fvgs if not f["filled"]][-4:]


def _fake_breakout(df):
    if len(df)<20: return []
    results=[]; atr=float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 5
    rh=float(df["High"].rolling(20).max().iloc[-2])
    rl=float(df["Low"].rolling(20).min().iloc[-2])
    for i in range(2,min(10,len(df))):
        h=float(df["High"].iloc[-i]); l=float(df["Low"].iloc[-i])
        c=float(df["Close"].iloc[-i])
        if h>rh and c<rh-atr*0.1:
            results.append({"type":"Bull Trap","color":"#e74c3c","signal":"SELL",
                "desc":f"Broke above {rh:.2f} then reversed — trap!",
                "date":str(df.index[-i])[:10]})
        if l<rl and c>rl+atr*0.1:
            results.append({"type":"Bear Trap","color":"#00b880","signal":"BUY",
                "desc":f"Broke below {rl:.2f} then reversed — trap!",
                "date":str(df.index[-i])[:10]})
    return results[:3]


def _demand_supply(df, n=3):
    if len(df)<20: return [],[]
    h=df["High"].values; l=df["Low"].values; c=df["Close"].values
    atr=float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 5
    supply=[]; demand=[]
    for i in range(n,len(df)-n):
        if h[i]==max(h[max(0,i-n):i+n+1]) and c[i]<c[i-1]:
            supply.append({"top":round(h[i],2),"bottom":round(h[i]-atr*0.5,2),
                           "strength":min(5,int((h[i]-l[i])/atr*2)+1),
                           "date":str(df.index[i])[:10]})
        if l[i]==min(l[max(0,i-n):i+n+1]) and c[i]>c[i-1]:
            demand.append({"top":round(l[i]+atr*0.5,2),"bottom":round(l[i],2),
                           "strength":min(5,int((h[i]-l[i])/atr*2)+1),
                           "date":str(df.index[i])[:10]})
    return supply[-3:], demand[-3:]


def _volume_profile(df, bins=20):
    if len(df)<10: return {}
    pmin=float(df["Low"].min()); pmax=float(df["High"].max())
    bs=(pmax-pmin)/bins if pmax>pmin else 1
    vap={}
    for i in range(len(df)):
        h=float(df["High"].iloc[i]); l=float(df["Low"].iloc[i])
        v=float(df["Volume"].iloc[i]); cb=max(1,int((h-l)/bs))
        vpb=v/cb
        for b in range(cb):
            pl=round(l+b*bs+bs/2,2)
            bk=round((pl-pmin)/bs)*bs+pmin
            vap[round(bk,2)]=vap.get(round(bk,2),0)+vpb
    if not vap: return {}
    sl=sorted(vap.items(),key=lambda x:-x[1])
    poc=sl[0][0]; tv=sum(vap.values()); tgt=tv*0.70
    acc=0; vap_ls=[]
    for prc,vol in sl:
        acc+=vol; vap_ls.append(prc)
        if acc>=tgt: break
    vah=max(vap_ls) if vap_ls else poc
    val=min(vap_ls) if vap_ls else poc
    cur=float(df["Close"].iloc[-1])
    if cur>vah: pos="ABOVE Value Area — potential rejection"
    elif cur<val: pos="BELOW Value Area — potential support"
    elif abs(cur-poc)/poc<0.005: pos="AT Point of Control — key level"
    else: pos="INSIDE Value Area — balanced"
    return {"poc":round(poc,2),"vah":round(vah,2),"val":round(val,2),
            "current":round(cur,2),"position":pos,
            "top_levels":[(round(p,2),round(v/1e6,2)) for p,v in sl[:5]]}
