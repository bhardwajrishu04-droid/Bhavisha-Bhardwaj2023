# modules/backtest.py
# =============================================================
# Professional Backtesting Engine
# Metrics: Sharpe, Sortino, Calmar, Max Drawdown,
#          Expectancy, Monthly Returns, Trade Replay
# Strategies: master, trend, momentum, mean_revert
# =============================================================
import numpy as np
import pandas as pd
from typing import Optional


def run(df: pd.DataFrame, strategy: str = "master",
        initial_capital: float = 100000,
        sl_atr_mult: float = 1.5,
        tp_rr: float = 2.0,
        risk_pct: float = 1.5) -> Optional[dict]:
    """
    Full walk-forward backtest.
    Returns complete metrics dict or None if insufficient data.
    """
    df = df.copy().dropna(subset=["Close"])
    if len(df) < 50: return None

    capital   = float(initial_capital)
    position  = 0; entry_px = 0.0
    stop_px   = 0.0; target_px = 0.0
    trades    = []; equity = [capital]
    monthly_r = {}

    for i in range(30, len(df)-1):
        row  = df.iloc[i]
        p    = float(row["Close"])
        atr  = float(row.get("ATR", p*0.015))
        rsi  = float(row.get("RSI", 50))
        macd = float(row.get("MACD", 0))
        ms   = float(row.get("MACD_Signal", 0))
        adx  = float(row.get("ADX", 15))
        std  = float(row.get("ST_Dir", 0))
        vr   = float(row.get("Vol_Ratio", 1))
        e20  = float(row.get("EMA20", p))
        e50  = float(row.get("EMA50", p))
        mfi  = float(row.get("MFI", 50))
        mh   = float(row.get("MACD_Hist", 0))
        date = str(df.index[i])[:10]
        mo   = str(df.index[i])[:7]

        if strategy == "master":
            conds = [p>e20>e50, 45<rsi<72, macd>ms and mh>0,
                     std>0, adx>20, vr>1.1, mfi>50]
            buy  = sum(conds)>=5 and rsi<75
            sell = sum(conds)<=2 or rsi>78 or std<0
        elif strategy == "trend":
            buy  = p>e20>e50 and adx>25 and std>0
            sell = p<e20 or std<0
        elif strategy == "momentum":
            buy  = macd>ms and 50<rsi<70 and vr>1.2
            sell = macd<ms or rsi>75
        elif strategy == "mean_revert":
            buy  = rsi<35 and p<float(row.get("BB_Lower",p))
            sell = rsi>65 or p>float(row.get("BB_Mid",p))
        else:
            buy = sell = False

        if position==0 and buy and capital>p*5:
            sl_d = atr*sl_atr_mult
            qty  = max(1, int(capital*(risk_pct/100)/sl_d))
            if qty*p <= capital:
                position=qty; entry_px=p
                stop_px=round(p-sl_d,2); target_px=round(p+sl_d*tp_rr,2)
                capital -= qty*p
                trades.append({"type":"BUY","date":date,"price":round(p,2),
                               "qty":qty,"sl":stop_px,"target":target_px})

        elif position>0:
            h_sl  = p<=stop_px
            h_tgt = p>=target_px
            if sell or h_sl or h_tgt:
                ep  = stop_px if h_sl else (target_px if h_tgt else p)
                pnl = (ep-entry_px)*position
                capital += position*ep
                reason = "SL" if h_sl else ("Target" if h_tgt else "Signal")
                trades.append({"type":"SELL","date":date,"price":round(ep,2),
                               "qty":position,"pnl":round(pnl,2),"reason":reason})
                position=0; entry_px=0; stop_px=0; target_px=0

        equity.append(capital+position*p)
        monthly_r[mo] = capital+position*p

    if position>0:
        fp  = float(df["Close"].iloc[-1])
        pnl = (fp-entry_px)*position
        capital += position*fp
        trades.append({"type":"SELL","date":str(df.index[-1])[:10],
                       "price":round(fp,2),"qty":position,
                       "pnl":round(pnl,2),"reason":"EOD"})

    return _metrics(trades, equity, monthly_r, initial_capital, df)


def _metrics(trades, equity, monthly_r, initial_capital, df) -> dict:
    eq  = np.array(equity)
    ret = np.diff(eq)/(eq[:-1]+1e-9)
    cl  = [t for t in trades if "pnl" in t]
    ws  = [t for t in cl if t["pnl"]>0]
    ls  = [t for t in cl if t["pnl"]<=0]
    tp  = sum(t["pnl"] for t in cl)
    wr  = len(ws)/len(cl)*100 if cl else 0
    aw  = float(np.mean([t["pnl"] for t in ws]))    if ws else 0
    al  = abs(float(np.mean([t["pnl"] for t in ls]))) if ls else 1
    gp  = sum(t["pnl"] for t in ws)
    gl  = abs(sum(t["pnl"] for t in ls))
    pf  = gp/(gl+1e-9)
    exp = (wr/100*aw)-((1-wr/100)*al)
    pk  = np.maximum.accumulate(eq)
    dd  = (eq-pk)/(pk+1e-9)
    mdd = float(dd.min())*100
    sh  = float(np.mean(ret)/np.std(ret+1e-9)*np.sqrt(252)) if len(ret)>2 else 0
    nr  = ret[ret<0]
    so  = float(np.mean(ret)/(np.std(nr)+1e-9)*np.sqrt(252)) if len(nr)>1 else 0
    ca  = abs(tp/initial_capital*100)/(abs(mdd)+1e-9)
    bhr = (float(df["Close"].iloc[-1])/float(df["Close"].iloc[30])-1)*100

    months = sorted(monthly_r.keys())
    mo_rets= [round((monthly_r[months[j]]-monthly_r[months[j-1]])/
                    monthly_r[months[j-1]]*100,2)
              for j in range(1,len(months))]

    # Consecutive streaks
    cw=cl2=mcw=mcl=0
    for t in cl:
        if t["pnl"]>0: cw+=1; cl2=0
        else: cl2+=1; cw=0
        mcw=max(mcw,cw); mcl=max(mcl,cl2)

    fin = eq[-1] if len(eq)>0 else initial_capital
    return {
        "trades": trades, "closed": cl,
        "total_trades": len(cl), "wins": len(ws), "losses": len(ls),
        "total_pnl":     round(tp,2),
        "final_capital": round(fin,2),
        "return_pct":    round((fin-initial_capital)/initial_capital*100,2),
        "bh_return":     round(bhr,2),
        "win_rate":      round(wr,1),
        "avg_win":       round(aw,2),
        "avg_loss":      round(al,2),
        "profit_factor": round(pf,2),
        "expectancy":    round(exp,2),
        "max_drawdown":  round(mdd,2),
        "recovery_factor": round(abs(tp/initial_capital*100/(abs(mdd)+1e-9)),2),
        "max_consec_wins": mcw,
        "max_consec_loss": mcl,
        "sharpe":  round(sh,2),
        "sortino": round(so,2),
        "calmar":  round(ca,2),
        "equity":     equity,
        "monthly_rets": mo_rets,
        "months":     months[1:] if len(months)>1 else [],
    }
