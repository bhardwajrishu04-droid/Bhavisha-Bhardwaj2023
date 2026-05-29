# =============================================================
# AI Trading PRO+ v5.0 — Modular Institutional Architecture
# =============================================================
# app.py — THIN ORCHESTRATOR (imports from modules/)
#
# modules/
#   config.py      — secrets, constants, stock universe
#   auth.py        — SQLite + bcrypt + session tokens
#   data.py        — cached yfinance + KiteTicker websocket
#   indicators.py  — 25+ cached technical indicators
#   ai_engine.py   — XGBoost + LightGBM + LSTM ensemble
#   backtest.py    — professional backtesting
#   scanner.py     — cached stock scanner
#   smc.py         — SMC, patterns, price action (cached)
#   alerts.py      — email + WhatsApp
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import json

from modules import config, auth, data, indicators, ai_engine
from modules import backtest, scanner, smc, alerts

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=config.API_KEY)
    KITE_OK = True
except Exception:
    KITE_OK = False

st.set_page_config(
    page_title="AI Trading PRO+",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# ── DB + Migration ─────────────────────────────────────────────
auth.init_db()
if os.path.exists("users.json"):
    n = auth.migrate_json("users.json")
    if n > 0:
        st.toast(f"Migrated {n} users from users.json to SQLite")

# ── Session State defaults ─────────────────────────────────────
_defaults = {
    "token": None, "user": None, "user_data": None,
    "access_token": None,
    "paper_balance": 500000.0,
    "paper_positions": {},
    "trade_log": [], "pnl_history": [],
    "last_alert_time": {}, "alert_log": [],
    "scan_results": [], "backtest_result": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Auth helpers ───────────────────────────────────────────────
def current_user():
    if not st.session_state.token: return None
    u = auth.validate_session(st.session_state.token)
    if not u:
        st.session_state.token = None
        st.session_state.user  = None
    return u

def do_logout():
    if st.session_state.token:
        auth.revoke_session(st.session_state.token)
    st.session_state.token     = None
    st.session_state.user      = None
    st.session_state.user_data = None
    st.rerun()

# ── Alert helper ───────────────────────────────────────────────
def fire_alert(action, stock, price, qty, sl, tgt, score, mode, pnl=None):
    if not config.ALERT_ON_SIGNAL: return
    if score < config.ALERT_MIN_SCORE and pnl is None: return
    key = f"{stock}_{action}"
    last = st.session_state.last_alert_time.get(key)
    if last and (datetime.datetime.now()-last).seconds/60 < config.ALERT_COOLDOWN_MIN:
        return
    try:
        res = alerts.send(action,stock,price,qty,sl,tgt,score,mode,pnl)
        st.session_state.last_alert_time[key] = datetime.datetime.now()
        for r in res:
            st.session_state.alert_log.insert(0,{"time":datetime.datetime.now().strftime("%H:%M"),"result":r})
        st.session_state.alert_log = st.session_state.alert_log[:10]
    except Exception as e:
        pass

# ==============================================================
# SIDEBAR
# ==============================================================
with st.sidebar:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0a0c10,#161b22);
    border:1px solid #21262d;border-radius:10px;padding:12px;
    text-align:center;margin-bottom:12px;'>
    <div style='font-size:18px;font-weight:700;color:#00e5a0;'>📈 AI Trading PRO+</div>
    <div style='font-size:10px;color:#8b949e;'>v5.0 Institutional Architecture</div>
    </div>""", unsafe_allow_html=True)

    u = current_user()

    if u:
        udata = st.session_state.user_data or auth.get_user(u)
        st.session_state.user_data = udata
        st.markdown(f"👤 **{u}** `{udata.get('plan','Free')}`")
        if st.button("Logout", use_container_width=True):
            do_logout()
        st.markdown("---")

        pages = {
            "📊 Dashboard":     "dashboard",
            "🔍 Analyzer":      "analyzer",
            "🎯 Scanner":       "scanner",
            "📈 Backtest":      "backtest",
            "💼 Portfolio":     "portfolio",
            "📋 Trade Log":     "tradelog",
        }
        if udata.get("role") == "admin":
            pages["🛠 Admin"] = "admin"

        page = st.session_state.get("page", "dashboard")
        for lbl, pid in pages.items():
            if st.button(lbl, use_container_width=True, key=f"nav_{pid}",
                         type="primary" if page==pid else "secondary"):
                st.session_state.page = pid; st.rerun()

        st.markdown("---")
        # Portfolio summary
        bal = st.session_state.paper_balance
        pnl_total = sum(x.get("pnl",0) for x in st.session_state.pnl_history)
        st.markdown(f"""<div style='background:#161b22;border-radius:8px;padding:10px;font-size:12px;'>
<div style='color:#8b949e;margin-bottom:4px;'>Portfolio</div>
<div style='font-size:18px;font-weight:700;color:#00e5a0;'>Rs.{bal:,.0f}</div>
<div style='color:{"#00b880" if pnl_total>=0 else "#e74c3c"};'>P&L: Rs.{pnl_total:+,.2f}</div>
</div>""", unsafe_allow_html=True)

    else:
        if st.button("🔐 Login", use_container_width=True, type="primary"):
            st.session_state.page = "login"; st.rerun()
        if st.button("📝 Sign Up", use_container_width=True):
            st.session_state.page = "signup"; st.rerun()

# ==============================================================
# LOGIN / SIGNUP
# ==============================================================
if not current_user() and st.session_state.get("page","login") == "login":
    st.title("🔐 Login")
    col1, col2 = st.columns([1,1])
    with col1:
        u_in = st.text_input("Username")
        p_in = st.text_input("Password", type="password")
        if st.button("Login", type="primary", use_container_width=True):
            res = auth.login(u_in, p_in)
            if res["ok"]:
                st.session_state.token     = res["token"]
                st.session_state.user      = res["user"]["username"]
                st.session_state.user_data = res["user"]
                st.session_state.page      = "dashboard"
                st.rerun()
            else:
                st.error(res["error"])
        st.markdown("---")
        if st.button("Create Account", use_container_width=True):
            st.session_state.page = "signup"; st.rerun()
    with col2:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1a237e,#1565c0);
        border-radius:12px;padding:24px;color:white;'>
        <div style='font-size:18px;font-weight:700;margin-bottom:12px;'>Features</div>
        <div style='font-size:13px;line-height:2;'>
        XGBoost + LightGBM + LSTM AI<br>
        Professional Backtesting<br>
        SMC + Order Blocks + FVG<br>
        Options PCR + Max Pain<br>
        KiteTicker WebSocket<br>
        Admin Panel + Alerts
        </div></div>""", unsafe_allow_html=True)
    st.stop()

elif not current_user() and st.session_state.get("page") == "signup":
    st.title("📝 Create Account")
    col1, col2 = st.columns([1,1])
    with col1:
        s_name  = st.text_input("Full Name *")
        s_user  = st.text_input("Username *")
        s_pass  = st.text_input("Password * (6+ chars)", type="password")
        s_email = st.text_input("Email")
        s_phone = st.text_input("Phone")
        if st.button("Create Account", type="primary", use_container_width=True):
            if not s_name or not s_user or not s_pass:
                st.error("Fill required fields")
            else:
                res = auth.signup(s_user, s_pass, s_email, s_phone)
                if res["ok"]:
                    st.success("Account created! Please login.")
                    st.session_state.page = "login"; st.rerun()
                else:
                    st.error(res["error"])
        if st.button("Back to Login"):
            st.session_state.page = "login"; st.rerun()
    st.stop()

elif not current_user():
    st.session_state.page = "login"; st.rerun()

# ==============================================================
# AUTHENTICATED PAGES
# ==============================================================
u     = current_user()
udata = st.session_state.user_data or auth.get_user(u)
page  = st.session_state.get("page", "dashboard")

# ── DASHBOARD ─────────────────────────────────────────────────
if page == "dashboard":
    st.title("📊 Market Dashboard")
    st.caption(f"👤 {u} | {datetime.datetime.now().strftime('%d %b %Y  %H:%M:%S')}")

    # Market overview (cached)
    with st.spinner("Loading market data..."):
        mkt = data.fetch_market_overview()
    cols = st.columns(len(mkt))
    for i, (name, val) in enumerate(mkt.items()):
        c = "#00b880" if val["chg"]>=0 else "#e74c3c"
        cols[i].markdown(f"""<div style='background:#161b22;border:1px solid #21262d;
border-radius:8px;padding:12px;text-align:center;'>
<div style='font-size:10px;color:#888;'>{name}</div>
<div style='font-size:20px;font-weight:700;color:#e6edf3;'>
{val["price"]:,.0f}</div>
<div style='font-size:12px;color:{c};'>{"▲" if val["chg"]>=0 else "▼"} {abs(val["chg"]):.2f}%</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    # Quick scan
    st.markdown("### 🏆 Top Signals")
    univ  = config.STOCKS["⭐ Nifty 50 Top 20"]
    with st.spinner("Scanning..."):
        results = scanner.scan(tuple(univ), "5d", "15m")

    top5 = [r for r in results if r["Signal"]=="BUY"][:5]
    if top5:
        tc = st.columns(5)
        for i, r in enumerate(top5):
            c = "#00b880" if r["Chg%"]>=0 else "#e74c3c"
            tc[i].markdown(f"""<div style='background:#0d2818;border:2px solid #00b880;
border-radius:10px;padding:10px;text-align:center;'>
<div style='font-size:15px;font-weight:700;color:#00e5a0;'>{r["Stock"]}</div>
<div style='font-size:16px;color:#e6edf3;'>Rs.{r["Price"]:,.1f}</div>
<div style='font-size:11px;color:{c};'>{"▲" if r["Chg%"]>=0 else "▼"} {abs(r["Chg%"]):.2f}%</div>
<div style='font-size:10px;color:#00b880;'>{r["Score"]}/7 ⭐</div>
</div>""", unsafe_allow_html=True)
    else:
        st.info("No strong BUY signals right now")

# ── ANALYZER ──────────────────────────────────────────────────
elif page == "analyzer":
    st.title("🔍 Stock Analyzer")

    col_set, col_main = st.columns([1,3])
    with col_set:
        univ_name = st.selectbox("Universe", list(config.STOCKS.keys()))
        stocks    = config.STOCKS[univ_name]
        stock     = st.selectbox("Stock", stocks,
                                  format_func=lambda x:x.replace(".NS",""))
        mode_name = st.radio("Mode", list(config.MODES.keys()),
                              label_visibility="collapsed")
        mcfg      = config.MODES[mode_name]
        capital   = st.number_input("Capital", 10000, 10000000, 500000, 5000)
        risk      = st.number_input("Risk %", 0.5, 5.0, 1.5, 0.1)
        order_mode= st.radio("Order Mode", ["Paper","Live"], horizontal=True)
        force     = st.checkbox("Force Signal (Test)")

    with col_main:
        # 1. Fetch data (cached)
        with st.spinner(f"Loading {stock.replace('.NS','')}..."):
            df_raw = data.fetch_ohlcv(stock, mcfg["period"], mcfg["interval"])

        if df_raw.empty:
            st.error("No data — try Swing mode"); st.stop()

        # 2. Compute indicators (cached via JSON)
        df = indicators.compute(df_raw.to_json())
        last  = df.iloc[-1]
        price = float(last["Close"])
        prev  = float(df["Close"].iloc[-2])
        chg_p = (price-prev)/prev*100

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Price",  f"Rs.{price:.2f}", f"{chg_p:+.2f}%")
        c2.metric("RSI",    f"{last.get('RSI',0):.1f}")
        c3.metric("MACD",   f"{last.get('MACD',0):.2f}")
        c4.metric("ATR",    f"Rs.{last.get('ATR',0):.2f}")
        c5.metric("Volume", f"{last.get('Vol_Ratio',0):.2f}x")

        # 3. AI prediction (module call)
        with st.spinner("Running AI ensemble..."):
            ai = ai_engine.predict(df)

        ai_prob = ai["prob"]
        ai_pct  = round(ai_prob*100)

        # 4. Master Signal calculation
        c_trend = last["Close"] > last.get("EMA20",0) > last.get("EMA50",0)
        c_rsi   = 45 < float(last.get("RSI",50)) < 68
        c_macd  = float(last.get("MACD",0)) > float(last.get("MACD_Signal",0))
        c_vol   = float(last.get("Vol_Ratio",1)) > 1.1
        c_stoch = float(last.get("Stoch_K",50)) < 70
        c_adx   = float(last.get("ADX",0)) > 20
        c_mfi   = float(last.get("MFI",50)) > 50
        rsi_ob  = float(last.get("RSI",50)) > 75
        tech_score = sum([c_trend,c_rsi,c_macd,c_vol,c_stoch,c_adx,c_mfi])
        tech_pct   = round(tech_score/7*100)
        master     = round(tech_pct*0.40 + ai_pct*0.35 + 25)  # base 25%
        master     = min(master, 95)
        if rsi_ob: master = max(0, master-15)
        if not c_vol: master = max(0, master-5)

        # Verdict
        if force:            verdict="STRONG BUY"; vc="#00b880"; vb="#003d2a"
        elif rsi_ob:         verdict="NO TRADE";   vc="#e74c3c"; vb="#2d0a0a"
        elif master>=78:     verdict="STRONG BUY"; vc="#00b880"; vb="#003d2a"
        elif master>=65:     verdict="BUY";        vc="#27ae60"; vb="#0a1f10"
        elif master>=50:     verdict="WAIT";       vc="#f39c12"; vb="#1a1200"
        elif master>=35:     verdict="AVOID";      vc="#e07b39"; vb="#2d1800"
        else:                verdict="NO TRADE";   vc="#e74c3c"; vb="#2d0a0a"

        signal = verdict in ["STRONG BUY","BUY"]

        # Trade plan
        atr_v    = max(float(last.get("ATR",price*0.01)), 0.01)
        sl_d     = atr_v * mcfg["sl_mult"]
        tgt_d    = sl_d  * mcfg["rr"]
        sl       = round(price-sl_d,2)
        tgt      = round(price+tgt_d,2)
        qty      = max(1, int(capital*(risk/100)/sl_d))
        max_loss = round(sl_d*qty,2)
        max_gain = round(tgt_d*qty,2)
        win_prob = min(82, round(master*0.7+15))

        # MASTER SIGNAL DISPLAY
        st.markdown(f"""
<div style='background:{vb};border:3px solid {vc};border-radius:14px;
padding:20px 24px;margin:12px 0;'>
  <div style='display:flex;justify-content:space-between;align-items:center;'>
    <div>
      <div style='font-size:11px;color:#888;text-transform:uppercase;'>
        MASTER SIGNAL — {stock.replace(".NS","")} {mode_name}</div>
      <div style='font-size:38px;font-weight:800;color:{vc};'>{verdict}</div>
    </div>
    <div style='text-align:right;'>
      <div style='font-size:52px;font-weight:800;color:{vc};'>{master}%</div>
      <div style='font-size:11px;color:#888;'>Confidence</div>
    </div>
  </div>
  <div style='background:rgba(255,255,255,0.08);border-radius:99px;
  height:14px;margin:12px 0;'>
    <div style='width:{master}%;background:{vc};border-radius:99px;
    height:14px;box-shadow:0 0 10px {vc}55;'></div>
  </div>
  <div style='display:grid;grid-template-columns:repeat(5,1fr);gap:8px;
  text-align:center;margin-bottom:10px;'>
    <div style='background:rgba(0,0,0,0.3);border-radius:6px;padding:8px;'>
      <div style='font-size:10px;color:#888;'>AI Model</div>
      <div style='font-size:18px;font-weight:700;
      color:{"#00b880" if ai_pct>=60 else "#f39c12"};'>{ai_pct}%</div>
    </div>
    <div style='background:rgba(0,0,0,0.3);border-radius:6px;padding:8px;'>
      <div style='font-size:10px;color:#888;'>Technical</div>
      <div style='font-size:18px;font-weight:700;
      color:{"#00b880" if tech_pct>=65 else "#f39c12"};'>{tech_pct}%</div>
    </div>
    <div style='background:rgba(0,0,0,0.3);border-radius:6px;padding:8px;'>
      <div style='font-size:10px;color:#888;'>Win Prob</div>
      <div style='font-size:18px;font-weight:700;color:#a78bfa;'>{win_prob}%</div>
    </div>
    <div style='background:rgba(0,0,0,0.3);border-radius:6px;padding:8px;'>
      <div style='font-size:10px;color:#888;'>Max Loss</div>
      <div style='font-size:18px;font-weight:700;color:#e74c3c;'>
      Rs.{max_loss:,.0f}</div>
    </div>
    <div style='background:rgba(0,0,0,0.3);border-radius:6px;padding:8px;'>
      <div style='font-size:10px;color:#888;'>Max Gain</div>
      <div style='font-size:18px;font-weight:700;color:#00b880;'>
      Rs.{max_gain:,.0f}</div>
    </div>
  </div>
  <div style='background:rgba(0,0,0,0.3);border-radius:8px;padding:10px;
  display:grid;grid-template-columns:repeat(4,1fr);gap:6px;text-align:center;'>
    <div><div style='font-size:10px;color:#888;'>Entry</div>
    <div style='color:#e6edf3;font-weight:600;'>Rs.{price:.2f}</div></div>
    <div><div style='font-size:10px;color:#e74c3c;'>Stop Loss</div>
    <div style='color:#e74c3c;font-weight:600;'>Rs.{sl}</div></div>
    <div><div style='font-size:10px;color:#00b880;'>Target</div>
    <div style='color:#00b880;font-weight:600;'>Rs.{tgt}</div></div>
    <div><div style='font-size:10px;color:#f39c12;'>R:R</div>
    <div style='color:#f39c12;font-weight:600;'>{round(tgt_d/sl_d,1)}:1</div></div>
  </div>
</div>""", unsafe_allow_html=True)

        if signal and config.ALERT_ON_SIGNAL:
            fire_alert(f"{verdict} [{mode_name}]", stock, price,
                       qty, sl, tgt, master, order_mode)

        # AI Engine Details
        with st.expander("🤖 AI Engine Details"):
            st.caption(f"Model: {ai['model_name']}")
            st.caption(f"Models used: {', '.join(ai['models_used']) or 'Fallback RF'}")

            if ai["wf_results"]:
                st.markdown("**Walk-Forward Validation:**")
                wc = st.columns(len(ai["wf_results"]))
                for i, acc in enumerate(ai["wf_results"]):
                    c2 = "#00b880" if acc>=60 else ("#f39c12" if acc>=50 else "#e74c3c")
                    wc[i].markdown(f"""<div style='background:{c2}22;border:1px solid {c2};
border-radius:6px;padding:6px;text-align:center;'>
<div style='font-size:14px;font-weight:700;color:{c2};'>{acc}%</div>
<div style='font-size:9px;color:#888;'>Fold {i+1}</div></div>""",
                    unsafe_allow_html=True)
                avg = round(float(np.mean(ai["wf_results"])),1)
                st.caption(f"Mean: {avg}% | Std: ±{round(float(np.std(ai['wf_results'])),1)}%")

            if ai["feature_importance"]:
                st.markdown("**Feature Importance:**")
                mx = max(ai["feature_importance"].values())+1e-9
                for feat, imp in ai["feature_importance"].items():
                    bw = int(imp/mx*100)
                    st.markdown(f"""<div style='display:flex;align-items:center;
gap:8px;margin-bottom:3px;'>
<div style='min-width:85px;font-size:11px;color:#ccc;'>{feat}</div>
<div style='flex:1;background:#21262d;border-radius:3px;height:7px;'>
<div style='width:{bw}%;background:#4e8fff;height:7px;border-radius:3px;'></div></div>
<div style='font-size:11px;color:#888;min-width:38px;text-align:right;'>
{imp:.3f}</div></div>""", unsafe_allow_html=True)

            if ai.get("lstm_forecast"):
                fc = ai["lstm_forecast"]
                trend = "UP" if fc[-1]>price else "DOWN"
                tc2 = "#00b880" if trend=="UP" else "#e74c3c"
                chg_fc = (fc[-1]-price)/price*100
                st.markdown(f"""<div style='background:{tc2}22;border:1px solid {tc2};
border-radius:8px;padding:10px;margin-top:8px;'>
<div style='font-size:13px;font-weight:600;color:{tc2};'>
LSTM Forecast: {trend} ({chg_fc:+.2f}%)</div>
<div style='font-size:11px;color:#888;margin-top:3px;'>
{"  →  ".join([f"Rs.{p:,.1f}" for p in fc])}</div>
</div>""", unsafe_allow_html=True)

        # SMC Analysis (cached)
        st.markdown("### Advanced Analysis")
        at1, at2 = st.tabs(["Price Action + SMC", "Volume Profile"])

        with at1:
            with st.spinner("Computing SMC..."):
                smc_data = smc.full_analysis(df.tail(80).to_json())

            sc1, sc2 = st.columns(2)
            with sc1:
                ms = smc_data["structure"]
                if ms.get("trend"):
                    tc3 = ms.get("trend_color","#888")
                    st.markdown(f"""<div style='background:#1a1a2e;
border:2px solid {tc3};border-radius:10px;padding:12px;margin-bottom:8px;'>
<div style='font-size:16px;font-weight:700;color:{tc3};'>{ms["trend"]}</div>
{"<div style='font-size:12px;color:#00e5a0;margin-top:4px;'>"+ms['mss']+"</div>" if ms.get("mss") else ""}
</div>""", unsafe_allow_html=True)

                for pt in smc_data["patterns"][:5]:
                    c4 = "#00b880" if pt["type"]=="bullish" else ("#e74c3c" if pt["type"]=="bearish" else "#f39c12")
                    st.markdown(f"""<div style='background:{c4}11;border-left:3px solid {c4};
border-radius:4px;padding:6px 10px;margin-bottom:4px;font-size:12px;'>
<b style='color:{c4};'>{pt["pattern"]}</b> — {pt["signal"]}
<span style='color:#666;float:right;'>{"★"*pt["strength"]}</span>
</div>""", unsafe_allow_html=True)

            with sc2:
                for ob in smc_data["order_blocks"][-3:]:
                    c5 = ob["color"]
                    st.markdown(f"""<div style='background:{c5}11;border:1px solid {c5};
border-radius:6px;padding:8px;margin-bottom:5px;font-size:12px;'>
<b style='color:{c5};'>{ob["type"]}</b>
Zone: Rs.{ob["bottom"]} — Rs.{ob["top"]}
<div style='font-size:10px;color:#888;'>{ob["date"]}</div>
</div>""", unsafe_allow_html=True)

                for fv in smc_data["fvg"][-3:]:
                    c6 = fv["color"]
                    st.markdown(f"""<div style='background:{c6}11;border:1px solid {c6};
border-radius:6px;padding:8px;margin-bottom:5px;font-size:12px;'>
<b style='color:{c6};'>{fv["type"]}</b> Gap: Rs.{fv["gap"]}
</div>""", unsafe_allow_html=True)

        with at2:
            vp = smc_data["volume_profile"]
            if vp:
                vc2 = "#f39c12"
                st.markdown(f"""<div style='background:#161b22;border:2px solid {vc2};
border-radius:10px;padding:14px;'>
<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;text-align:center;'>
  <div><div style='font-size:10px;color:#888;'>POC</div>
  <div style='font-size:18px;font-weight:700;color:#f39c12;'>Rs.{vp["poc"]}</div></div>
  <div><div style='font-size:10px;color:#888;'>VAH</div>
  <div style='font-size:18px;font-weight:700;color:#e74c3c;'>Rs.{vp["vah"]}</div></div>
  <div><div style='font-size:10px;color:#888;'>VAL</div>
  <div style='font-size:18px;font-weight:700;color:#00b880;'>Rs.{vp["val"]}</div></div>
</div>
<div style='margin-top:8px;font-size:13px;font-weight:600;color:{vc2};'>
{vp["position"]}</div>
</div>""", unsafe_allow_html=True)

        # Trade buttons
        st.markdown("---")
        b1, b2, b3 = st.columns(3)

        def do_paper(txn):
            sym = stock.replace(".NS","")
            live_px = data.fetch_live_price(stock) or price
            if txn=="BUY":
                if sym in st.session_state.paper_positions:
                    st.warning("Already holding — sell first"); return
                st.session_state.paper_positions[sym] = {
                    "stock":stock,"price":live_px,"qty":qty,
                    "sl":sl,"target":tgt,"mode":mode_name,
                    "time":str(datetime.datetime.now())[:19]}
                st.session_state.paper_balance -= live_px*qty
                st.session_state.trade_log.append({
                    "time":str(datetime.datetime.now())[:19],"stock":sym,
                    "action":"BUY","price":live_px,"qty":qty,
                    "sl":sl,"target":tgt,"mode":mode_name})
                st.success(f"Paper BUY {sym} Rs.{live_px:.2f}×{qty}")
                fire_alert(f"BUY [{mode_name}]",stock,live_px,qty,sl,tgt,master,order_mode)
            elif txn=="SELL":
                if sym not in st.session_state.paper_positions:
                    st.warning("No position"); return
                pos = st.session_state.paper_positions[sym]
                pnl = (live_px-pos["price"])*pos["qty"]
                st.session_state.paper_balance += live_px*pos["qty"]
                st.session_state.pnl_history.append({
                    "time":str(datetime.datetime.now())[:19],
                    "stock":sym,"pnl":round(pnl,2)})
                del st.session_state.paper_positions[sym]
                st.session_state.trade_log.append({
                    "time":str(datetime.datetime.now())[:19],"stock":sym,
                    "action":"SELL","price":live_px,"pnl":round(pnl,2)})
                st.success(f"Paper SELL {sym} | P&L: Rs.{pnl:+.2f}")

        if b1.button("🚀 BUY", type="primary", use_container_width=True):
            do_paper("BUY")
        if b2.button("🛑 SELL", use_container_width=True):
            do_paper("SELL")
        if b3.button("🔁 Refresh", use_container_width=True):
            st.cache_data.clear(); st.rerun()

        # Open position
        sym = stock.replace(".NS","")
        if sym in st.session_state.paper_positions:
            pos = st.session_state.paper_positions[sym]
            opnl = (price-pos["price"])*pos["qty"]
            pc = "#00b880" if opnl>=0 else "#e74c3c"
            st.markdown(f"""<div style='background:#0d2818;border:1px solid #00b880;
border-radius:8px;padding:10px;margin:8px 0;font-size:12px;'>
Open: {sym} | Entry Rs.{pos["price"]:.2f} | Qty {pos["qty"]}
| SL Rs.{pos["sl"]} | TGT Rs.{pos["target"]}<br>
<b style='color:{pc};font-size:15px;'>Unrealised P&L: Rs.{opnl:+.2f}</b>
</div>""", unsafe_allow_html=True)

        # Chart
        if PLOTLY_OK and len(df) > 20:
            st.markdown("### 📉 Chart")
            cd = df.tail(100).copy()
            cd.index = pd.to_datetime(cd.index)
            lc = mcfg["color"]

            fig = make_subplots(rows=3,cols=1,shared_xaxes=True,
                row_heights=[0.55,0.22,0.23],vertical_spacing=0.02)
            fig.add_trace(go.Candlestick(
                x=cd.index,open=cd["Open"],high=cd["High"],
                low=cd["Low"],close=cd["Close"],name="Price",
                increasing_line_color="#00b880",decreasing_line_color="#e74c3c",
                increasing_fillcolor="#0d2818",decreasing_fillcolor="#2d0a0a"),
                row=1,col=1)
            for ema,ec in [("EMA9","#00e5a0"),("EMA20","#4e8fff"),("EMA50","#a78bfa")]:
                if ema in cd.columns:
                    fig.add_trace(go.Scatter(x=cd.index,y=cd[ema],
                        line=dict(color=ec,width=1),name=ema),row=1,col=1)
            if "VWAP" in cd.columns:
                fig.add_trace(go.Scatter(x=cd.index,y=cd["VWAP"],
                    line=dict(color="#ffa94d",width=1,dash="dot"),name="VWAP"),row=1,col=1)
            fig.add_hline(y=sl,  line_color="#e74c3c",line_dash="dot",row=1,col=1)
            fig.add_hline(y=tgt, line_color="#00b880",line_dash="dot",row=1,col=1)
            fig.add_hline(y=price,line_color="#f39c12",row=1,col=1)

            if "RSI" in cd.columns:
                fig.add_trace(go.Scatter(x=cd.index,y=cd["RSI"],
                    line=dict(color=lc,width=1.5),name="RSI"),row=2,col=1)
                fig.add_hline(y=70,line_color="#e74c3c",line_dash="dot",row=2,col=1)
                fig.add_hline(y=30,line_color="#00b880",line_dash="dot",row=2,col=1)

            if "MACD" in cd.columns:
                hc = ["#00b880" if v>=0 else "#e74c3c"
                      for v in cd.get("MACD_Hist",pd.Series([0]*len(cd)))]
                fig.add_trace(go.Bar(x=cd.index,y=cd.get("MACD_Hist",pd.Series()),
                    marker_color=hc,name="MACD Hist",opacity=0.7),row=3,col=1)
                fig.add_trace(go.Scatter(x=cd.index,y=cd["MACD"],
                    line=dict(color=lc,width=1.5),name="MACD"),row=3,col=1)
                fig.add_trace(go.Scatter(x=cd.index,y=cd.get("MACD_Signal",pd.Series()),
                    line=dict(color="#f39c12",width=1),name="Signal"),row=3,col=1)

            fig.update_layout(height=600,xaxis_rangeslider_visible=False,
                plot_bgcolor="#0d1117",paper_bgcolor="#0d1117",
                font=dict(color="#8b949e",size=9),
                margin=dict(l=0,r=0,t=10,b=0),
                legend=dict(orientation="h",y=1.04,bgcolor="rgba(0,0,0,0)",font_size=9))
            for r in [1,2,3]:
                fig.update_xaxes(gridcolor="#21262d",row=r,col=1)
                fig.update_yaxes(gridcolor="#21262d",row=r,col=1)
            st.plotly_chart(fig,use_container_width=True,
                           config={"displayModeBar":False})

# ── SCANNER ────────────────────────────────────────────────────
elif page == "scanner":
    st.title("🎯 Signal Scanner")
    univ_nm = st.selectbox("Universe", list(config.STOCKS.keys()))
    stocks  = config.STOCKS[univ_nm]
    mode_nm = st.selectbox("Mode", list(config.MODES.keys()))
    mcfg2   = config.MODES[mode_nm]

    if st.button("🔍 Scan", type="primary", use_container_width=True):
        st.cache_data.clear()

    with st.spinner("Scanning..."):
        res = scanner.scan(tuple(stocks), mcfg2["period"], mcfg2["interval"])

    top = [r for r in res if r["Signal"]=="BUY"][:5]
    if top:
        st.markdown("#### 🏆 Top Picks")
        tc = st.columns(min(5,len(top)))
        for i,r in enumerate(top):
            c = "#00b880" if r["Chg%"]>=0 else "#e74c3c"
            tc[i].markdown(f"""<div style='background:#0d2818;
border:2px solid #00b880;border-radius:10px;padding:10px;text-align:center;'>
<div style='font-weight:700;color:#00e5a0;font-size:14px;'>{r["Stock"]}</div>
<div style='font-size:15px;color:#e6edf3;'>Rs.{r["Price"]:,.1f}</div>
<div style='font-size:11px;color:{c};'>{"▲" if r["Chg%"]>=0 else "▼"}{abs(r["Chg%"]):.2f}%</div>
<div style='font-size:10px;color:#00b880;'>{r["Score"]}/7</div>
</div>""", unsafe_allow_html=True)

    disp = pd.DataFrame(res)[["Stock","Price","Chg%","RSI","ADX","MFI","Vol","Score","Signal","ST"]]
    disp["Score"] = disp["Score"].apply(lambda x:f"{x}/7")
    st.dataframe(disp, hide_index=True, use_container_width=True, height=400)

# ── BACKTEST ───────────────────────────────────────────────────
elif page == "backtest":
    st.title("📈 Professional Backtest")

    c1,c2,c3,c4 = st.columns(4)
    bt_stk  = c1.selectbox("Stock", config.STOCKS["⭐ Nifty 50 Top 20"],
                            format_func=lambda x:x.replace(".NS",""))
    bt_per  = c2.selectbox("Period", ["3mo","6mo","1y","2y","3y"],index=2)
    bt_cap  = c3.number_input("Capital", 50000, 5000000, 500000, 50000)
    bt_str  = c4.selectbox("Strategy",
                           ["master","trend","momentum","mean_revert"],
                           format_func=str.title)
    c5,c6,c7 = st.columns(3)
    bt_sl   = c5.slider("SL ATR mult", 1.0, 3.0, 1.5, 0.1)
    bt_rr   = c6.slider("Target R:R",  1.0, 5.0, 2.0, 0.1)
    bt_risk = c7.slider("Risk %",      0.5, 3.0, 1.5, 0.5)

    if st.button("▶ Run Backtest", type="primary", use_container_width=True):
        with st.spinner("Running professional backtest..."):
            raw = data.fetch_ohlcv(bt_stk, bt_per, "1d")
            if not raw.empty:
                df_bt = indicators.compute(raw.to_json())
                res   = backtest.run(df_bt, bt_str, bt_cap, bt_sl, bt_rr, bt_risk)
                st.session_state.backtest_result = res

    res = st.session_state.backtest_result
    if res:
        ret_c = "#00b880" if res["return_pct"]>=0 else "#e74c3c"
        al    = res["return_pct"] - res["bh_return"]
        al_c  = "#00b880" if al>=0 else "#e74c3c"

        mc = st.columns(8)
        for col,(lbl,val,cc) in zip(mc,[
            ("Return",  f"{res['return_pct']:+.1f}%", ret_c),
            ("vs B&H",  f"{al:+.1f}%",                al_c),
            ("Win%",    f"{res['win_rate']:.0f}%",     "#f39c12"),
            ("Trades",  str(res["total_trades"]),       "#a78bfa"),
            ("PF",      f"{res['profit_factor']:.2f}", "#4e8fff"),
            ("Max DD",  f"{res['max_drawdown']:.1f}%", "#e74c3c"),
            ("Sharpe",  f"{res['sharpe']:.2f}",        "#00e5a0"),
            ("Sortino", f"{res['sortino']:.2f}",       "#ffa94d"),
        ]):
            col.markdown(f"""<div style='background:#161b22;border:1px solid #21262d;
border-radius:8px;padding:8px;text-align:center;'>
<div style='font-size:16px;font-weight:700;color:{cc};'>{val}</div>
<div style='font-size:9px;color:#888;'>{lbl}</div></div>""",unsafe_allow_html=True)

        st.markdown("")
        m2 = st.columns(4)
        m2[0].metric("Expectancy/Trade", f"Rs.{res['expectancy']:.2f}")
        m2[1].metric("Calmar Ratio",     f"{res['calmar']:.2f}")
        m2[2].metric("Max Consec Wins",  str(res["max_consec_wins"]))
        m2[3].metric("Max Consec Loss",  str(res["max_consec_loss"]))

        if res["profit_factor"]>=2 and res["win_rate"]>=55:
            st.success(f"EXCELLENT — PF {res['profit_factor']} | WR {res['win_rate']}% | Sharpe {res['sharpe']}")
        elif res["profit_factor"]>=1.5:
            st.info(f"GOOD — PF {res['profit_factor']} | WR {res['win_rate']}%")
        elif res["profit_factor"]>=1:
            st.warning(f"MARGINAL — PF {res['profit_factor']:.2f}")
        else:
            st.error(f"LOSING — Do NOT use real money. PF {res['profit_factor']:.2f}")

        if PLOTLY_OK and len(res.get("equity",[])) > 5:
            eq_fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                row_heights=[0.65,0.35],vertical_spacing=0.04)
            xr = list(range(len(res["equity"])))
            eq_fig.add_trace(go.Scatter(x=xr,y=res["equity"],
                line=dict(color="#00e5a0",width=2),name="Strategy",
                fill="tozeroy",fillcolor="rgba(0,229,160,0.06)"),row=1,col=1)
            bh_eq = [bt_cap*(1+res["bh_return"]/100*j/max(len(xr)-1,1))
                     for j in range(len(xr))]
            eq_fig.add_trace(go.Scatter(x=xr,y=bh_eq,
                line=dict(color="#4e8fff",width=1.5,dash="dot"),
                name=f"Buy&Hold {res['bh_return']:+.1f}%"),row=1,col=1)
            eq_arr = np.array(res["equity"])
            pk = np.maximum.accumulate(eq_arr)
            dd = (eq_arr-pk)/(pk+1e-9)*100
            eq_fig.add_trace(go.Scatter(x=xr,y=dd,fill="tozeroy",
                fillcolor="rgba(231,76,60,0.15)",
                line=dict(color="#e74c3c",width=1),name="Drawdown%"),row=2,col=1)
            eq_fig.update_layout(height=380,plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117",font=dict(color="#8b949e",size=9),
                legend=dict(orientation="h",y=1.04,bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=0,r=0,t=10,b=0))
            for r2 in [1,2]:
                eq_fig.update_xaxes(gridcolor="#21262d",row=r2,col=1)
                eq_fig.update_yaxes(gridcolor="#21262d",row=r2,col=1)
            st.plotly_chart(eq_fig,use_container_width=True,
                           config={"displayModeBar":False})

            if res.get("monthly_rets") and len(res["monthly_rets"])>1:
                mo_c = ["#00b880" if v>=0 else "#e74c3c"
                        for v in res["monthly_rets"]]
                mf = go.Figure(go.Bar(x=res["months"],y=res["monthly_rets"],
                    marker_color=mo_c))
                mf.update_layout(title="Monthly Returns %",height=200,
                    plot_bgcolor="#0d1117",paper_bgcolor="#0d1117",
                    font=dict(color="#8b949e"),margin=dict(l=0,r=0,t=30,b=0))
                mf.add_hline(y=0,line_color="#555")
                st.plotly_chart(mf,use_container_width=True,
                               config={"displayModeBar":False})

        # Trade Replay
        cl = res.get("closed",[])
        if cl:
            st.markdown("**Trade Replay:**")
            ri = st.slider("Trade #", 1, max(len(cl),1), max(len(cl),1),
                           key="bt_replay")
            rt = cl[min(ri-1,len(cl)-1)]
            rc = st.columns(5)
            rp = rt.get("pnl",0); rpc = "#00b880" if rp>=0 else "#e74c3c"
            for col2,(lbl2,val2,cc2) in zip(rc,[
                ("Date",    rt.get("date",""),           "#ccc"),
                ("Entry",   f"Rs.{rt.get('price',0):.2f}", "#4e8fff"),
                ("Exit",    f"Rs.{rt.get('price',0):.2f}", "#f39c12"),
                ("P&L",     f"Rs.{rp:+,.0f}",             rpc),
                ("Reason",  rt.get("reason",""),          "#a78bfa"),
            ]):
                col2.markdown(f"""<div style='text-align:center;
background:#161b22;border-radius:6px;padding:8px;'>
<div style='font-size:10px;color:#888;'>{lbl2}</div>
<div style='font-size:13px;font-weight:600;color:{cc2};'>{val2}</div>
</div>""", unsafe_allow_html=True)

            # P&L bar chart
            if PLOTLY_OK:
                pnls = [t["pnl"] for t in cl]
                cumul = [sum(pnls[:i+1]) for i in range(len(pnls))]
                bc = ["#00b880" if p>=0 else "#e74c3c" for p in pnls]
                rf = go.Figure()
                rf.add_trace(go.Bar(x=list(range(1,len(cl)+1)),y=pnls,
                    marker_color=bc,name="P&L",opacity=0.75))
                rf.add_trace(go.Scatter(x=list(range(1,len(cl)+1)),y=cumul,
                    line=dict(color="#f39c12",width=2),
                    name="Cumulative",yaxis="y2"))
                rf.add_vline(x=ri,line_color="#fff",line_dash="dot",line_width=1)
                rf.update_layout(height=220,plot_bgcolor="#0d1117",
                    paper_bgcolor="#0d1117",font=dict(color="#8b949e",size=9),
                    yaxis2=dict(overlaying="y",side="right",gridcolor="#21262d"),
                    legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",y=1.08),
                    margin=dict(l=0,r=0,t=10,b=0))
                rf.update_xaxes(gridcolor="#21262d",title="Trade #")
                rf.update_yaxes(gridcolor="#21262d")
                rf.add_hline(y=0,line_color="#555")
                st.plotly_chart(rf,use_container_width=True,
                               config={"displayModeBar":False})

# ── TRADE LOG ──────────────────────────────────────────────────
elif page == "tradelog":
    st.title("📋 Trade Log")
    log = st.session_state.trade_log
    pnl_hist = st.session_state.pnl_history
    if log:
        st.dataframe(pd.DataFrame(log), hide_index=True, use_container_width=True)
        if pnl_hist and len(pnl_hist)>1 and PLOTLY_OK:
            pdf = pd.DataFrame(pnl_hist)
            pdf["Cumulative"] = pdf["pnl"].cumsum()
            fig_pnl = go.Figure(go.Scatter(y=pdf["Cumulative"],
                line=dict(color="#00e5a0",width=2),fill="tozeroy",
                fillcolor="rgba(0,229,160,0.06)"))
            fig_pnl.update_layout(height=250,title="Cumulative P&L",
                plot_bgcolor="#0d1117",paper_bgcolor="#0d1117",
                font=dict(color="#8b949e"),margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_pnl,use_container_width=True,
                           config={"displayModeBar":False})
        if st.button("Clear All"):
            st.session_state.trade_log=[]
            st.session_state.pnl_history=[]
            st.rerun()
    else:
        st.info("No trades yet")

# ── ADMIN ──────────────────────────────────────────────────────
elif page == "admin" and udata.get("role")=="admin":
    st.title("🛠 Admin Portal")

    all_u = auth.get_all_users()
    active = [u for u in all_u if u["status"]=="active"]
    pending= [u for u in all_u if u["status"]=="pending"]
    paid   = [u for u in all_u if u.get("plan","Free")!="Free"]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Users", len(all_u))
    c2.metric("Active",      len(active))
    c3.metric("Pending",     len(pending))
    c4.metric("Paid",        len(paid))

    tab1,tab2,tab3 = st.tabs(["Add User","All Users","Dashboard"])

    with tab1:
        st.subheader("Add New User")
        with st.form("add_user"):
            f1,f2 = st.columns(2)
            a_user = f1.text_input("Username *")
            a_pass = f2.text_input("Password *", type="password")
            f3,f4 = st.columns(2)
            a_plan = f3.selectbox("Plan",["Free","Basic","Premium"])
            a_days = f4.number_input("Days", 1, 365, 30)
            f5,f6 = st.columns(2)
            a_email= f5.text_input("Email")
            a_phone= f6.text_input("Phone")
            a_txn  = st.text_input("UPI Txn ID")
            sub = st.form_submit_button("Create Account", type="primary",
                                        use_container_width=True)
        if sub:
            if not a_user or not a_pass:
                st.error("Username and password required")
            else:
                res2 = auth.create_user_admin(a_user,a_pass,a_plan,
                                              int(a_days),a_email,a_phone,a_txn)
                if res2["ok"]:
                    st.success(f"User '{a_user}' created! Expiry: {res2['expiry']}")
                    if a_phone:
                        import urllib.parse
                        ph = a_phone.strip().replace(" ","")
                        if not ph.startswith("+"): ph="+91"+ph.lstrip("0")
                        txt = urllib.parse.quote(
                            f"AI Trading PRO+ Account Active!\n"
                            f"Username: {a_user}\nPassword: {a_pass}\n"
                            f"Plan: {a_plan} | Valid: {res2['expiry']}\n"
                            f"App: {config.APP_URL}")
                        st.markdown(
                            f"<a href='https://wa.me/{ph.replace('+','')}?text={txt}'"
                            f" target='_blank' style='background:#25d366;color:#000;"
                            f"padding:8px 16px;border-radius:6px;font-weight:700;"
                            f"text-decoration:none;'>Send on WhatsApp</a>",
                            unsafe_allow_html=True)
                else:
                    st.error(res2["error"])

    with tab2:
        st.subheader(f"All Users ({len(all_u)})")
        search = st.text_input("Search")
        for usr in all_u:
            if search and search.lower() not in (usr["username"]+usr.get("email","")).lower():
                continue
            with st.expander(f"{'✅' if usr['status']=='active' else '⏳'} "
                             f"{usr['username']} | {usr.get('plan','Free')} | {usr.get('expiry','')}"):
                st.caption(f"Email: {usr.get('email','—')} | Phone: {usr.get('phone','—')}")
                ec = st.columns(4)
                new_plan = ec[0].selectbox("Plan",["Free","Basic","Premium"],
                    index=["Free","Basic","Premium"].index(usr.get("plan","Free")),
                    key=f"pl_{usr['username']}")
                ext = ec[1].number_input("Extend days",1,365,30,key=f"ex_{usr['username']}")
                if ec[2].button("Update",key=f"upd_{usr['username']}"):
                    import datetime as dt2
                    try:
                        cur_e = dt2.datetime.strptime(
                            usr.get("expiry","2099-12-31"),"%Y-%m-%d").date()
                        new_e = str(max(cur_e,dt2.date.today())+dt2.timedelta(days=int(ext)))
                    except Exception:
                        new_e = str(dt2.date.today()+dt2.timedelta(days=int(ext)))
                    auth.update_user(usr["username"],plan=new_plan,
                                     status="active",expiry=new_e)
                    st.success(f"Updated {usr['username']}"); st.rerun()
                if ec[3].button("Delete",key=f"del_{usr['username']}"):
                    auth.delete_user(usr["username"]); st.rerun()

    with tab3:
        st.subheader("Registration Dashboard")
        import datetime as dt3
        today = str(dt3.date.today())
        week  = str(dt3.date.today()-dt3.timedelta(days=7))
        td_cnt= sum(1 for u3 in all_u if u3.get("joined","")==today)
        wk_cnt= sum(1 for u3 in all_u if u3.get("joined","")>=week)
        d1,d2,d3,d4 = st.columns(4)
        d1.metric("Today",    td_cnt)
        d2.metric("This Week",wk_cnt)
        d3.metric("Paid",     len(paid))
        d4.metric("Free",     len(all_u)-len(paid))

elif page == "portfolio":
    st.title("💼 Portfolio")
    pos = st.session_state.paper_positions
    if pos:
        for sym2, p2 in pos.items():
            live2 = data.fetch_live_price(p2["stock"]) or p2["price"]
            opnl2 = (live2-p2["price"])*p2["qty"]
            c2 = "#00b880" if opnl2>=0 else "#e74c3c"
            st.markdown(f"""<div style='background:#161b22;border:1px solid #21262d;
border-radius:8px;padding:12px;margin-bottom:8px;'>
<b>{sym2}</b> | Entry Rs.{p2["price"]:.2f} | Qty {p2["qty"]}
| Mode {p2.get("mode","—")}<br>
Current Rs.{live2:.2f} | <b style='color:{c2};'>P&L Rs.{opnl2:+.2f}</b>
</div>""", unsafe_allow_html=True)
    else:
        st.info("No open positions")

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div style='text-align:center;font-size:10px;color:#555;'>
AI Trading PRO+ v5.0 | Modular Architecture |
XGBoost+LightGBM+LSTM | Professional Backtesting | Paper Trading Only
</div>""", unsafe_allow_html=True)
