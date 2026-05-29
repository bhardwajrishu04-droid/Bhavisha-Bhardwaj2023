# modules/ai_engine.py
# =============================================================
# Advanced AI Engine
# Models: XGBoost + LightGBM + GradientBoosting + RF + AdaBoost
# Walk-Forward Validation (5 folds)
# LSTM Price Forecast (5 candles)
# =============================================================
import numpy as np
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, AdaBoostClassifier
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

try:
    import xgboost as xgb; XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    import lightgbm as lgb; LGB_OK = True
except ImportError:
    LGB_OK = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_OK = True
except ImportError:
    KERAS_OK = False


FEAT_COLS = [
    "EMA9","EMA20","EMA50","RSI","MACD","MACD_Hist",
    "Stoch_K","BB_Pct","BB_Width","Vol_Ratio",
    "Return_1","Return_3","Price_Pos",
    "RSI_MA","Stoch_D","ADX","MFI","CCI","Williams_R","OBV"
]


def predict(df: pd.DataFrame) -> dict:
    """
    Full ensemble prediction.
    Returns:
        prob       — float 0-1 (bullish probability)
        accuracy   — validation accuracy %
        confidence — High/Medium/Low
        model_name — model description
        wf_results — list of fold accuracies
        feature_importance — dict
        lstm_forecast — list of 5 prices (or None)
    """
    result = {
        "prob": 0.5, "accuracy": 0.0,
        "confidence": "Low", "model_name": "Default (no data)",
        "wf_results": [], "feature_importance": {},
        "lstm_forecast": None, "models_used": [],
    }

    # ── Build feature matrix ─────────────────────────────────
    feat_cols = [c for c in FEAT_COLS if c in df.columns]
    if len(feat_cols) < 5 or len(df) < 30:
        return result

    fd_raw = df[feat_cols].ffill().fillna(0).iloc[20:]
    if len(fd_raw) < 30:
        return result

    df_local = df.copy()
    df_local["T3"] = (df_local["Close"].shift(-3) > df_local["Close"]*1.005).astype(int)
    df_local["T1"] = (df_local["Close"].shift(-1) > df_local["Close"]*1.002).astype(int)
    target_col = "T3" if len(fd_raw) >= 60 else "T1"

    td = df_local[target_col].loc[fd_raw.index].fillna(0).iloc[:-3]
    fd = fd_raw.loc[td.index]
    if len(fd) < 20: return result

    scaler = StandardScaler()
    X = scaler.fit_transform(fd.values)
    y = td.values

    # ── Walk-Forward Validation ──────────────────────────────
    n_folds = min(5, max(3, len(X)//20))
    tscv    = TimeSeriesSplit(n_splits=n_folds)
    fold_accs = []
    for ftr, fval in tscv.split(X):
        if len(set(y[ftr])) < 2 or len(fval) < 2: continue
        try:
            _rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
            _rf.fit(X[ftr], y[ftr])
            fold_accs.append(round(accuracy_score(y[fval],_rf.predict(X[fval]))*100,1))
        except Exception:
            continue
    result["wf_results"] = fold_accs

    # ── Final split ─────────────────────────────────────────
    splits = list(tscv.split(X))
    if not splits: return result
    tr_idx, val_idx = splits[-1]
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    if len(set(y_tr)) < 2 or len(X_val) < 2: return result

    model_probs = []; model_labels = []; all_imps = []

    # ── XGBoost ─────────────────────────────────────────────
    if XGB_OK:
        try:
            _n = min(200, max(50, len(X_tr)))
            m = xgb.XGBClassifier(
                n_estimators=_n, max_depth=4, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42, verbosity=0)
            m.fit(X_tr, y_tr, eval_set=[(X_val,y_val)], verbose=False)
            model_probs.append(m.predict_proba(X[-1:])[0][1])
            acc = accuracy_score(y_val,m.predict(X_val))*100
            model_labels.append(f"XGB {acc:.0f}%")
            all_imps.append(dict(zip(feat_cols, m.feature_importances_)))
            result["models_used"].append("XGBoost")
        except Exception: pass

    # ── LightGBM ────────────────────────────────────────────
    if LGB_OK:
        try:
            _n = min(200, max(50, len(X_tr)))
            m = lgb.LGBMClassifier(
                n_estimators=_n, num_leaves=15, learning_rate=0.08,
                subsample=0.8, min_child_samples=3,
                random_state=42, verbose=-1)
            m.fit(X_tr, y_tr, eval_set=[(X_val,y_val)],
                  callbacks=[lgb.early_stopping(20,verbose=False),
                             lgb.log_evaluation(-1)])
            model_probs.append(m.predict_proba(X[-1:])[0][1])
            acc = accuracy_score(y_val,m.predict(X_val))*100
            model_labels.append(f"LGB {acc:.0f}%")
            _fi = m.feature_importances_/(m.feature_importances_.sum()+1e-9)
            all_imps.append(dict(zip(feat_cols, _fi)))
            result["models_used"].append("LightGBM")
        except Exception: pass

    # ── Gradient Boosting ───────────────────────────────────
    try:
        m = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42)
        m.fit(X_tr, y_tr)
        model_probs.append(m.predict_proba(X[-1:])[0][1])
        acc = accuracy_score(y_val,m.predict(X_val))*100
        model_labels.append(f"GB {acc:.0f}%")
        all_imps.append(dict(zip(feat_cols, m.feature_importances_)))
        result["models_used"].append("GradBoost")
    except Exception: pass

    # ── RandomForest ────────────────────────────────────────
    try:
        m = RandomForestClassifier(
            n_estimators=200, max_depth=6,
            min_samples_leaf=3, random_state=42)
        m.fit(X_tr, y_tr)
        model_probs.append(m.predict_proba(X[-1:])[0][1])
        acc = accuracy_score(y_val,m.predict(X_val))*100
        model_labels.append(f"RF {acc:.0f}%")
        all_imps.append(dict(zip(feat_cols, m.feature_importances_)))
        result["models_used"].append("RandomForest")
    except Exception: pass

    # ── AdaBoost ────────────────────────────────────────────
    try:
        m = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        m.fit(X_tr, y_tr)
        model_probs.append(m.predict_proba(X[-1:])[0][1])
        acc = accuracy_score(y_val,m.predict(X_val))*100
        model_labels.append(f"Ada {acc:.0f}%")
        result["models_used"].append("AdaBoost")
    except Exception: pass

    # ── Ensemble average ────────────────────────────────────
    if model_probs:
        result["prob"] = float(np.mean(model_probs))
        result["accuracy"] = round(float(np.mean([
            float(l.split()[-1].replace("%",""))
            for l in model_labels])), 1)
        result["model_name"] = (
            "+".join(result["models_used"]) +
            f" ({len(model_probs)} models | acc {result['accuracy']}%)")
        if all_imps:
            combined = {}
            for d in all_imps:
                for k,v in d.items(): combined[k] = combined.get(k,0)+v/len(all_imps)
            result["feature_importance"] = dict(
                sorted(combined.items(),key=lambda x:-x[1])[:8])

    acc = result["accuracy"]
    result["confidence"] = "High" if acc>=65 else ("Medium" if acc>=55 else "Low")

    # ── LSTM Forecast ────────────────────────────────────────
    if KERAS_OK and len(df) >= 30:
        result["lstm_forecast"] = _lstm_forecast(df)

    return result


def _lstm_forecast(df: pd.DataFrame, seq_len=20, steps=5):
    try:
        close = df["Close"].values.reshape(-1,1)
        sc    = MinMaxScaler()
        scaled= sc.fit_transform(close)
        Xs, ys = [], []
        for i in range(seq_len, len(scaled)-1):
            Xs.append(scaled[i-seq_len:i])
            ys.append(scaled[i])
        Xs, ys = np.array(Xs), np.array(ys)
        if len(Xs) < 20: return None

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_len,1)),
            Dropout(0.2),
            LSTM(30),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        es = EarlyStopping(patience=3, restore_best_weights=True)
        model.fit(Xs[:-5], ys[:-5], epochs=15, batch_size=16,
                  validation_split=0.1, callbacks=[es], verbose=0)

        seq = scaled[-seq_len:].reshape(1,seq_len,1)
        preds_sc = []
        for _ in range(steps):
            p = model.predict(seq, verbose=0)[0][0]
            preds_sc.append(p)
            seq = np.append(seq[:,1:,:], [[[p]]], axis=1)

        return sc.inverse_transform(
            np.array(preds_sc).reshape(-1,1)).flatten().tolist()
    except Exception:
        return None
