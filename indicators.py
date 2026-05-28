
import streamlit as st

@st.cache_data(ttl=300)
def compute_indicators(df):
    df = df.copy()

    for span in [9,20,50,200]:
        df[f"EMA{span}"] = df["Close"].ewm(span=span).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

    df["RSI"] = 100 - (100 / (1 + gain / loss))
    return df
