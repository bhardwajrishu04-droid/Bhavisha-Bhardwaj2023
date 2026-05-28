
import yfinance as yf
import streamlit as st

@st.cache_data(ttl=120)
def fetch_stock_data(symbol, period="1mo", interval="1d"):
    return yf.download(symbol, period=period, interval=interval)
