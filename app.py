
import streamlit as st
from services.market_data import fetch_stock_data
from core.indicators import compute_indicators
from core.smc import detect_market_structure
from core.risk import calculate_position_size

st.set_page_config(page_title="AI Trading PRO+")

st.title("AI Trading PRO+ Refactored")

symbol = st.text_input("Symbol", "RELIANCE.NS")

if st.button("Analyze"):
    df = fetch_stock_data(symbol)

    if len(df) > 0:
        df = compute_indicators(df)

        trend = detect_market_structure(df)

        qty = calculate_position_size(100000,1,10)

        st.success(f"Trend: {trend['trend']}")
        st.write(f"Suggested Qty: {qty}")
        st.dataframe(df.tail())
