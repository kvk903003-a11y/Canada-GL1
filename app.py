import datetime as dt

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="TSX Intraday Scanner", layout="wide")

DEFAULT_TICKERS = [
    "RY.TO", "TD.TO", "BNS.TO", "ENB.TO", "SHOP.TO",
    "CNQ.TO", "SU.TO", "BCE.TO", "TRP.TO", "MFC.TO"
]

OPENING_RANGE_MINUTES = 15

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_intraday_data(tickers, interval="1m", period="1d"):
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, interval=interval, period=period, progress=False)
            if not df.empty:
                df.dropna(inplace=True)
                data[t] = df
        except Exception:
            continue
    return data

def compute_vwap(df):
    df = df.copy()
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df

def compute_emas(df, spans=(9, 20)):
    df = df.copy()
    for span in spans:
        df[f"EMA_{span}"] = df["Close"].ewm(span=span, adjust=False).mean()
    return df

def get_opening_range(df, opening_minutes=15):
    df_local = df.copy()
    try:
        df_local["Time"] = df_local.index.tz_convert("America/Toronto").time
    except Exception:
        df_local["Time"] = df_local.index.time

    start_dt = df_local.index[0]
    end_dt = start_dt + dt.timedelta(minutes=opening_minutes)
    or_df = df_local[(df_local.index >= start_dt) & (df_local.index < end_dt)]
    if or_df.empty:
        return None, None
    return or_df["High"].max(), or_df["Low"].min()

def score_stock(ticker, df):
    if df.empty:
        return None

    df = compute_vwap(df)
    df = compute_emas(df)

    last = df.iloc[-1]
    orh, orl = get_opening_range(df)

    score = 0
    reasons = []

    # Above VWAP
    if last["Close"] > last["VWAP"]:
        score += 1
        reasons.append("Above VWAP (bullish bias)")

    # Trend: EMA 9 > EMA 20
    if last["EMA_9"] > last["EMA_20"]:
        score += 1
        reasons.append("Short-term uptrend (EMA 9 > EMA 20)")

    # Opening range logic
    if orh is not None:
        if last["Close"] > orh and last["Close"] > last["Open"]:
            score += 2
            reasons.append("Opening range breakout")
        elif (orh - last["Close"]) / orh < 0.003:
            score += 1
            reasons.append("Near opening range high")

    # Volume spike
    if len(df) > 20:
        avg_vol = df["Volume"].tail(20).mean()
        if last["Volume"] > 1.5 * avg_vol:
            score += 1
            reasons.append("Volume spike")

    return {
        "ticker": ticker,
        "score": score,
        "price": float(last["Close"]),
        "vwap": float(last["VWAP"]),
        "ema9": float(last["EMA_9"]),
        "ema20": float(last["EMA_20"]),
        "orh": float(orh) if orh is not None else None,
        "orl": float(orl) if orl is not None else None,
        "reasons": reasons,
    }

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🇨🇦 TSX Intraday Scanner (Auto‑Refresh)")

st.markdown(
    """
Scans selected **TSX stocks** and ranks them by a simple intraday score based on:

- Opening range breakout
- VWAP bias
- Short-term trend (EMA 9 vs EMA 20)
- Volume spike
"""
)

tickers_input = st.text_input(
    "TSX tickers (comma-separated, with .TO suffix):",
    value=", ".join(DEFAULT_TICKERS),
)

tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

refresh_interval = st.slider("Auto-refresh interval (seconds)", 30, 300, 60)

# True timed auto-refresh
st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")

if st.button("Refresh now"):
    st.experimental_rerun()

st.info("Leave this tab open; the table will refresh automatically.")

def run_scan():
    with st.spinner("Fetching data and scanning..."):
        data = get_intraday_data(tickers)
        rows = []
        for t, df in data.items():
            res = score_stock(t, df)
            if res is not None:
                rows.append(res)

        if not rows:
            st.warning("No data available for the selected tickers.")
            return

        df_scores = pd.DataFrame(rows)
        df_scores.sort_values("score", ascending=False, inplace=True)

        st.subheader("Ranked Intraday Candidates")
        st.dataframe(
            df_scores[["ticker", "score", "price", "vwap", "ema9", "ema20", "orh", "orl"]],
            use_container_width=True,
        )

        st.subheader("Details")
        for _, row in df_scores.iterrows():
            with st.expander(f"{row['ticker']} (Score: {row['score']})"):
                st.write(f"**Price:** {row['price']:.2f}")
                st.write(f"**VWAP:** {row['vwap']:.2f}")
                st.write(f"**EMA 9:** {row['ema9']:.2f}")
                st.write(f"**EMA 20:** {row['ema20']:.2f}")
                st.write(f"**ORH:** {row['orh']}")
                st.write(f"**ORL:** {row['orl']}")
                st.write("**Reasons:**")
                for r in row["reasons"]:
                    st.write(f"- {r}")

run_scan()
