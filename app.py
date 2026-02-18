import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="TSX Intraday Scanner", layout="wide")

DEFAULT_TICKERS = [
    "RY.TO", "TD.TO", "BNS.TO", "ENB.TO", "SHOP.TO",
    "CNQ.TO", "SU.TO", "BCE.TO", "TRP.TO", "MFC.TO"
]

OPENING_RANGE_MINUTES = 15


# ---------------------------------------------------------
# DATA FUNCTIONS
# ---------------------------------------------------------
def get_intraday_data(tickers, interval="1m", period="1d"):
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, interval=interval, period=period, progress=False)
            if not df.empty:
                df = df.dropna()
                data[t] = df
        except Exception:
            continue
    return data


def compute_vwap(df):
    df = df.copy()
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df


def compute_emas(df):
    df = df.copy()
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    return df


def get_opening_range(df):
    start = df.index[0]
    end = start + dt.timedelta(minutes=OPENING_RANGE_MINUTES)
    or_df = df[(df.index >= start) & (df.index < end)]
    if or_df.empty:
        return None, None
    return or_df["High"].max(), or_df["Low"].min()


# ---------------------------------------------------------
# SCORING + BUY/SELL LOGIC
# ---------------------------------------------------------
def score_stock(ticker, df):
    if df.empty:
        return None

    df = compute_vwap(df)
    df = compute_emas(df)

    last = df.iloc[-1]

    close = float(last["Close"])
    vwap = float(last["VWAP"])
    ema9 = float(last["EMA_9"])
    ema20 = float(last["EMA_20"])

    orh, orl = get_opening_range(df)

    score = 0
    reasons = []

    # Above VWAP
    if close > vwap:
        score += 1
        reasons.append("Above VWAP")

    # Trend
    if ema9 > ema20:
        score += 1
        reasons.append("EMA9 > EMA20 (uptrend)")

    # Opening range breakout
    if orh is not None:
        if close > orh:
            score += 2
            reasons.append("Opening range breakout")
        elif (orh - close) / orh < 0.003:
            score += 1
            reasons.append("Near ORH")

    # Volume spike
    if len(df) > 20:
        avg_vol = df["Volume"].tail(20).mean()
        if last["Volume"] > avg_vol * 1.5:
            score += 1
            reasons.append("Volume spike")

    # BUY PRICE = current price
    buy_price = close

    # SELL PRICE = simple target (0.3% above)
    sell_price = round(close * 1.003, 4)

    # PERFORMANCE SCORE = score + momentum factor
    momentum = ema9 - ema20
    performance = score + momentum

    return {
        "ticker": ticker,
        "score": score,
        "performance": float(performance),
        "buy_price": float(buy_price),
        "sell_price": float(sell_price),
        "vwap": vwap,
        "ema9": ema9,
        "ema20": ema20,
        "orh": float(orh) if orh else None,
        "orl": float(orl) if orl else None,
        "reasons": reasons,
    }


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("🇨🇦 TSX Intraday Scanner — Auto Refresh")

tickers_input = st.text_input(
    "TSX tickers (comma-separated):",
    value=", ".join(DEFAULT_TICKERS),
)

tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

refresh_interval = st.slider("Auto-refresh interval (seconds)", 30, 300, 60)

# SAFE auto-refresh
st_autorefresh(interval=refresh_interval * 1000, key="refresh")


# ---------------------------------------------------------
# MAIN SCAN
# ---------------------------------------------------------
def run_scan():
    with st.spinner("Scanning TSX stocks..."):
        data = get_intraday_data(tickers)
        rows = []

        for t, df in data.items():
            res = score_stock(t, df)
            if res:
                rows.append(res)

        if not rows:
            st.warning("No data found.")
            return

        df_scores = pd.DataFrame(rows)

        # Sort by performance
        df_scores = df_scores.sort_values("performance", ascending=False)

        st.subheader("📊 Ranked Intraday Opportunities")
        st.dataframe(
            df_scores[
                ["ticker", "performance", "score", "buy_price", "sell_price", "vwap", "ema9", "ema20", "orh", "orl"]
            ],
            use_container_width=True,
        )

        st.subheader("Details")
        for _, row in df_scores.iterrows():
            with st.expander(f"{row['ticker']} — Score {row['score']}"):
                st.write("**Reasons:**")
                for r in row["reasons"]:
                    st.write(f"- {r}")


run_scan()
