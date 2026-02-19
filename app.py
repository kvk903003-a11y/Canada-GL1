import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="US Stock Scanner (Reliable)", layout="wide")

DEFAULT_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META",
    "GOOGL","TSLA","AMD","NFLX","AVGO",
    "JPM","BAC","XOM","CVX","UNH",
    "LLY","V","MA","COST","PEP"
]

def fetch_data(ticker, interval="5m"):
    """
    Try intraday first.
    If empty, fall back to daily candles.
    This guarantees data is always returned.
    """
    try:
        df = yf.download(ticker, period="1d", interval=interval, progress=False)
        if df is not None and not df.empty:
            df = df.reset_index()
            df.rename(columns={
                "Datetime": "t",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }, inplace=True)
            df = df.drop_duplicates(subset="t", keep="last")
            return df
    except:
        pass

    # FALLBACK: daily candles (always available)
    df = yf.download(ticker, period="5d", interval="1d", progress=False)
    if df is None or df.empty:
        return None

    df = df.reset_index()
    df.rename(columns={
        "Date": "t",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)
    df = df.drop_duplicates(subset="t", keep="last")
    return df

def compute_indicators(df):
    if df is None or len(df) < 5:
        return None

    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    return df

def score(df):
    latest = df.iloc[-1]
    s = 0
    if latest["close"] > latest["ema9"]:
        s += 1
    if latest["close"] > latest["ema20"]:
        s += 1
    if latest["close"] > latest["vwap"]:
        s += 1
    return s

def process_ticker(ticker, interval="5m"):
    df = fetch_data(ticker, interval=interval)
    df = compute_indicators(df)
    if df is None:
        return None

    latest = df.iloc[-1]
    return {
        "ticker": ticker,
        "score": score(df),
        "price": latest["close"],
        "ema9": latest["ema9"],
        "ema20": latest["ema20"],
        "vwap": latest["vwap"],
        "time": latest["t"]
    }

def run_scan(tickers, interval="5m"):
    results = []
    for t in tickers:
        r = process_ticker(t, interval=interval)
        if r:
            results.append(r)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)
    return df

st.title("US Stock Scanner (Reliable Yahoo Version)")

tickers_input = st.text_area(
    "Tickers (comma-separated):",
    value=",".join(DEFAULT_TICKERS)
)

interval = st.selectbox("Interval", ["1m","2m","5m","15m","30m","60m"], index=2)

if st.button("Run Scan"):
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    df = run_scan(tickers, interval=interval)

    if df.empty:
        st.error("Still no data — Streamlit Cloud may be blocking Yahoo Finance.")
    else:
        st.dataframe(df.reset_index(drop=True))
