import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="US Stock Scanner (Yahoo Finance)", layout="wide")

# You can expand this list or load from a file
DEFAULT_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META",
    "GOOGL","TSLA","AMD","NFLX","AVGO",
    "JPM","BAC","XOM","CVX","UNH",
    "LLY","V","MA","COST","PEP"
]

def get_intraday_yf(ticker, period="1d", interval="5m"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        df = df.reset_index()
        df.rename(columns={
            "Datetime": "t",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)
        return df
    except Exception:
        return None

def compute_indicators(df):
    if df is None or len(df) < 20:
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

def process_ticker(ticker, period="1d", interval="5m"):
    df = get_intraday_yf(ticker, period=period, interval=interval)
    df = compute_indicators(df)
    if df is None:
        return None
    s = score(df)
    latest = df.iloc[-1]
    return {
        "ticker": ticker,
        "score": s,
        "price": latest["close"],
        "ema9": latest["ema9"],
        "ema20": latest["ema20"],
        "vwap": latest["vwap"],
        "time": latest["t"]
    }

def run_scan(tickers, period="1d", interval="5m"):
    results = []
    for t in tickers:
        r = process_ticker(t, period=period, interval=interval)
        if r:
            results.append(r)
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)
    return df

st.title("US Stock Scanner (Yahoo Finance, No API Key)")

tickers_input = st.text_area(
    "Tickers (comma-separated):",
    value=",".join(DEFAULT_TICKERS),
    help="Enter US tickers separated by commas."
)

period = st.selectbox("Period", ["1d","5d","1mo"], index=0)
interval = st.selectbox("Interval", ["1m","2m","5m","15m","30m","60m"], index=2)

if st.button("Run Scan"):
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        st.error("Please enter at least one ticker.")
    else:
        df = run_scan(tickers, period=period, interval=interval)
        if df.empty:
            st.error("No valid stocks found. Try different tickers, period, or interval.")
        else:
            st.dataframe(df.reset_index(drop=True))
