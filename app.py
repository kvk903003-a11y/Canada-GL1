import streamlit as st
import requests
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime, timedelta

POLYGON_KEY = "abc123xyz456POLYGONKEY"

st.set_page_config(page_title="US Stock Scanner", layout="wide")

@st.cache_data(ttl=3600)
def get_us_tickers():
    url = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true",
        "limit": 1000,
        "apiKey": POLYGON_KEY
    }
    tickers = []
    while True:
        r = requests.get(url, params=params).json()
        for t in r.get("results", []):
            if t.get("type") == "CS":  # common stocks only
                tickers.append(t["ticker"])
        if "next_url" in r:
            url = r["next_url"]
            params = {"apiKey": POLYGON_KEY}
        else:
            break
    return tickers

def get_intraday(ticker):
    end = datetime.utcnow()
    start = end - timedelta(hours=6)  # safe window for free tier
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/{int(start.timestamp()*1000)}/{int(end.timestamp()*1000)}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_KEY}
    try:
        r = requests.get(url, params=params).json()
        if "results" not in r:
            return None
        df = pd.DataFrame(r["results"])
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
        return df
    except:
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

def process_ticker(ticker):
    df = get_intraday(ticker)
    df = compute_indicators(df)
    if df is None:
        return None
    s = score(df)
    return {"ticker": ticker, "score": s, "price": df.iloc[-1]["close"]}

def run_scan():
    tickers = get_us_tickers()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        for r in ex.map(process_ticker, tickers):
            if r:
                results.append(r)
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)
    return df

st.title("US Stock Scanner (Polygon Free Tier)")

if st.button("Run Scan"):
    df = run_scan()
    if df.empty:
        st.error("No valid stocks found. Try during market hours.")
    else:
        st.dataframe(df)
