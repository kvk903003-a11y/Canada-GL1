import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="TSX Intraday Scanner — ML, Sectors, Pre‑Market", layout="wide")
OPENING_RANGE_MINUTES = 15

# ---------------------------------------------------------
# GLOBAL COLUMN FLATTENER
# ---------------------------------------------------------
def flatten(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    try:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    except Exception:
        pass
    return df

# ---------------------------------------------------------
# CACHED HELPERS
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_tsx_tickers_live():
    try:
        url = "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"
        tables = pd.read_html(url)
        comp = tables[0]
        tickers = comp.iloc[:, 0].astype(str).str.strip()
        tickers = [t + ".TO" if not t.endswith(".TO") else t for t in tickers]
        return sorted(list(set(tickers)))
    except Exception:
        return [
            "RY.TO","TD.TO","BNS.TO","BMO.TO","CM.TO","NA.TO","MFC.TO","SLF.TO","GWO.TO","IFC.TO",
            "BN.TO","BAM.TO","FFH.TO","CP.TO","CNR.TO","TFII.TO","WSP.TO","WCN.TO","SU.TO","CNQ.TO",
            "CVE.TO","TOU.TO","ARX.TO","MEG.TO","WCP.TO","VET.TO","ENB.TO","TRP.TO","PPL.TO","KEY.TO",
            "NPI.TO","FTS.TO","EMA.TO","AQN.TO","TA.TO","SHOP.TO","CSU.TO","GIB.A.TO","OTEX.TO","LSPD.TO",
        ]

@st.cache_data(ttl=3600)
def get_sector(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("sector", "Unknown")
    except Exception:
        return "Unknown"

@st.cache_data(ttl=60)
def cached_download(ticker: str, interval: str = "1m", period: str = "1d", prepost: bool = False):
    df = yf.download(ticker, interval=interval, period=period, progress=False, prepost=prepost)
    return flatten(df)

# ---------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------
st.sidebar.header("Scanner Settings")

session_type = st.sidebar.selectbox(
    "Session",
    ["Regular Hours", "Pre‑Market"],
)

backtest_strategy_name = st.sidebar.selectbox(
    "Backtest Strategy",
    [
        "Opening Range Breakout",
        "VWAP Pullback",
        "Trend Continuation",
        "Reversal Setup",
        "All Strategies (Weighted)",
    ],
)

risk_reward = st.sidebar.slider("Risk/Reward Target (%)", 1.0, 5.0, 2.0)
pullback_depth = st.sidebar.slider("VWAP Pullback Depth (%)", 0.1, 2.0, 0.5)
ema_trend_strength = st.sidebar.slider("Trend Strength (EMA9‑EMA20)", 0.0, 1.0, 0.2)

ALL_STRATEGIES = [
    "Opening Range Breakout",
    "VWAP Pullback",
    "Trend Continuation",
    "Reversal Setup",
    "All Strategies (Weighted)",
]

# ---------------------------------------------------------
# DATA FUNCTIONS
# ---------------------------------------------------------
def get_intraday_data(tickers, interval="1m", period="1d", max_workers=16, retries=3, prepost=False):
    data = {}

    def fetch_one(t):
        for _ in range(retries):
            try:
                df = cached_download(t, interval=interval, period=period, prepost=prepost)
                df = flatten(df)
                if df is None or df.empty:
                    continue
                if not {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
                    continue
                df = df.dropna()
                df = df[~df.index.duplicated(keep="last")]
                df = df.sort_index()
                if df.empty:
                    continue

                # If pre‑market, optionally filter to pre‑open (TSX ~ 09:30 ET)
                if session_type == "Pre‑Market":
                    # yfinance index is usually timezone‑aware; keep everything before 09:30 local
                    try:
                        idx = df.index.tz_convert("America/Toronto")
                        df = df[idx.time < dt.time(9, 30)]
                        if df.empty:
                            continue
                    except Exception:
                        pass

                return t, df
            except Exception:
                continue
        return t, None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            t, df = fut.result()
            if df is not None:
                data[t] = df

    return data

def compute_emas(df):
    df = flatten(df.copy())
    if "Close" not in df.columns:
        return df
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    return df

def compute_vwap(df):
    df = flatten(df.copy())
    if not {"Close", "Volume"}.issubset(df.columns):
        return df
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df

def get_opening_range(df):
    if df.empty:
        return None, None
    start = df.index[0]
    end = start + dt.timedelta(minutes=OPENING_RANGE_MINUTES)
    or_df = df[(df.index >= start) & (df.index < end)]
    if or_df.empty:
        return None, None
    return float(or_df["High"].max()), float(or_df["Low"].min())

# ---------------------------------------------------------
# STRATEGY LOGIC
# ---------------------------------------------------------
def apply_strategy_logic(strategy, close, vwap, ema9, ema20, orh, orl, df):
    score = 0
    reasons = []

    if strategy == "Opening Range Breakout":
        if orh is not None and close > orh:
            score += 3
            reasons.append("OR Breakout")
        if orl is not None and close < orl:
            score += 3
            reasons.append("OR Breakdown")

    elif strategy == "VWAP Pullback":
        if vwap > 0:
            distance = abs(close - vwap) / vwap * 100
            if distance <= pullback_depth:
                score += 2
                reasons.append("Near VWAP (pullback zone)")
        if close > vwap:
            score += 1
            reasons.append("Above VWAP (bullish bias)")

    elif strategy == "Trend Continuation":
        if ema9 > ema20 + ema_trend_strength:
            score += 2
            reasons.append("Strong EMA trend")
        if close > ema9:
            score += 1
            reasons.append("Price above EMA9")

    elif strategy == "Reversal Setup":
        last = df.tail(1).iloc[0]
        body = abs(float(last["Close"] - last["Open"]))
        range_ = float(last["High"] - last["Low"])
        if range_ > 0 and body / range_ < 0.3:
            score += 2
            reasons.append("Indecision candle (reversal)")
        if close < vwap:
            score += 1
            reasons.append("Below VWAP (oversold)")

    elif strategy == "All Strategies (Weighted)":
        if orh is not None and close > orh:
            score += 2
            reasons.append("OR Breakout (weighted)")
        if vwap > 0 and abs(close - vwap) / vwap * 100 < pullback_depth:
            score += 1
            reasons.append("VWAP pullback (weighted)")
        if ema9 > ema20:
            score += 1
            reasons.append("Trend up (weighted)")

    return score, reasons

# ---------------------------------------------------------
# LIGHT ML SCORING (LINEAR MODEL ON PAST RETURNS)
# ---------------------------------------------------------
def ml_score_from_df(df: pd.DataFrame) -> float:
    df = flatten(df.copy())
    if "Close" not in df.columns or "Volume" not in df.columns:
        return 0.0

    df["ret"] = df["Close"].pct_change()
    df["future_ret"] = df["ret"].shift(-1)

    df["vol_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / (df["Volume"].rolling(20).std() + 1e-9)
    df = compute_emas(df)
    if not {"EMA_9", "EMA_20"}.issubset(df.columns):
        return 0.0
    df["ema_diff"] = df["EMA_9"] - df["EMA_20"]

    feat_cols = ["ret", "vol_z", "ema_diff"]
    if any(c not in df.columns for c in feat_cols):
        return 0.0

    sub = df.dropna(subset=feat_cols + ["future_ret"]).copy()
    if len(sub) < 40:
        return 0.0

    X = sub[feat_cols].values
    y = sub["future_ret"].values

    X = np.hstack([np.ones((X.shape[0], 1)), X])  # add bias
    try:
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return 0.0

    last_row = df.iloc[[-2]] if len(df) > 2 else df.tail(1)
    x_last = last_row[feat_cols].values
    x_last = np.hstack([np.ones((x_last.shape[0], 1)), x_last])
    pred = float(x_last @ w)
    return float(np.clip(pred * 100, -5, 5))  # scale to a small range

# ---------------------------------------------------------
# PATTERN DETECTION
# ---------------------------------------------------------
def detect_patterns(df):
    df = compute_vwap(compute_emas(df))
    if not {"Close", "VWAP", "EMA_9", "EMA_20"}.issubset(df.columns):
        return []
    last = df.tail(1).iloc[0]

    close = float(last["Close"])
    vwap = float(last["VWAP"])
    ema9 = float(last["EMA_9"])
    ema20 = float(last["EMA_20"])

    patterns = []

    if ema9 > ema20 and close > vwap:
        patterns.append("Strong uptrend")
    elif ema9 < ema20 and close < vwap:
        patterns.append("Strong downtrend")
    else:
        patterns.append("Choppy / mixed trend")

    body = abs(float(last["Close"] - last["Open"]))
    range_ = float(last["High"] - last["Low"])
    if range_ > 0 and body / range_ < 0.3:
        patterns.append("Indecision / possible reversal candle")

    return patterns

# ---------------------------------------------------------
# SCORING (ALL STRATEGIES + ML)
# ---------------------------------------------------------
def score_stock(ticker, df):
    df = flatten(df)
    df = compute_vwap(compute_emas(df))

    if not {"Close", "Volume", "VWAP", "EMA_9", "EMA_20"}.issubset(df.columns):
        return None

    last = df.tail(1).iloc[0]

    try:
        close = float(last["Close"])
        vwap = float(last["VWAP"])
        ema9 = float(last["EMA_9"])
        ema20 = float(last["EMA_20"])
        volume = float(last["Volume"])
    except Exception:
        return None

    orh, orl = get_opening_range(df)

    score = 0
    reasons = []

    if close > vwap:
        score += 1
        reasons.append("Above VWAP")
    if ema9 > ema20:
        score += 1
        reasons.append("EMA9 > EMA20 (uptrend)")

    for strat in ALL_STRATEGIES:
        strat_score, strat_reasons = apply_strategy_logic(
            strat, close, vwap, ema9, ema20, orh, orl, df
        )
        score += strat_score
        reasons.extend([f"{strat}: {r}" for r in strat_reasons])

    if len(df) > 20:
        avg_vol = float(df["Volume"].tail(20).mean())
        if volume > avg_vol * 1.5:
            score += 1
            reasons.append("Volume spike")

    ml_score = ml_score_from_df(df)

    buy_price = close
    sell_price = round(close * (1 + (risk_reward / 100)), 4)
    momentum = ema9 - ema20
    performance = score + momentum + ml_score

    patterns = detect_patterns(df)
    sector = get_sector(ticker)

    return {
        "ticker": ticker,
        "sector": sector,
        "score": score,
        "ml_score": ml_score,
        "performance": float(performance),
        "buy_price": float(buy_price),
        "sell_price": float(sell_price),
        "vwap": vwap,
        "ema9": ema9,
        "ema20": ema20,
        "orh": orh,
        "orl": orl,
        "reasons": reasons,
        "patterns": patterns,
        "df": df,
    }

# ---------------------------------------------------------
# BACKTEST (NO CHARTS)
# ---------------------------------------------------------
def backtest_strategy(ticker, strategy_name, days=5):
    try:
        df = yf.download(ticker, interval="5m", period=f"{days}d", progress=False)
    except Exception:
        return None, 0.0

    df = flatten(df)
    df = compute_vwap(compute_emas(df))
    if not {"Close", "VWAP", "EMA_9", "EMA_20"}.issubset(df.columns):
        return None, 0.0

    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    position = 0
    entry = 0.0
    pnl = 0.0
    trades = []

    for ts, row in df.iterrows():
        close = float(row["Close"])
        vwap = float(row["VWAP"])
        ema9 = float(row["EMA_9"])
        ema20 = float(row["EMA_20"])

        score, _ = apply_strategy_logic(strategy_name, close, vwap, ema9, ema20, None, None, df)

        if position == 0 and score >= 2:
            position = 1
            entry = close
            trades.append({"timestamp": ts, "action": "BUY", "price": close})

        elif position == 1 and close < ema20:
            pnl += close - entry
            trades.append({"timestamp": ts, "action": "SELL", "price": close})
            position = 0
            entry = 0.0

    if position == 1:
        pnl += close - entry
        trades.append({"timestamp": ts, "action": "SELL_EOD", "price": close})

    if not trades:
        return None, 0.0

    bt_df = pd.DataFrame(trades)
    return bt_df, pnl

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("🇨🇦 TSX Intraday Scanner — ML Scoring, Sector‑Balanced, Pre‑Market")

all_tickers = fetch_tsx_tickers_live()
st.write(f"Scanning {len(all_tickers)} Canadian tickers live — Session: {session_type}")

refresh_interval = st.slider("Auto-refresh interval (seconds)", 30, 300, 60)
st_autorefresh(interval=refresh_interval * 1000, key="refresh")

# ---------------------------------------------------------
# MAIN SCAN
# ---------------------------------------------------------
def run_scan():
    prepost = session_type == "Pre‑Market"
    data = get_intraday_data(all_tickers, prepost=prepost)

    rows = []
    for t, df in data.items():
        res = score_stock(t, df)
        if res:
            rows.append(res)

    if not rows:
        st.error("No valid stocks found (market may be closed or data limited).")
        return

    df_scores = pd.DataFrame(
        [
            {
                "ticker": r["ticker"],
                "sector": r["sector"],
                "performance": r["performance"],
                "score": r["score"],
                "ml_score": r["ml_score"],
                "buy_price": r["buy_price"],
                "sell_price": r["sell_price"],
                "vwap": r["vwap"],
                "ema9": r["ema9"],
                "ema20": r["ema20"],
                "orh": r["orh"],
                "orl": r["orl"],
            }
            for r in rows
        ]
    )

    df_scores = df_scores.sort_values("performance", ascending=False)

    # Sector‑balanced top 10
    max_per_sector = 3
    picked = []
    counts = {}

    for _, row in df_scores.iterrows():
        sec = row["sector"]
        counts.setdefault(sec, 0)
        if counts[sec] < max_per_sector:
            picked.append(row)
            counts[sec] += 1
        if len(picked) >= 10:
            break

    top10 = pd.DataFrame(picked)

    st.subheader("🏆 Sector‑Balanced Top 10 (All Strategies + ML)")
    st.dataframe(
        top10[
            [
                "ticker",
                "sector",
                "performance",
                "score",
                "ml_score",
                "buy_price",
                "sell_price",
                "vwap",
                "ema9",
                "ema20",
                "orh",
                "orl",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Details for Top 10")
    top10_tickers = set(top10["ticker"].tolist())

    for r in rows:
        if r["ticker"] not in top10_tickers:
            continue

        with st.expander(f"{r['ticker']} — Sector {r['sector']} — Score {r['score']} — ML {r['ml_score']:.2f} — Perf {r['performance']:.2f}"):
            st.write("Buy price:", r["buy_price"])
            st.write("Sell price:", r["sell_price"])
            st.write("VWAP:", r["vwap"])
            st.write("EMA9 / EMA20:", r["ema9"], "/", r["ema20"])
            st.write("ORH / ORL:", r["orh"], "/", r["orl"])
            st.write("Patterns:", ", ".join(r["patterns"]))
            st.write("Reasons:")
            for reason in r["reasons"]:
                st.write("-", reason)

            st.write("Backtest (5 days, selected strategy)")
            bt_df, pnl = backtest_strategy(r["ticker"], backtest_strategy_name, days=5)
            if bt_df is not None:
                st.write(f"Total PnL: {pnl:.2f}")
                st.dataframe(bt_df, use_container_width=True)
                csv = bt_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Backtest CSV",
                    data=csv,
                    file_name=f"{r['ticker']}_backtest_{backtest_strategy_name.replace(' ', '_')}.csv",
                    mime="text/csv",
                )
            else:
                st.write("No backtest trades generated.")

run_scan()
