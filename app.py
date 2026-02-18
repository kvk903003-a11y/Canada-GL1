import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="TSX Intraday Scanner", layout="wide")

OPENING_RANGE_MINUTES = 15

# ---------------------------------------------------------
# CACHED HELPERS
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_tsx_tickers_live():
    """
    Try to scrape a broad TSX universe.
    If scraping fails, fall back to a large static list.
    """
    try:
        # Example: scrape TSX Composite components from a public source
        # You can replace this URL with a more reliable TSX components source.
        url = "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"
        tables = pd.read_html(url)
        comp = tables[0]
        # Expect a column with ticker symbols
        # Wikipedia often lists them without .TO, so we add it.
        tickers = (
            comp.iloc[:, 0]
            .astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)  # sometimes dots used differently
        )
        tickers = [t + ".TO" if not t.endswith(".TO") else t for t in tickers]
        tickers = sorted(list(set(tickers)))
        return tickers
    except Exception:
        # Fallback: large static universe
        fallback = [
            "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO", "MFC.TO", "SLF.TO", "GWO.TO", "IFC.TO",
            "BN.TO", "BAM.TO", "FFH.TO", "CP.TO", "CNR.TO", "TFII.TO", "WSP.TO", "WCN.TO", "SU.TO", "CNQ.TO",
            "CVE.TO", "TOU.TO", "ARX.TO", "MEG.TO", "WCP.TO", "VET.TO", "ENB.TO", "TRP.TO", "PPL.TO", "KEY.TO",
            "NPI.TO", "FTS.TO", "EMA.TO", "AQN.TO", "TA.TO", "SHOP.TO", "CSU.TO", "GIB.A.TO", "OTEX.TO", "LSPD.TO",
            "KXS.TO", "DSG.TO", "DCBO.TO", "NVEI.TO", "ATD.TO", "QSR.TO", "DOL.TO", "L.TO", "MRU.TO", "GOOS.TO",
            "ATZ.TO", "GIL.TO", "TECK.B.TO", "LUN.TO", "HBM.TO", "IVN.TO", "FVI.TO", "PAAS.TO", "ELD.TO", "AGI.TO",
            "AEM.TO", "FR.TO", "WPM.TO", "CCO.TO", "NXE.TO", "EFR.TO", "BHC.TO", "WELL.TO", "JWEL.TO", "BCE.TO",
            "T.TO", "RCI.B.TO", "QBR.B.TO", "TRI.TO", "CAR.UN.TO", "REI.UN.TO", "GRT.UN.TO", "SRU.UN.TO", "DIR.UN.TO",
            "AIF.TO", "CAE.TO", "ATS.TO", "WTE.TO", "BBD.B.TO", "DOO.TO", "MG.TO", "NFI.TO", "STN.TO", "TIH.TO",
            "RBA.TO", "FNV.TO", "ABX.TO", "NGD.TO", "OR.TO", "SSL.TO", "SAP.TO", "EMP.A.TO", "FCR.UN.TO", "BEI.UN.TO",
            "CHP.UN.TO", "HR.UN.TO", "AP.UN.TO", "NWH.UN.TO", "PKI.TO", "ATCO.TO", "ACO.X.TO", "CU.TO", "H.TO",
            "BLDP.TO", "XTC.TO", "MRE.TO", "LIF.TO", "SCL.TO", "SJR.B.TO", "CIX.TO", "POW.TO", "WN.TO", "CPX.TO",
            "ALA.TO", "GEI.TO", "BIR.TO", "PEY.TO", "ERF.TO", "TVE.TO", "KEL.TO", "AR.TO", "CR.TO", "BTE.TO",
            "CPG.TO", "NVA.TO", "VII.TO", "BIPC.TO", "BEPC.TO", "NTR.TO", "CF.TO", "HCG.TO", "EQB.TO", "LB.TO",
            "MKP.TO", "FSZ.TO", "GSY.TO", "PRM.TO", "HPS.A.TO", "BAD.TO", "ARE.TO", "AC.TO", "CJT.TO", "MDA.TO",
            "MAL.TO", "RCH.TO", "RUS.TO", "WPK.TO", "CAS.TO", "CCL.B.TO", "ITP.TO", "RPI.UN.TO", "YRI.TO", "PBH.TO",
            "LNF.TO", "BYD.TO", "MTY.TO", "PZA.TO", "AW.UN.TO", "BPF.UN.TO", "KEG.UN.TO", "CRT.UN.TO", "IIP.UN.TO",
            "MRG.UN.TO", "SOT.UN.TO", "SMU.UN.TO", "TCN.TO", "HOM.UN.TO", "NVU.UN.TO", "GDI.TO", "CJR.B.TO", "BB.TO",
            "BBTV.TO", "REAL.TO", "CTS.TO", "TCS.TO", "ENGH.TO", "SYZ.TO", "MDF.TO", "QIPT.TO", "PHO.TO", "DND.TO",
            "ABCL.TO", "CRDL.TO", "ACB.TO", "TLRY.TO", "OGI.TO", "HEXO.TO", "FIRE.TO",
        ]
        return fallback


@st.cache_data(ttl=60)
def cached_download(ticker: str, interval: str = "1m", period: str = "1d"):
    """Cached yfinance download to reduce rate limits."""
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    return df


# ---------------------------------------------------------
# SIDEBAR SETTINGS (for backtest)
# ---------------------------------------------------------
st.sidebar.header("Strategy Settings")

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
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def get_intraday_data(tickers, interval="1m", period="1d", max_workers=16, retries=3):
    data = {}

    def fetch_one(t):
        for _ in range(retries):
            try:
                df = cached_download(t, interval=interval, period=period)
                if not df.empty:
                    df = flatten_columns(df)
                    df = df.dropna()
                    df = df[~df.index.duplicated(keep="last")]
                    df = df.sort_index()
                    # Ensure required columns exist
                    if {"Close", "Volume"}.issubset(df.columns):
                        return t, df
                return t, None
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


def compute_vwap(df):
    df = df.copy()
    df = flatten_columns(df)
    if not {"Close", "Volume"}.issubset(df.columns):
        return df
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df


def compute_emas(df):
    df = df.copy()
    df = flatten_columns(df)
    if "Close" not in df.columns:
        return df
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
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
# SCORING + BUY/SELL LOGIC (ALL STRATEGIES COMBINED)
# ---------------------------------------------------------
def score_stock(ticker, df):
    if df.empty:
        return None

    df = compute_vwap(compute_emas(df))
    required = {"Close", "Volume", "VWAP", "EMA_9", "EMA_20"}
    if not required.issubset(df.columns):
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
        for r in strat_reasons:
            reasons.append(f"{strat}: {r}")

    if len(df) > 20:
        avg_vol = float(df["Volume"].tail(20).mean())
        if volume > avg_vol * 1.5:
            score += 1
            reasons.append("Volume spike")

    buy_price = close
    sell_price = round(close * (1 + (risk_reward / 100)), 4)
    momentum = ema9 - ema20
    performance = score + momentum

    patterns = detect_patterns(df)

    return {
        "ticker": ticker,
        "score": score,
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
# MULTI‑TIMEFRAME SNAPSHOT
# ---------------------------------------------------------
def get_multi_tf_snapshot(ticker):
    frames = {}
    for interval in ["1m", "5m", "15m"]:
        try:
            df = yf.download(ticker, interval=interval, period="1d", progress=False)
            if not df.empty:
                df = compute_emas(df)
                df = df.dropna()
                df = flatten_columns(df)
                if {"Close", "EMA_9", "EMA_20"}.issubset(df.columns):
                    last = df.tail(1).iloc[0]
                    frames[interval] = {
                        "close": float(last["Close"]),
                        "ema9": float(last["EMA_9"]),
                        "ema20": float(last["EMA_20"]),
                    }
        except Exception:
            continue
    return frames


# ---------------------------------------------------------
# BACKTEST ENGINE + CSV EXPORT
# ---------------------------------------------------------
def backtest_strategy(ticker, strategy_name, days=5):
    try:
        df = yf.download(ticker, interval="5m", period=f"{days}d", progress=False)
    except Exception:
        return None, 0.0

    if df.empty:
        return None, 0.0

    df = compute_vwap(compute_emas(df))
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = flatten_columns(df)

    if not {"Close", "VWAP", "EMA_9", "EMA_20"}.issubset(df.columns):
        return None, 0.0

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
st.title("🇨🇦 TSX Intraday Scanner — All Stocks, Top 10 Across All Strategies")

all_tickers = fetch_tsx_tickers_live()
st.write(f"Scanning {len(all_tickers)} Canadian tickers live.")

refresh_interval = st.slider("Auto-refresh interval (seconds)", 30, 300, 60)
st_autorefresh(interval=refresh_interval * 1000, key="refresh")


# ---------------------------------------------------------
# MAIN SCAN
# ---------------------------------------------------------
def run_scan():
    data = get_intraday_data(all_tickers)

    st.subheader("Debug: Data Status")
    debug_rows = []
    for t in all_tickers:
        if t in data:
            debug_rows.append([t, "OK", len(data[t])])
        else:
            debug_rows.append([t, "NO DATA", 0])
    st.dataframe(pd.DataFrame(debug_rows, columns=["Ticker", "Status", "Rows"]), use_container_width=True)

    rows = []

    for t, df in data.items():
        res = score_stock(t, df)
        if res:
            rows.append(res)

    if not rows:
        st.error("❌ No valid stocks found. This is normal when the market is closed or data is limited.")
        return

    df_scores = pd.DataFrame(
        [
            {
                "ticker": r["ticker"],
                "performance": r["performance"],
                "score": r["score"],
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

    top10 = df_scores.head(10)

    st.subheader("🏆 Top 10 Canadian Stocks (All Strategies Combined)")
    st.dataframe(
        top10[
            [
                "ticker",
                "performance",
                "score",
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
    top_rows = [r for r in rows if r["ticker"] in top10_tickers]

    for r in top_rows:
        t = r["ticker"]
        df = r["df"]

        with st.expander(f"{t} — Score {r['score']} — Perf {r['performance']:.2f}"):
            st.write("**Buy price:**", r["buy_price"])
            st.write("**Sell price:**", r["sell_price"])
            st.write("**VWAP:**", r["vwap"])
            st.write("**EMA 9 / EMA 20:**", r["ema9"], "/", r["ema20"])
            st.write("**ORH / ORL:**", r["orh"], "/", r["orl"])

            st.write("### Patterns")
            for p in r["patterns"]:
                st.write(f"- {p}")

            st.write("### Live Chart (1m)")
            df_plot = compute_vwap(compute_emas(df))
            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=df_plot.index,
                    open=df_plot["Open"],
                    high=df_plot["High"],
                    low=df_plot["Low"],
                    close=df_plot["Close"],
                    name="Price",
                )
            )
            if "VWAP" in df_plot.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_plot.index,
                        y=df_plot["VWAP"],
                        line=dict(color="orange"),
                        name="VWAP",
                    )
                )
            if "EMA_9" in df_plot.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_plot.index,
                        y=df_plot["EMA_9"],
                        line=dict(color="blue"),
                        name="EMA 9",
                    )
                )
            if "EMA_20" in df_plot.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_plot.index,
                        y=df_plot["EMA_20"],
                        line=dict(color="purple"),
                        name="EMA 20",
                    )
                )
            fig.update_layout(height=400, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            st.write("### Multi‑Timeframe Trend (1m / 5m / 15m)")
            mtf = get_multi_tf_snapshot(t)
            for tf, vals in mtf.items():
                st.write(
                    f"- {tf}: Close={vals['close']:.2f}, "
                    f"EMA9={vals['ema9']:.2f}, EMA20={vals['ema20']:.2f}"
                )

            st.write("### Backtest (5 days, selected strategy)")
            bt_df, pnl = backtest_strategy(t, backtest_strategy_name, days=5)
            if bt_df is not None:
                st.write(f"Total PnL: {pnl:.2f}")
                st.dataframe(bt_df, use_container_width=True)

                csv = bt_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Backtest CSV",
                    data=csv,
                    file_name=f"{t}_backtest_{backtest_strategy_name.replace(' ', '_')}.csv",
                    mime="text/csv",
                )
            else:
                st.write("No backtest trades generated.")


run_scan()
