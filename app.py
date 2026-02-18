import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

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
# SIDEBAR STRATEGY SETTINGS
# ---------------------------------------------------------
st.sidebar.header("Strategy Settings")

strategy = st.sidebar.selectbox(
    "Choose Strategy",
    [
        "Opening Range Breakout",
        "VWAP Pullback",
        "Trend Continuation",
        "Reversal Setup",
        "All Strategies (Weighted)",
    ],
)

risk_reward = st.sidebar.slider("Risk/Reward Target", 1.0, 5.0, 2.0)
pullback_depth = st.sidebar.slider("VWAP Pullback Depth (%)", 0.1, 2.0, 0.5)
ema_trend_strength = st.sidebar.slider("Trend Strength (EMA9‑EMA20)", 0.0, 1.0, 0.2)

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
                df = df[~df.index.duplicated(keep="last")]
                df = df.sort_index()
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
# PATTERN DETECTION (LIGHT "AI")
# ---------------------------------------------------------
def detect_patterns(df):
    df = compute_vwap(compute_emas(df))
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
# SCORING + BUY/SELL LOGIC
# ---------------------------------------------------------
def score_stock(ticker, df):
    if df.empty:
        return None

    df = compute_vwap(compute_emas(df))

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

    strat_score, strat_reasons = apply_strategy_logic(
        strategy, close, vwap, ema9, ema20, orh, orl, df
    )
    score += strat_score
    reasons.extend(strat_reasons)

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
        df = yf.download(ticker, interval=interval, period="1d", progress=False)
        if not df.empty:
            df = compute_emas(df)
            df = df.dropna()
            last = df.tail(1).iloc[0]
            frames[interval] = {
                "close": float(last["Close"]),
                "ema9": float(last["EMA_9"]),
                "ema20": float(last["EMA_20"]),
            }
    return frames


# ---------------------------------------------------------
# BACKTEST ENGINE + CSV EXPORT
# ---------------------------------------------------------
def backtest_strategy(ticker, strategy, days=5):
    df = yf.download(ticker, interval="5m", period=f"{days}d", progress=False)
    if df.empty:
        return None, 0.0

    df = compute_vwap(compute_emas(df))
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

        score, _ = apply_strategy_logic(strategy, close, vwap, ema9, ema20, None, None, df)

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
st.title("🇨🇦 TSX Intraday Scanner — Strategies, Charts, Backtests")

tickers_input = st.text_input(
    "TSX tickers (comma-separated, with .TO):",
    value=", ".join(DEFAULT_TICKERS),
)

tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

refresh_interval = st.slider("Auto-refresh interval (seconds)", 30, 300, 60)
st_autorefresh(interval=refresh_interval * 1000, key="refresh")


# ---------------------------------------------------------
# MAIN SCAN
# ---------------------------------------------------------
def run_scan():
    data = get_intraday_data(tickers)

    st.subheader("Debug: Data Status")
    debug_rows = []
    for t in tickers:
        if t in data:
            debug_rows.append([t, "OK", len(data[t])])
        else:
            debug_rows.append([t, "NO DATA", 0])
    st.table(pd.DataFrame(debug_rows, columns=["Ticker", "Status", "Rows"]))

    rows = []

    for t, df in data.items():
        res = score_stock(t, df)
        if res:
            rows.append(res)
        else:
            st.write(f"⚠️ Skipped {t} — scoring returned None")

    if not rows:
        st.error("❌ No valid stocks found. This is normal when the market is closed or data is limited.")
        return

    df_scores = pd.DataFrame([
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
    ])

    df_scores = df_scores.sort_values("performance", ascending=False)

    st.subheader("📊 Ranked Intraday Opportunities")
    st.dataframe(
        df_scores[
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

    st.subheader("Details")

    for r in rows:
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
            fig.add_trace(go.Candlestick(
                x=df_plot.index,
                open=df_plot["Open"],
                high=df_plot["High"],
                low=df_plot["Low"],
                close=df_plot["Close"],
                name="Price",
            ))
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["VWAP"],
                line=dict(color="orange"), name="VWAP"
            ))
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["EMA_9"],
                line=dict(color="blue"), name="EMA 9"
            ))
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["EMA_20"],
                line=dict(color="purple"), name="EMA 20"
            ))
            fig.update_layout(height=400, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            st.write("### Multi‑Timeframe Trend (1m / 5m / 15m)")
            mtf = get_multi_tf_snapshot(t)
            for tf, vals in mtf.items():
                st.write(
                    f"- {tf}: Close={vals['close']:.2f}, "
                    f"EMA9={vals['ema9']:.2f}, EMA20={vals['ema20']:.2f}"
                )

            st.write("### Backtest (5 days, current strategy)")
            bt_df, pnl = backtest_strategy(t, strategy, days=5)
            if bt_df is not None:
                st.write(f"Total PnL: {pnl:.2f}")
                st.dataframe(bt_df, use_container_width=True)

                csv = bt_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Backtest CSV",
                    data=csv,
                    file_name=f"{t}_backtest_{strategy.replace(' ', '_')}.csv",
                    mime="text/csv",
                )
            else:
                st.write("No backtest trades generated.")


run_scan()
