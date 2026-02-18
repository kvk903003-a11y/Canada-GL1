def score_stock(ticker, df):
    if df.empty:
        return None

    # Fix duplicate timestamps (Yahoo bug)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    df = compute_vwap(df)
    df = compute_emas(df)

    # Force last row to be a single row
    last = df.iloc[-1]

    # Convert everything to float scalars
    close = float(last["Close"])
    vwap = float(last["VWAP"])
    ema9 = float(last["EMA_9"])
    ema20 = float(last["EMA_20"])
    volume = float(last["Volume"])

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
        avg_vol = float(df["Volume"].tail(20).mean())
        if volume > avg_vol * 1.5:
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
        "orh": orh,
        "orl": orl,
        "reasons": reasons,
    }
