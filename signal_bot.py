"""
BTC & GOLD Multi-Timeframe Signal Bot (Daily + 1H + 15m)
Free data: yfinance | Alerts: Telegram

Install:  pip install yfinance pandas numpy requests
Backtest: python signal_bot.py backtest
Live bot: python signal_bot.py live
Cloud (mobile setup): python signal_bot.py once  (GitHub Actions har 15 min chalata hai)
"""
import os, sys, json, time, datetime as dt
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ================= SETTINGS (yahan apni values daalo) =================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YAHAN_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YAHAN_CHAT_ID")

SYMBOLS = {"BTC": "BTC-USD", "GOLD": "GC=F"}  # GC=F = Gold futures (spot se thoda alag)
RR = 2.0            # Take profit = 2 x risk
ATR_MULT = 1.5      # Stop loss = 1.5 x ATR
MAX_BARS = 96       # 96 x 15m = 1 din; tab tak SL/TP na lage to exit
CHECK_EVERY = 300   # live mode: har 5 minute check
# =====================================================================


def get(sym, interval, period):
    df = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = pd.DatetimeIndex(df.index.tz_convert("UTC")).as_unit("ns")
    return df


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def atr(df, n=14):
    tr = pd.concat([
        df.high - df.low,
        (df.high - df.close.shift()).abs(),
        (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def build(sym):
    """3 timeframe data + signals. Lookahead se bachne ke liye D/1H ko shift(1) kiya hai."""
    # --- Daily trend ---
    d = get(sym, "1d", "2y")
    e50, e200 = ema(d.close, 50), ema(d.close, 200)
    d["trend"] = np.where((d.close > e50) & (e50 > e200), 1,
                  np.where((d.close < e50) & (e50 < e200), -1, 0))
    d = d[["trend"]].shift(1).dropna()

    # --- 1H bias ---
    h = get(sym, "1h", "730d")
    h20, h50, hr = ema(h.close, 20), ema(h.close, 50), rsi(h.close)
    h["bias"] = np.where((h20 > h50) & (hr > 50), 1,
                 np.where((h20 < h50) & (hr < 50), -1, 0))
    h["h_rsi"] = hr
    h = h[["bias", "h_rsi"]].shift(1).dropna()

    # --- 15m entry ---
    m = get(sym, "15m", "60d")
    m["ema20"] = ema(m.close, 20)
    m["rsi"] = rsi(m.close)
    m["atr"] = atr(m)
    m["volavg"] = m.volume.rolling(20).mean()
    m = pd.merge_asof(m, h, left_index=True, right_index=True)
    m = pd.merge_asof(m, d, left_index=True, right_index=True)

    long_ = ((m.low.shift() <= m.ema20.shift()) & (m.close > m.ema20) &
             (m.close > m.open) & (m.trend == 1) & (m.bias == 1))
    short_ = ((m.high.shift() >= m.ema20.shift()) & (m.close < m.ema20) &
              (m.close < m.open) & (m.trend == -1) & (m.bias == -1))
    m["sig"] = np.where(long_, 1, np.where(short_, -1, 0))

    # --- Confluence score (55 base + 15 per extra check, max 100) ---
    s = m.sig.values
    c1 = np.where(s == 1, m.h_rsi > 55, m.h_rsi < 45).astype(int)
    c2 = (m.volume > m.volavg).values.astype(int)
    c3 = np.where(s == 1, (m.rsi > 50) & (m.rsi < 65), (m.rsi < 50) & (m.rsi > 35)).astype(int)
    m["score"] = 55 + 15 * (c1 + c2 + c3)
    return m.dropna()


def backtest(m):
    res, i, n = [], 0, len(m)
    c, hi, lo = m.close.values, m.high.values, m.low.values
    a, s = m.atr.values, m.sig.values
    while i < n - 1:
        if s[i] == 0:
            i += 1
            continue
        e, risk, d = c[i], a[i] * ATR_MULT, s[i]
        sl, tp, r = e - d * risk, e + d * risk * RR, None
        for j in range(i + 1, min(i + 1 + MAX_BARS, n)):
            hit_sl = lo[j] <= sl if d == 1 else hi[j] >= sl
            hit_tp = hi[j] >= tp if d == 1 else lo[j] <= tp
            if hit_sl:          # same candle me dono lage to SL maan lo (conservative)
                r = -1
                break
            if hit_tp:
                r = RR
                break
        if r is None:
            j = min(i + MAX_BARS, n - 1)
            r = (c[j] - e) * d / risk
        res.append(r)
        i = j + 1
    return res


def stats(res):
    if not res:
        return {"trades": 0, "win_rate": 0.0, "avg_R": 0.0}
    r = np.array(res)
    return {"trades": len(r), "win_rate": round((r > 0).mean() * 100, 1),
            "avg_R": round(r.mean(), 2)}


def send(text):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=15)
    except Exception as ex:
        print("Telegram error:", ex)


def run_backtest():
    for name, sym in SYMBOLS.items():
        st = stats(backtest(build(sym)))
        print(f"{name}: trades={st['trades']}  win_rate={st['win_rate']}%  avg_R={st['avg_R']}")
    print("\nNote: 15m data sirf ~60 din ka milta hai, isliye sample chhota hai.")


STATE_FILE = "last_sent.json"


def make_msg(name, row, wr):
    d = int(row.sig)
    e, risk = row.close, row.atr * ATR_MULT
    sl, tp = e - d * risk, e + d * risk * RR
    return (f"{'🟢 BUY' if d == 1 else '🔴 SELL'} {name}\n"
            f"Entry: {e:.2f}\nSL: {sl:.2f}\nTP: {tp:.2f} (RR 1:{RR})\n"
            f"Confluence score: {int(row.score)}/100\n"
            f"Backtest win-rate: {wr['win_rate']}% ({wr['trades']} trades)\n"
            f"TF: Daily + 1H + 15m aligned\n"
            f"Time: {dt.datetime.utcnow():%d-%b %H:%M} UTC")


def run_once():
    """Ek baar check karke exit (GitHub Actions ke liye). Duplicate alert state file se rokta hai."""
    try:
        state = json.load(open(STATE_FILE))
    except Exception:
        state = {}
    changed = False
    for name, sym in SYMBOLS.items():
        try:
            m = build(sym)
            wr = stats(backtest(m))
            for k in (-3, -2):   # last 2 closed candles (cron delay ke liye)
                row, ts = m.iloc[k], m.index[k]
                age = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60
                if row.sig != 0 and 15 <= age <= 45 and state.get(name) != str(ts):
                    msg = make_msg(name, row, wr)
                    send(msg)
                    print(msg)
                    state[name] = str(ts)
                    changed = True
        except Exception as ex:
            print(name, "error:", ex)
    if changed:
        json.dump(state, open(STATE_FILE, "w"))
    print("check done", dt.datetime.utcnow())


def run_live():
    win = {n: stats(backtest(build(s))) for n, s in SYMBOLS.items()}
    send("✅ Signal bot start ho gaya.")
    last_sent = {}
    while True:
        for name, sym in SYMBOLS.items():
            try:
                m = build(sym)
                row, ts = m.iloc[-2], m.index[-2]
                age = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60
                if row.sig != 0 and 15 <= age <= 45 and last_sent.get(name) != ts:
                    send(make_msg(name, row, win[name]))
                    last_sent[name] = ts
            except Exception as ex:
                print(name, "error:", ex)
        time.sleep(CHECK_EVERY)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    if mode == "backtest":
        run_backtest()
    elif mode == "once":
        run_once()
    elif mode == "test":
        send("✅ Test: bot Telegram se connect ho gaya.")
        print("test sent")
    else:
        run_live()
  
