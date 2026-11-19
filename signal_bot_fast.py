"""
FAST bot: same strategy, sirf timeframe change
  15m = trend (price > EMA50 > EMA200)
   5m = bias  (EMA20 vs EMA50 + RSI)
   1m = entry (EMA20 pullback + candle close)
Alag state file (fast_state.json) aur alag alerts (⚡FAST). Purana bot (signal_bot.py) waise hi chalta rahega.

Modes:  python signal_bot_fast.py once | test | backtest
"""
import os, sys, json, datetime as dt
import numpy as np
import pandas as pd
from signal_bot import get, ema, rsi, atr, send, SYMBOLS   # helpers purani file se

RR = 2.0
ATR_MULT = 1.5
MAX_BARS = 240                 # 240 x 1m = 4 ghante, tab tak TP/SL na lage to time-exit
MAX_AGE = 12                   # minute: isse purana signal ignore (1m me jaldi bekaar ho jata hai)
COST = {"BTC": 0.0008, "GOLD": 0.0003}   # round-trip spread/fees
STATE_FILE = "fast_state.json"


def build_fast(sym):
    t = get(sym, "15m", "60d")                       # trend
    e50, e200 = ema(t.close, 50), ema(t.close, 200)
    t["trend"] = np.where((t.close > e50) & (e50 > e200), 1,
                  np.where((t.close < e50) & (e50 < e200), -1, 0))
    t = t[["trend"]].shift(1).dropna()

    b = get(sym, "5m", "60d")                        # bias
    b20, b50, br = ema(b.close, 20), ema(b.close, 50), rsi(b.close)
    b["bias"] = np.where((b20 > b50) & (br > 50), 1,
                 np.where((b20 < b50) & (br < 50), -1, 0))
    b["h_rsi"] = br
    b = b[["bias", "h_rsi"]].shift(1).dropna()

    m = get(sym, "1m", "7d")                         # entry
    m["ema20"] = ema(m.close, 20)
    m["rsi"] = rsi(m.close)
    m["atr"] = atr(m)
    m["volavg"] = m.volume.rolling(20).mean()
    m = pd.merge_asof(m, b, left_index=True, right_index=True)
    m = pd.merge_asof(m, t, left_index=True, right_index=True)

    long_ = ((m.low.shift() <= m.ema20.shift()) & (m.close > m.ema20) &
             (m.close > m.open) & (m.trend == 1) & (m.bias == 1))
    short_ = ((m.high.shift() >= m.ema20.shift()) & (m.close < m.ema20) &
              (m.close < m.open) & (m.trend == -1) & (m.bias == -1))
    m["sig"] = np.where(long_, 1, np.where(short_, -1, 0))

    s = m.sig.values
    c1 = np.where(s == 1, m.h_rsi > 55, m.h_rsi < 45).astype(int)
    c2 = (m.volume > m.volavg).values.astype(int)
    c3 = np.where(s == 1, (m.rsi > 50) & (m.rsi < 65), (m.rsi < 50) & (m.rsi > 35)).astype(int)
    m["score"] = 55 + 15 * (c1 + c2 + c3)
    return m.dropna()


def backtest(m, name):
    """Cost ke baad R list."""
    c, hi, lo = m.close.values, m.high.values, m.low.values
    a, s = m.atr.values, m.sig.values
    res, i, n = [], 0, len(m)
    while i < n - 1:
        if s[i] == 0 or a[i] <= 0:
            i += 1
            continue
        e, risk, d = c[i], a[i] * ATR_MULT, s[i]
        sl, tp, r = e - d * risk, e + d * risk * RR, None
        j = i
        for j in range(i + 1, min(i + 1 + MAX_BARS, n)):
            hit_sl = lo[j] <= sl if d == 1 else hi[j] >= sl
            hit_tp = hi[j] >= tp if d == 1 else lo[j] <= tp
            if hit_sl:
                r = -1
                break
            if hit_tp:
                r = RR
                break
        if r is None:
            j = min(i + MAX_BARS, n - 1)
            r = (c[j] - e) * d / risk
        res.append(r - COST[name] * e / risk)
        i = j + 1
    return res


def stats(res):
    if not res:
        return {"trades": 0, "win_rate": 0.0, "avg_R": 0.0}
    r = np.array(res)
    return {"trades": len(r), "win_rate": round((r > 0).mean() * 100, 1),
            "avg_R": round(float(r.mean()), 2)}


def load_state():
    try:
        st = json.load(open(STATE_FILE))
    except Exception:
        st = {}
    st.setdefault("open", {})
    st.setdefault("last", {})
    st.setdefault("tally", {"W": 0, "L": 0, "T": 0, "R": 0.0})
    return st


def new_trade(row, ts):
    d = int(row.sig)
    e, risk = float(row.close), float(row.atr * ATR_MULT)
    return {"dir": d, "entry": e, "risk": risk, "sl": e - d * risk,
            "tp": e + d * risk * RR, "ts": str(ts)}


def make_msg(name, t, row, wr, age):
    d = t["dir"]
    return (f"⚡FAST {'🟢 BUY' if d == 1 else '🔴 SELL'} {name}\n"
            f"Entry: {t['entry']:.2f}\nSL: {t['sl']:.2f}\nTP: {t['tp']:.2f} (RR 1:{RR})\n"
            f"Confluence score: {int(row.score)}/100\n"
            f"Backtest (7 din, cost ke baad): win {wr['win_rate']}% | avgR {wr['avg_R']} | {wr['trades']} trades\n"
            f"TF: 15m + 5m + 1m aligned\n"
            f"Signal {age:.0f} min purana hai, entry 1m candle ke close par bana tha, price check karke hi lo.\n"
            f"(Is trade ka TP/SL result alag message me aayega)")


def check_open(name, m, st):
    t = st["open"].get(name)
    if not t:
        return False
    d, e, sl, tp, risk = t["dir"], t["entry"], t["sl"], t["tp"], t["risk"]
    after = m[m.index > pd.Timestamp(t["ts"])]
    result = None
    for k, (ts, row) in enumerate(after.iterrows()):
        hit_sl = row.low <= sl if d == 1 else row.high >= sl
        hit_tp = row.high >= tp if d == 1 else row.low <= tp
        if hit_sl:                       # dono ek candle me lage to SL maano
            result = ("SL", -1.0, ts)
            break
        if hit_tp:
            result = ("TP", RR, ts)
            break
        if k + 1 >= MAX_BARS:
            result = ("TIME", float((row.close - e) * d / risk), ts)
            break
    if not result:
        return False
    kind, r, ts = result
    tl = st["tally"]
    tl["R"] = round(tl["R"] + r, 2)
    if kind == "TP":
        tl["W"] += 1
        head = "✅ TP HIT"
    elif kind == "SL":
        tl["L"] += 1
        head = "❌ SL HIT"
    else:
        tl["T"] += 1
        head = "⏱ TIME EXIT (4 ghante ho gaye)"
    send(f"⚡FAST {head} - {name} {'BUY' if d == 1 else 'SELL'}\n"
         f"Entry: {e:.2f}\nSL: {sl:.2f}\nTP: {tp:.2f}\n"
         f"Result: {r:+.2f}R\n"
         f"FAST record: {tl['W']}W / {tl['L']}L / {tl['T']} time-exit | Total {tl['R']:+.2f}R")
    del st["open"][name]
    st["last"][name] = str(ts)
    return True


def run_once():
    st = load_state()
    changed = False
    for name, sym in SYMBOLS.items():
        try:
            m = build_fast(sym)
            if check_open(name, m, st):
                changed = True
                continue
            if name in st["open"]:
                continue
            wr = stats(backtest(m, name))
            now = pd.Timestamp.now(tz="UTC")
            for k in range(-2, -14, -1):          # sabse naya closed candle pehle
                row, ts = m.iloc[k], m.index[k]
                age = (now - ts).total_seconds() / 60
                last = st["last"].get(name)
                fresh = last is None or ts > pd.Timestamp(last)
                if row.sig != 0 and 1 <= age <= MAX_AGE and fresh:
                    t = new_trade(row, ts)
                    send(make_msg(name, t, row, wr, age))
                    st["open"][name] = t
                    st["last"][name] = str(ts)
                    changed = True
                    break
        except Exception as ex:
            print(name, "error:", ex)
    if changed:
        json.dump(st, open(STATE_FILE, "w"))
    print("fast check done", dt.datetime.utcnow())


def run_backtest():
    for name, sym in SYMBOLS.items():
        st = stats(backtest(build_fast(sym), name))
        print(f"{name} (15m+5m+1m, cost ke baad): trades={st['trades']} "
              f"win_rate={st['win_rate']}% avg_R={st['avg_R']}")
    print("\nNote: 1m data sirf 7 din ka hai, sample bahut chhota hai.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "once"
    if mode == "test":
        send("⚡FAST bot Telegram se connect ho gaya.")
        print("test sent")
    elif mode == "backtest":
        run_backtest()
    else:
        run_once()
  
