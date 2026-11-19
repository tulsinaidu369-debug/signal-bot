"""
5M bot: same strategy, sirf timeframe change (1m band)
  1H  = trend (price > EMA50 > EMA200)
  15m = bias  (EMA20 vs EMA50 + RSI)
  5m  = entry (EMA20 pullback + candle close)
Alag state file (fast_state.json) aur alag alerts (⚡5M).
TRAILING SL: +1R par SL breakeven, +1.5R par +0.5R lock (alert aate hain). Purana bot (signal_bot.py) waise hi chalta rahega.

Modes:  python signal_bot_fast.py once | test | backtest
"""
import os, sys, json, datetime as dt
import numpy as np
import pandas as pd
from signal_bot import get, ema, rsi, atr, send, SYMBOLS   # helpers purani file se

RR = 2.0
ATR_MULT = 1.5
TRAIL = [(1.0, 0.0), (1.5, 0.5)]   # (price kitne R chala, SL ko kitne R par le jao) - badal sakte ho
MAX_BARS = 96                  # 96 x 5m = 8 ghante, tab tak TP/SL na lage to time-exit
MIN_AGE = 5                    # minute: 5m candle close ho chuki ho
MAX_AGE = 10                   # minute: isse purana signal ignore (candle close ke ~5 min baad tak)
DRIFT_MAX = 0.3                # price entry se itne R se zyada hil chuka ho to alert nahi
COST = {"BTC": 0.0008, "GOLD": 0.0003}   # round-trip spread/fees
STATE_FILE = "fast_state.json"


def build_fast(sym):
    t = get(sym, "1h", "730d")                       # trend
    e50, e200 = ema(t.close, 50), ema(t.close, 200)
    t["trend"] = np.where((t.close > e50) & (e50 > e200), 1,
                  np.where((t.close < e50) & (e50 < e200), -1, 0))
    t = t[["trend"]].shift(1).dropna()

    b = get(sym, "15m", "60d")                       # bias
    b20, b50, br = ema(b.close, 20), ema(b.close, 50), rsi(b.close)
    b["bias"] = np.where((b20 > b50) & (br > 50), 1,
                 np.where((b20 < b50) & (br < 50), -1, 0))
    b["h_rsi"] = br
    b = b[["bias", "h_rsi"]].shift(1).dropna()

    m = get(sym, "5m", "60d")                        # entry
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


def sim_trade(c, hi, lo, i, d, e, risk, rr, maxbars, trail):
    """Entry candle i ke baad candle-by-candle chalata hai. Return (R, exit_index)."""
    n = len(c)
    sl_r, stage = -1.0, 0
    tp = e + d * risk * rr
    last = min(i + maxbars, n - 1)
    for j in range(i + 1, last + 1):
        sl = e + d * sl_r * risk
        hit_sl = lo[j] <= sl if d == 1 else hi[j] >= sl
        hit_tp = hi[j] >= tp if d == 1 else lo[j] <= tp
        if hit_sl:                       # pehle purane SL se check (conservative)
            return sl_r, j
        if hit_tp:
            return rr, j
        while stage < len(trail):        # candle ke baad SL aage badhao
            trig, new = trail[stage]
            tr_price = e + d * trig * risk
            reached = hi[j] >= tr_price if d == 1 else lo[j] <= tr_price
            if not reached:
                break
            sl_r, stage = new, stage + 1
    return (c[last] - e) * d / risk, last


def backtest(m, name, trail=TRAIL):
    """Cost ke baad R list. trail=[] to bina trailing."""
    c, hi, lo = m.close.values, m.high.values, m.low.values
    a, s = m.atr.values, m.sig.values
    res, i, n = [], 0, len(m)
    while i < n - 1:
        if s[i] == 0 or a[i] <= 0:
            i += 1
            continue
        e, risk, d = c[i], a[i] * ATR_MULT, s[i]
        r, j = sim_trade(c, hi, lo, i, d, e, risk, RR, MAX_BARS, trail)
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
    st["tally"].setdefault("B", 0)
    return st


def new_trade(row, ts):
    d = int(row.sig)
    e, risk = float(row.close), float(row.atr * ATR_MULT)
    return {"dir": d, "entry": e, "risk": risk, "sl": e - d * risk,
            "tp": e + d * risk * RR, "ts": str(ts), "stage": 0, "sl0": e - d * risk}


def make_msg(name, t, row, wr, age, cur, drift):
    d = t["dir"]
    return (f"⚡5M {'🟢 BUY' if d == 1 else '🔴 SELL'} {name}\n"
            f"Entry: {t['entry']:.2f}\nSL: {t['sl']:.2f}\nTP: {t['tp']:.2f} (RR 1:{RR})\n"
            f"Confluence score: {int(row.score)}/100\n"
            f"Backtest (60 din, cost ke baad): win {wr['win_rate']}% | avgR {wr['avg_R']} | {wr['trades']} trades\n"
            f"TF: 1H + 15m + 5m aligned\n"
            f"Signal {age:.0f} min purana | Abhi price {cur:.2f} (entry se {drift:+.2f}R)\n"
            f"Trailing: +1R par SL entry par, +1.5R par SL +0.5R par (alert aayega, SL tab khud badalna)\n"
            f"(Is trade ka TP/SL result alag message me aayega)")


def check_open(name, m, st):
    """Open trade: trailing SL alert + TP/SL/time exit result. True = state badla."""
    t = st["open"].get(name)
    if not t:
        return False
    d, e, tp, risk = t["dir"], t["entry"], t["tp"], t["risk"]
    after = m[m.index > pd.Timestamp(t["ts"])]
    sl_r, stage, result = -1.0, 0, None
    for k, (ts, row) in enumerate(after.iterrows()):
        sl = e + d * sl_r * risk
        hit_sl = row.low <= sl if d == 1 else row.high >= sl
        hit_tp = row.high >= tp if d == 1 else row.low <= tp
        if hit_sl:                       # pehle purane SL se check (conservative)
            result = ("SL", sl_r, ts)
            break
        if hit_tp:
            result = ("TP", RR, ts)
            break
        if k + 1 >= MAX_BARS:
            result = ("TIME", float((row.close - e) * d / risk), ts)
            break
        while stage < len(TRAIL):
            trig, new = TRAIL[stage]
            tr_price = e + d * trig * risk
            reached = row.high >= tr_price if d == 1 else row.low <= tr_price
            if not reached:
                break
            sl_r, stage = new, stage + 1

    side = "BUY" if d == 1 else "SELL"
    if not result:
        if stage > t.get("stage", 0):    # naya trailing level -> alert
            trig, new = TRAIL[stage - 1]
            new_sl = e + d * sl_r * risk
            label = "breakeven (entry price)" if new == 0 else f"+{new}R profit lock"
            send(f"⚡5M 🔒 SL TRAIL - {name} {side}\n"
                 f"Price +{trig}R tak pahunch gaya.\n"
                 f"Ab SL ko {new_sl:.2f} par le jao ({label}).\n"
                 f"TP wahi rahega: {tp:.2f}\n"
                 f"Purana SL: {t.get('sl0', e - d * risk):.2f}")
            t["stage"], t["sl"] = stage, new_sl
            return True
        return False

    kind, r, ts = result
    tl = st["tally"]
    tl["R"] = round(tl["R"] + r, 2)
    if kind == "TP":
        tl["W"] += 1
        head = "✅ TP HIT"
    elif kind == "SL":
        if r < -0.5:
            tl["L"] += 1
            head = "❌ SL HIT"
        elif r < 0.05:
            tl["B"] += 1
            head = "⚖️ BREAKEVEN EXIT (SL entry par laga, loss nahi)"
        else:
            tl["W"] += 1
            head = "🔒 TRAILING SL HIT (profit lock hua)"
    else:
        tl["T"] += 1
        head = "⏱ TIME EXIT (8 ghante ho gaye)"
    send(f"⚡5M {head} - {name} {side}\n"
         f"Entry: {e:.2f}\nSL (shuru ka): {t.get('sl0', e - d * risk):.2f}\nTP: {tp:.2f}\n"
         f"Result: {r:+.2f}R\n"
         f"5M record: {tl['W']}W / {tl['L']}L / {tl['B']} breakeven / {tl['T']} time-exit | Total {tl['R']:+.2f}R")
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
                if row.sig != 0 and MIN_AGE <= age <= MAX_AGE and fresh:
                    t = new_trade(row, ts)
                    cur = float(m.close.iloc[-1])
                    drift = (cur - t["entry"]) * t["dir"] / t["risk"]   # + = price aapke favor me
                    if abs(drift) > DRIFT_MAX:                          # price bhaag chuka -> alert bekaar
                        st["last"][name] = str(ts)
                        changed = True
                        print(name, "signal skip: price", round(drift, 2), "R door")
                        break
                    send(make_msg(name, t, row, wr, age, cur, drift))
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
        m = build_fast(sym)
        for label, tr in (("trailing OFF", []), ("trailing ON ", TRAIL)):
            st = stats(backtest(m, name, tr))
            print(f"{name} {label}: trades={st['trades']} win_rate={st['win_rate']}% avg_R={st['avg_R']}")
    print("\nNote: 5m data ~60 din ka hai. Cost (spread/fees) minus hai. Past result future ki guarantee nahi.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "once"
    if mode == "test":
        send("⚡5M bot Telegram se connect ho gaya.")
        print("test sent")
    elif mode == "backtest":
        run_backtest()
    else:
        run_once()
      
