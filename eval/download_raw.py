"""Download the Dow point-in-time universe from Yahoo, in the layout FinRL's YahooDownloader.fetch_data returns.

Own re-implementation of finrl/meta/preprocessor/yahoodownloader.py (auto_adjust=False, then OHLC scaled by
Adj Close / Close), so the CI jobs never touch the network. Ticker universe parsed as text from dow.py and
config_tickers.py.
"""
import json, pathlib, re, sys, time
import pandas as pd, yfinance as yf

repo, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
dow_src = (repo / "finrl/meta/preprocessor/dow.py").read_text()
tickers = set(re.findall(r'"([A-Z]{1,5})"', dow_src))
cfg = (repo / "finrl/config_tickers.py").read_text()
block = re.search(r"DOW_30_TICKER\s*=\s*\[(.*?)\]", cfg, re.S).group(1)
tickers |= set(re.findall(r'"([A-Z.\-]{1,6})"', block))
tickers = sorted(tickers)
print(len(tickers), "tickers:", " ".join(tickers))

START, END = "2014-01-01", "2026-03-21"
frames, coverage = [], []
for tic in tickers:
    for attempt in range(3):
        try:
            df = yf.download(tic, start=START, end=END, auto_adjust=False, progress=False, threads=False)
            break
        except Exception as e:  # noqa: BLE001
            print("retry", tic, e); time.sleep(5 * (attempt + 1)); df = pd.DataFrame()
    if df.columns.nlevels != 1:
        df.columns = df.columns.droplevel(1)
    if len(df) == 0:
        coverage.append({"tic": tic, "rows": 0, "first": None, "last": None}); print("NO DATA:", tic); continue
    df = df.reset_index().rename(columns={"Date": "date", "Adj Close": "adjcp", "Close": "close", "High": "high", "Low": "low", "Volume": "volume", "Open": "open"})
    adj = df["adjcp"] / df["close"]
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c] * adj
    df = df.drop(columns=["adjcp"])
    df["tic"] = tic
    df["day"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.dropna()
    coverage.append({"tic": tic, "rows": len(df), "first": df.date.min(), "last": df.date.max()})
    frames.append(df[["date", "open", "high", "low", "close", "volume", "tic", "day"]])
    time.sleep(0.5)
raw = pd.concat(frames).sort_values(["date", "tic"]).reset_index(drop=True)
raw.to_csv(out / "raw_prices.csv.gz", index=False)
pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
for sym, name in [("^VIX", "vix"), ("^DJI", "dji")]:
    d = yf.download(sym, start=START, end=END, auto_adjust=False, progress=False, threads=False)
    if d.columns.nlevels != 1:
        d.columns = d.columns.droplevel(1)
    d = d.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    d.to_csv(out / f"{name}.csv", index=False)
    print(name, len(d), d.date.min(), d.date.max())
print("raw rows", len(raw), "dates", raw.date.nunique(), raw.date.min(), raw.date.max())
print(pd.DataFrame(coverage).to_string())
