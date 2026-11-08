"""Build the frame the FinRL 2026 tutorial builds (examples/FinRL_StockTrading_2026_1_data.py), from the
committed raw file instead of live Yahoo calls, so the evaluation is reproducible and offline.

Deviations from the tutorial, all recorded in prepare_summary.json:
- VIX is merged from a committed file (the tutorial calls Yahoo inside FeatureEngineer.add_vix);
- a constituent with no Yahoo prices during part of its membership (delisted tickers such as DWDP or UTX)
  is marked non-member on those dates instead of raising, so the universe is 29 names on those days.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import pandas as pd

from finrl.config import INDICATORS
from finrl.meta.preprocessor.dow import DOW_30_HISTORICAL_UNIVERSE
from finrl.meta.preprocessor.dow import add_dow_membership
from finrl.meta.preprocessor.preprocessors import FeatureEngineer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--vix", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    raw = raw[raw.tic.isin(DOW_30_HISTORICAL_UNIVERSE)].copy()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=True,
        user_defined_feature=False,
        keep_incomplete_tickers=True,
    )
    processed = fe.preprocess_data(raw)

    vix = pd.read_csv(args.vix)[["date", "close"]].rename(columns={"close": "vix"})
    processed = processed.merge(vix, on="date", how="left").sort_values(["date", "tic"])
    processed["vix"] = processed["vix"].ffill()

    list_ticker = list(DOW_30_HISTORICAL_UNIVERSE)
    list_date = list(pd.date_range(processed["date"].min(), processed["date"].max()).astype(str))
    full = pd.DataFrame(itertools.product(list_date, list_ticker), columns=["date", "tic"]).merge(
        processed, on=["date", "tic"], how="left"
    )
    full = full[full["date"].isin(processed["date"])].sort_values(["date", "tic"])
    full = add_dow_membership(full)

    missing = full[full["is_member"] & full["close"].isna()]
    dropped = missing.groupby("tic")["date"].agg(["min", "max", "count"]).reset_index()
    full.loc[missing.index, "is_member"] = False

    feature_columns = [c for c in full.columns if c not in {"date", "tic", "is_member"}]
    full[feature_columns] = full.groupby("tic", sort=False)[feature_columns].transform(
        lambda values: values.ffill().bfill()
    )
    full[feature_columns] = full[feature_columns].fillna(0)
    full["is_member"] = full["is_member"].astype(int)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    full.to_csv(out / "processed_full.csv.gz", index=False)
    summary = {
        "rows": int(len(full)),
        "tickers": int(full["tic"].nunique()),
        "dates": int(full["date"].nunique()),
        "first": full["date"].min(),
        "last": full["date"].max(),
        "members_per_day_min": int(full.groupby("date")["is_member"].sum().min()),
        "members_per_day_max": int(full.groupby("date")["is_member"].sum().max()),
        "members_without_prices": dropped.to_dict("records"),
        "indicators": INDICATORS,
    }
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
