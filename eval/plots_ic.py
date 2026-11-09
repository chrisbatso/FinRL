#!/usr/bin/env python
"""Figures for the IC test. Static PNGs; palette and chrome follow the dataviz reference palette (light)."""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = pathlib.Path(sys.argv[1])
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LABELS = {"macd_rel": "MACD / close", "boll_pos": "Bollinger position", "rsi_30": "RSI 30", "cci_30": "CCI 30",
          "dx_30": "DX 30", "sma30_rel": "close / SMA 30", "sma60_rel": "close / SMA 60",
          "mom_12_1": "12-1 momentum (reference)", "rev_1m": "1-month reversal (reference)"}
ORDER = list(LABELS)
HCOL = {1: BLUE, 5: ORANGE, 21: AQUA}
HLAB = {1: "1 day", 5: "5 days", 21: "21 days"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def style(ax):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def ic_bars():
    cs = pd.read_csv(OUT / "ic_cross_sectional.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    titles = {"train": "Training window, Jan 2014 to Dec 2021", "test": "Test window, Jan 2022 to Mar 2026"}
    y = np.arange(len(ORDER))
    offs = {1: 0.26, 5: 0.0, 21: -0.26}
    mdes = {}
    for ax, w in zip(axes, ["train", "test"]):
        d = cs[(cs.window == w) & cs.feature.isin(ORDER)].set_index(["feature", "horizon"])
        for h in (1, 5, 21):
            m = d.xs(h, level="horizon").loc[ORDER]
            ax.barh(y + offs[h], m.mean_ic, height=0.22, color=HCOL[h], label=HLAB[h], zorder=3)
            ax.errorbar(m.mean_ic, y + offs[h], xerr=1.96 * m.se_nw, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=0, zorder=4)
        mde = d.xs(1, level="horizon").loc[[f for f in ORDER if f in STATF]].mde_80.median()
        for s in (-1, 1):
            ax.axvline(s * mde, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)), zorder=2)
        mdes[w] = mde
        ax.axvline(0, color=BASE, linewidth=1.0, zorder=2)
        ax.axhline(6.5, color=GRID, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[f] for f in ORDER])
        ax.set_title(titles[w], loc="left", fontsize=10, color=INK)
        ax.set_xlabel("mean daily cross-sectional rank IC")
        ax.set_xlim(-0.08, 0.08)
        ax.set_ylim(len(ORDER) - 0.5, -0.7)
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.99, 0.985), title="forward return horizon", title_fontsize=8)
    fig.text(0.01, 0.925, "Bars: mean over the window's trading days, with 95 % Newey-West intervals.", fontsize=8.5, color=INK2)
    fig.text(0.01, 0.895, f"Dotted lines: smallest 1-day IC detectable at 80 % power, {mdes['train']:.3f} in training and {mdes['test']:.3f} in test.", fontsize=8.5, color=INK2)
    fig.suptitle("How much do FinRL's state features know about which Dow stock does better next?", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "ic_bars.png", dpi=180)
    plt.close(fig)


def ic_cumulative():
    daily = pd.read_csv(OUT / "cs_ic_daily_h1.csv", index_col=0, parse_dates=True)
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.5), sharex=True, sharey=True)
    split = pd.Timestamp("2022-01-01")
    for ax, f in zip(axes.flat, ORDER):
        s = daily[f].fillna(0).cumsum()
        ax.axvspan(split, daily.index[-1], color=GRID, alpha=0.55, linewidth=0, zorder=1)
        ax.axhline(0, color=BASE, linewidth=1.0, zorder=2)
        ax.plot(s.index, s.values, color=BLUE if f not in ("mom_12_1", "rev_1m") else ORANGE, linewidth=1.4, zorder=3)
        ax.text(s.index[-1], s.iloc[-1], f" {s.iloc[-1]:+.0f}", va="center", fontsize=8, color=INK2)
        ax.set_title(LABELS[f], loc="left", fontsize=9.5, color=INK)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0, 0].text(split, axes[0, 0].get_ylim()[1] * 0.95 if axes[0, 0].get_ylim()[1] > 0 else 0, "  test window", fontsize=7.5, color=MUTED, va="top")
    fig.suptitle("Cumulative daily cross-sectional rank IC, 1-day horizon, Dow members 2014 to 2026", x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.935, "A feature with information climbs steadily. Random walks wander. The shaded span is the test window; the number is the final sum (about 3,000 days).", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "ic_cumulative.png", dpi=180)
    plt.close(fig)


def null_dots():
    p = OUT / "null_reconstruction.csv"
    if not p.exists():
        return
    nr = pd.read_csv(p)
    ts = nr[nr.kind == "ts"]
    names = ["close", "close_30_sma", "boll_pos", "rsi_30", "sma30_rel", "macd_rel", "cci_30", "dx_30", "rev_1m", "mom_12_1"]
    lab = {**LABELS, "close": "price level (raw state)", "close_30_sma": "SMA 30 level (raw state)"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    y = np.arange(len(names))
    for ax, w in zip(axes, ["train", "test"]):
        d = ts[(ts.window == w) & (ts.horizon == 21)].set_index("name").loc[names]
        ax.hlines(y, d.null_q05, d.null_q95, color=GRID, linewidth=7, zorder=2, label="random-walk null, 5th to 95th percentile")
        ax.scatter(d.null_mean, y, marker="|", color=MUTED, s=120, zorder=3, label="null mean")
        ax.scatter(d.observed, y, color=BLUE, s=34, zorder=4, label="observed", edgecolor=SURFACE, linewidth=1)
        ax.axvline(0, color=BASE, linewidth=1.0, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([lab[n] for n in names])
        ax.set_title({"train": "Training window", "test": "Test window"}[w], loc="left", fontsize=10)
        ax.set_xlabel("mean per-ticker Spearman, feature vs own 21-day forward return")
        style(ax)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.99, 0.945))
    fig.suptitle("Timing a stock with its own indicators: observed values sit inside the random-walk null", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "ts_ic_vs_null.png", dpi=180)
    plt.close(fig)


STATF = {"macd_rel", "boll_pos", "rsi_30", "cci_30", "dx_30", "sma30_rel", "sma60_rel"}
ic_bars()
ic_cumulative()
null_dots()
print("figures written")
