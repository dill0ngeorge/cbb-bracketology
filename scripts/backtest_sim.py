"""Backtest the season simulator: start from a past date, simulate to the end,
compare against what actually happened.

Checks: conference wins (RMSE), P(conference 1 seed) and P(auto bid)
(Brier / log loss and calibration). Past seasons use default conference
tournament formats (all teams, standard bracket), since only 2027 formats are
configured, so the auto-bid check is conservative.

Usage: python scripts/backtest_sim.py [--month 1 --day 15 --sims 2000]
"""
import argparse
import datetime as dt

import numpy as np
import pandas as pd

from cbb.models.current import build_model, load_tables
from cbb.sim.formats import FormatBook
from cbb.sim.season import simulate


def actuals(g: pd.DataFrame, teams: pd.DataFrame):
    reg = g[g["game_type"] == "conf_regular"]
    hw = reg["home_score"] > reg["away_score"]
    w = pd.concat([reg.loc[hw, "home_id"], reg.loc[~hw, "away_id"]]).value_counts()
    l = pd.concat([reg.loc[~hw, "home_id"], reg.loc[hw, "away_id"]]).value_counts()
    t = teams[["team_id", "conf"]].copy()
    t["cw"] = t["team_id"].map(w).fillna(0)
    t["cl"] = t["team_id"].map(l).fillna(0)
    t["pct"] = t["cw"] / (t["cw"] + t["cl"]).clip(lower=1)
    best = t.groupby("conf")["pct"].transform("max")
    n_tied = t.assign(top=t["pct"] == best).groupby("conf")["top"].transform("sum")
    t["one_seed"] = np.where(t["pct"] == best, 1 / n_tied, 0.0)   # split credit on ties
    ct = g[g["game_type"] == "conf_tourney"].sort_values("date")
    finals = ct.groupby(ct["home_id"].map(t.set_index("team_id")["conf"])).tail(1)
    champs = np.where(finals["home_score"] > finals["away_score"], finals["home_id"], finals["away_id"])
    t["auto"] = t["team_id"].isin(champs).astype(float)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", type=int, default=1)
    ap.add_argument("--day", type=int, default=15)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seasons", type=int, nargs="*", default=[2023, 2024, 2025, 2026])
    a = ap.parse_args()
    tab = load_tables()
    rows = []
    for s in a.seasons:
        today = dt.date(s if a.month < 8 else s - 1, a.month, a.day)
        model, _, teams = build_model(s, through=today, tables=tab)
        g = tab["games"][tab["games"]["season"] == s]
        played = g[g["date"] < today]
        rem = g[(g["date"] >= today) & g["game_type"].isin(["nonconf", "conf_regular"])]
        out = simulate(model, teams, played, rem, FormatBook({}, set()), today, n_sims=a.sims)
        act = actuals(g, teams)
        m = out.summary.merge(act, on=["team_id", "conf"])
        m["season"] = s
        rows.append(m)
    d = pd.concat(rows)

    def brier(p, y):
        return np.mean((p - y) ** 2)

    def ll(p, y):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    # Naive baselines: every team in a conference equally likely.
    n_conf = d.groupby(["season", "conf"])["team_id"].transform("count")
    naive = 1 / n_conf
    print(f"Start date: {a.month}/{a.day}, seasons {a.seasons}, {a.sims} sims each\n")
    cw_naive = d.groupby(["season", "conf"])["cw"].transform("mean")
    print(f"Conference wins RMSE: model {np.sqrt(np.mean((d.conf_wins - d.cw) ** 2)):.2f}"
          f"  vs. 'everyone average' {np.sqrt(np.mean((cw_naive - d.cw) ** 2)):.2f}")
    for col, y in [("p_conf_1seed", "one_seed"), ("p_auto", "auto")]:
        print(f"{col:13s} Brier {brier(d[col], d[y]):.4f} (naive {brier(naive, d[y]):.4f})"
              f"   log loss {ll(d[col], d[y]):.4f} (naive {ll(naive, d[y]):.4f})")
    bins = [0, .02, .05, .1, .2, .35, .5, .75, 1.0001]
    for col, y in [("p_conf_1seed", "one_seed"), ("p_auto", "auto")]:
        d["bin"] = pd.cut(d[col], bins, include_lowest=True)
        print(f"\nCalibration, {col}:")
        print(d.groupby("bin", observed=True).agg(predicted=(col, "mean"), actual=(y, "mean"),
                                                  teams=(y, "size")).round(3).to_string())


if __name__ == "__main__":
    main()
