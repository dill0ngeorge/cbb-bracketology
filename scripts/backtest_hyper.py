"""Backtest one Kalman hyperparameter over a grid of values.

Runs the Kalman walk-forward (2018-2026, weekly refits from Nov 15) for several
values of mismatch_c, the strength of the observation-variance inflation for
games the model expects to be lopsided, and compares predictions.

Usage: python scripts/backtest_hyper.py k 0.81 1.0 1.25 1.5
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from cbb.config import PROCESSED_DIR as D
from cbb.evaluate import score, walk_forward
from cbb.models.kalman import DEFAULT_HYPER, KalmanRatings
from cbb.models.prior import fit_prior

SEASONS = list(range(2018, 2027))


def run(args) -> pd.DataFrame:
    name, c = args
    tg = pd.read_parquet(D / "team_games.parquet")
    f = pd.read_parquet(D / "prior_features.parquet")
    teams = pd.read_parquet(D / "teams.parquet")
    final = pd.read_parquet(D / "final_ratings.parquet")
    setup = {}
    for s in SEASONS:
        pr = fit_prior(f[f.season < s]).predict(f[f.season == s])
        prev = final[final.season == s - 1].set_index("team_id")
        mu0 = tg[(tg.season == s - 1) & tg.has_box].off_eff.mean()
        setup[s] = (pr, prev.adj_t - prev.adj_t.mean(), mu0, prev.adj_t.mean())
    hyper = dict(DEFAULT_HYPER, **{name: c})

    def fit(train):
        s = int(train.season.iloc[0]); pr, pt, mu0, mt0 = setup[s]
        return (KalmanRatings(teams[teams.season == s].team_id.values, hyper=dict(hyper))
                .init(pr, mu0=mu0, prev_tempo=pt, mu_t0=mt0).run(train))

    p = walk_forward(tg, fit, seasons=SEASONS, start=(11, 15))
    p["c"] = c
    p.to_parquet(D / f"wf_{name}_{c}.parquet")
    return p


def main():
    name = sys.argv[1]
    cs = [float(x) for x in sys.argv[2:]]
    with ProcessPoolExecutor(2) as ex:
        preds = list(ex.map(run, [(name, c) for c in cs]))
    months = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]
    for label, seasons in [("2018-2021 (pre-portal)", range(2018, 2022)),
                           ("2022-2026 (portal era)", range(2022, 2027))]:
        rows, monthly = {}, {}
        for p in preds:
            c = p["c"].iloc[0]
            q = p[p.season.isin(seasons)].copy()
            q["month"] = pd.Categorical(pd.to_datetime(q.date).dt.month_name().str[:3], months)
            rows[c] = score(q).iloc[0]
            monthly[c] = score(q, "month")["margin_rmse"]
        print(f"\n=== {label}: {name} ===")
        print(pd.DataFrame(rows).T[["games", "margin_rmse", "log_loss", "brier"]].round(4).to_string())
        print(pd.DataFrame(monthly).round(3).to_string())


if __name__ == "__main__":
    main()
