"""Does down-weighting lopsided games improve the ratings?

Runs the Kalman walk-forward (2018-2026, weekly refits from Nov 15) for several
values of mismatch_c, the strength of the observation-variance inflation for
games the model expects to be lopsided, and compares predictions.

Usage: python scripts/backtest_mismatch.py 0 0.002 0.005 0.01
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


def run(c: float) -> pd.DataFrame:
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
    hyper = dict(DEFAULT_HYPER, mismatch_c=c)

    def fit(train):
        s = int(train.season.iloc[0]); pr, pt, mu0, mt0 = setup[s]
        return (KalmanRatings(teams[teams.season == s].team_id.values, hyper=dict(hyper))
                .init(pr, mu0=mu0, prev_tempo=pt, mu_t0=mt0).run(train))

    p = walk_forward(tg, fit, seasons=SEASONS, start=(11, 15))
    p["c"] = c
    p.to_parquet(D / f"wf_mismatch_{c}.parquet")
    return p


def main():
    cs = [float(x) for x in sys.argv[1:]] or [0.0, 0.002, 0.005, 0.01]
    with ProcessPoolExecutor(2) as ex:
        preds = list(ex.map(run, cs))
    months = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]
    rows, monthly, close = {}, {}, {}
    for p in preds:
        c = p["c"].iloc[0]
        p["month"] = pd.Categorical(pd.to_datetime(p.date).dt.month_name().str[:3], months)
        rows[c] = score(p).iloc[0]
        monthly[c] = score(p, "month")["margin_rmse"]
        close[c] = score(p[p.pred_margin.abs() < 10]).iloc[0]
    print("All games:\n", pd.DataFrame(rows).T.round(4).to_string())
    print("\nClose games only (|predicted margin| < 10):\n",
          pd.DataFrame(close).T[["games", "margin_rmse", "log_loss"]].round(4).to_string())
    print("\nMargin RMSE by month:\n", pd.DataFrame(monthly).round(3).to_string())


if __name__ == "__main__":
    main()
