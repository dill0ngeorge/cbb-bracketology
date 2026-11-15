"""Build the Bayesian rating model for a season as of a given date.

Shared by the daily pipeline, the simulator and backtests, so all three see
exactly the same model. The preseason prior is always trained only on seasons
before `season`.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from cbb.config import PROCESSED_DIR
from cbb.models.kalman import KalmanRatings
from cbb.models.prior import fit_prior


def load_tables() -> dict[str, pd.DataFrame]:
    names = ["team_games", "games", "teams", "prior_features", "final_ratings",
             "schedule_remaining"]
    return {n: pd.read_parquet(PROCESSED_DIR / f"{n}.parquet") for n in names}


def build_model(season: int, through: dt.date | None = None, tables=None):
    """Kalman posterior using games played strictly before `through` (all if None)."""
    t = tables or load_tables()
    tg, f, final = t["team_games"], t["prior_features"], t["final_ratings"]
    teams = t["teams"][t["teams"]["season"] == season]
    prior = fit_prior(f[f["season"] < season]).predict(f[f["season"] == season])
    prev = final[final["season"] == season - 1].set_index("team_id")
    mu0 = tg[(tg["season"] == season - 1) & tg["has_box"]]["off_eff"].mean()
    cur = tg[tg["season"] == season]
    if through is not None:
        cur = cur[cur["date"] < through]
    model = (KalmanRatings(teams["team_id"].values)
             .init(prior, mu0=mu0, prev_tempo=prev["adj_t"] - prev["adj_t"].mean(),
                   mu_t0=prev["adj_t"].mean())
             .run(cur))
    return model, prior, teams
