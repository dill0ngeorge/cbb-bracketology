"""Full Bayesian dynamic rating model in PyMC.

Used to learn the model's hyperparameters from past seasons; the daily
ratings come from the exact Kalman-filter posterior of the same model
(see kalman.py).

For team-game i (team t, opponent u, week w):
    off_eff_i ~ Normal(mu + o[w, t] + d[w, u] + HCA * loc_i, sigma)

    o[0, t] ~ Normal(prior_o[t], k * prior_sd_o)        preseason prior
    o[w, t] = o[w-1, t] + tau * z[w, t],  z ~ Normal(0, 1)   weekly random walk
    (d is identical, with its own prior)

Hyperparameters learned:
    sigma  single-game noise in points per 100 possessions
    tau    weekly drift: how fast true team quality changes in-season
    k      prior scale: >1 means the preseason prior is overconfident
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm

from cbb.config import HOME_COURT_PER100


def prepare(tg: pd.DataFrame, prior: pd.DataFrame):
    d = tg[tg["has_box"]].copy()
    teams = np.unique(np.concatenate([d["team_id"], d["opp_id"]]))
    idx = {t: i for i, t in enumerate(teams)}
    start = pd.to_datetime(d["date"]).min()
    week = ((pd.to_datetime(d["date"]) - start).dt.days // 7).values
    pr = prior.set_index("team_id").reindex(teams)
    return dict(
        y=d["off_eff"].values - HOME_COURT_PER100 * d["loc"].values,
        ti=d["team_id"].map(idx).values, ui=d["opp_id"].map(idx).values, week=week,
        n_teams=len(teams), n_weeks=int(week.max()) + 1, teams=teams,
        prior_o=pr["prior_o_rel"].fillna(0).values, prior_d=pr["prior_d_rel"].fillna(0).values,
        sd_o=float(prior["prior_sd_o"].iloc[0]), sd_d=float(prior["prior_sd_d"].iloc[0]),
    )


def build_model(data: dict) -> pm.Model:
    K, W = data["n_teams"], data["n_weeks"]
    coords = {"team": data["teams"], "week": np.arange(W)}
    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu", 105, 10)
        sigma = pm.HalfNormal("sigma", 20)
        tau = pm.HalfNormal("tau", 1.0)
        k = pm.HalfNormal("k", 2.0)

        z0_o = pm.Normal("z0_o", 0, 1, dims="team")
        z0_d = pm.Normal("z0_d", 0, 1, dims="team")
        o0 = data["prior_o"] + k * data["sd_o"] * z0_o
        d0 = data["prior_d"] + k * data["sd_d"] * z0_d

        # Non-centered random walk; week 0 has no innovation.
        zo = pm.Normal("zo", 0, 1, shape=(W - 1, K))
        zd = pm.Normal("zd", 0, 1, shape=(W - 1, K))
        pad = pm.math.zeros((1, K))
        o = pm.Deterministic("o", o0 + tau * pm.math.concatenate([pad, zo], 0).cumsum(0),
                             dims=("week", "team"))
        dd = pm.Deterministic("d", d0 + tau * pm.math.concatenate([pad, zd], 0).cumsum(0),
                              dims=("week", "team"))

        eta = mu + o[data["week"], data["ti"]] + dd[data["week"], data["ui"]]
        pm.Normal("y", eta, sigma, observed=data["y"])
    return model


def fit(tg, prior, draws=500, tune=500, chains=2, seed=0):
    import nutpie
    data = prepare(tg, prior)
    model = build_model(data)
    compiled = nutpie.compile_pymc_model(model)
    idata = nutpie.sample(compiled, draws=draws, tune=tune, chains=chains, seed=seed,
                          progress_bar=False)
    return idata, data
