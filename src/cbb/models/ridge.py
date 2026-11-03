"""Baseline team ratings: opponent-adjusted efficiency via ridge regression.

For every team-game with a box score:
    off_eff[team vs opp] = mu + o[team] + d[opp] + hca * loc + noise
    poss[game]           = mu_t + t[team] + t[opp] + noise

AdjO = mu + o, AdjD = mu + d (points allowed per 100; lower is better),
AdjEM = AdjO - AdjD, AdjT = mu_t + t  (tempo vs an average opponent).

This is the non-Bayesian benchmark the hierarchical model has to beat.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm
from sklearn.linear_model import Ridge


@dataclass
class RidgeRatings:
    ratings: pd.DataFrame  # team_id, adj_o, adj_d, adj_em, adj_t
    mu: float
    mu_t: float
    hca: float             # home-court, points per 100 possessions per side
    sigma: float           # residual SD of game margin (points)

    def predict(self, team_id, opp_id, loc):
        """Expected margin (points) and win prob for team vs opp. loc: +1 home, 0 neutral, -1 away."""
        r = self.ratings.set_index("team_id")
        team_id, opp_id, loc = map(np.asarray, (team_id, opp_id, loc))
        a, b = r.reindex(team_id), r.reindex(opp_id)
        # Teams we have never seen get league-average ratings.
        pa = (a["adj_o"].fillna(self.mu).values + b["adj_d"].fillna(self.mu).values - self.mu
              + self.hca * loc)
        pb = (b["adj_o"].fillna(self.mu).values + a["adj_d"].fillna(self.mu).values - self.mu
              - self.hca * loc)
        poss = a["adj_t"].fillna(self.mu_t).values + b["adj_t"].fillna(self.mu_t).values - self.mu_t
        margin = (pa - pb) * poss / 100
        return margin, norm.cdf(margin / self.sigma)


def _design(tg: pd.DataFrame, team_index: dict[int, int]):
    n, k = len(tg), len(team_index)
    rows = np.arange(n)
    ti = tg["team_id"].map(team_index).values
    oi = tg["opp_id"].map(team_index).values
    ones = np.ones(n)
    X_off = sparse.hstack([
        sparse.csr_matrix((ones, (rows, ti)), shape=(n, k)),
        sparse.csr_matrix((ones, (rows, oi)), shape=(n, k)),
        sparse.csr_matrix(tg["loc"].values.astype(float).reshape(-1, 1)),
    ]).tocsr()
    X_tempo = (sparse.csr_matrix((ones, (rows, ti)), shape=(n, k))
               + sparse.csr_matrix((ones, (rows, oi)), shape=(n, k))).tocsr()
    return X_off, X_tempo


def fit_ridge(tg: pd.DataFrame, alpha: float = 1.0, weights=None) -> RidgeRatings:
    """Fit on a team_games slice (one season, or one season up to a date)."""
    d = tg[tg["has_box"]].copy()
    teams = np.unique(np.concatenate([d["team_id"].values, d["opp_id"].values]))
    idx = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    X_off, X_tempo = _design(d, idx)

    m_off = Ridge(alpha=alpha).fit(X_off, d["off_eff"].values, sample_weight=weights)
    o, dd, hca = m_off.coef_[:k], m_off.coef_[k:2 * k], m_off.coef_[-1]
    mu = m_off.intercept_

    # Each game appears twice (once per team); fine for tempo since it is symmetric.
    m_t = Ridge(alpha=alpha).fit(X_tempo, d["poss"].values, sample_weight=weights)
    t, mu_t = m_t.coef_, m_t.intercept_

    ratings = pd.DataFrame({
        "team_id": teams,
        "adj_o": mu + o,
        "adj_d": mu + dd,
        "adj_t": mu_t + t,
    })
    ratings["adj_em"] = ratings["adj_o"] - ratings["adj_d"]

    model = RidgeRatings(ratings, mu, mu_t, hca, sigma=11.0)
    # Residual SD of margins on the training set (one row per game).
    home_side = tg[tg["loc"] >= 0].drop_duplicates("game_id")
    pred, _ = model.predict(home_side["team_id"], home_side["opp_id"], home_side["loc"])
    model.sigma = float(np.std(home_side["margin"].values - pred))
    return model


def rank_table(model: RidgeRatings, teams: pd.DataFrame) -> pd.DataFrame:
    out = model.ratings.merge(teams[["team_id", "team_short", "conf"]], on="team_id", how="left")
    out = out.sort_values("adj_em", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out
