"""Exact posterior of the dynamic Bayesian rating model via a Kalman filter.

Same model as bayes_pymc.py; because it is linear-Gaussian, the filtering
posterior p(ratings today | all games so far) is a multivariate normal that the
Kalman filter computes exactly. Hyperparameters (sigma, tau, k) come from the
PyMC fits.

State (efficiency):  [o_1..o_K, d_1..d_K, mu]   (offense/defense relative to mu)
State (tempo):       [t_1..t_K, mu_t]
Process noise per day: tau^2 / 7 on every o and d (tempo: tau_t^2 / 7).
Observation per team-game: off_eff - HCA*loc = mu + o[team] + d[opp] + noise(sigma).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from cbb.config import HOME_COURT_PER100

# sigma, tau, k: posterior means from PyMC fits of 2023, 2025, 2026, averaged
#   (2023: 12.58/0.68/0.76, 2025: 12.58/0.73/0.79, 2026: 12.79/0.62/0.88).
# margin_noise: game-level margin noise in points, tuned for calibration on the
#   2018-2026 walk-forward (log loss minimized near 9.7).
# Tempo terms are set by hand; tempo only scales margins.
DEFAULT_HYPER = dict(sigma=12.65, tau=0.68, k=0.81, mu_sd=3.0, mu_tau=0.1,
                     sigma_t=5.0, tau_t=0.3, tempo_sd=2.5, tempo_regress=0.75,
                     margin_noise=9.7)


@dataclass
class KalmanRatings:
    teams: np.ndarray
    hyper: dict = field(default_factory=lambda: dict(DEFAULT_HYPER))
    hca: float = HOME_COURT_PER100

    # -------------------------------------------------------------- init
    def init(self, prior: pd.DataFrame, mu0: float = 104.0, prior_sd_o=4.3, prior_sd_d=4.1,
             prev_tempo: pd.Series | None = None, mu_t0: float = 68.5):
        h, K = self.hyper, len(self.teams)
        self.idx = {t: i for i, t in enumerate(self.teams)}
        pr = prior.set_index("team_id").reindex(self.teams)
        self.m = np.concatenate([pr["prior_o_rel"].fillna(0).values,
                                 pr["prior_d_rel"].fillna(0).values, [mu0]])
        self.P = np.diag(np.concatenate([np.full(K, (h["k"] * prior_sd_o) ** 2),
                                         np.full(K, (h["k"] * prior_sd_d) ** 2),
                                         [h["mu_sd"] ** 2]]))
        t0 = np.zeros(K)
        if prev_tempo is not None:
            t0 = h["tempo_regress"] * prev_tempo.reindex(self.teams).fillna(0).values
        self.mt = np.concatenate([t0, [mu_t0]])
        self.Pt = np.diag(np.concatenate([np.full(K, h["tempo_sd"] ** 2), [2.0 ** 2]]))
        self.last_date = None
        return self

    # ----------------------------------------------------------- predict
    def _advance(self, date):
        if self.last_date is not None:
            days = (pd.Timestamp(date) - pd.Timestamp(self.last_date)).days
            if days > 0:
                K, h = len(self.teams), self.hyper
                q = np.full(2 * K + 1, h["tau"] ** 2 / 7 * days)
                q[-1] = h["mu_tau"] ** 2 / 7 * days
                self.P[np.diag_indices_from(self.P)] += q
                qt = np.full(K + 1, h["tau_t"] ** 2 / 7 * days)
                qt[-1] = h["mu_tau"] ** 2 / 7 * days
                self.Pt[np.diag_indices_from(self.Pt)] += qt
        self.last_date = date

    @staticmethod
    def _update(m, P, H, y, r2):
        S = H @ P @ H.T + r2 * np.eye(len(y))
        PHt = P @ H.T
        Kg = np.linalg.solve(S, PHt.T).T
        m = m + Kg @ (y - H @ m)
        P = P - Kg @ PHt.T
        return m, (P + P.T) / 2

    # ------------------------------------------------------------ update
    def update_day(self, day: pd.DataFrame):
        """day: team_games rows (both sides of each game) for one date."""
        self._advance(day["date"].iloc[0])
        d = day[day["has_box"] & day["team_id"].isin(self.idx) & day["opp_id"].isin(self.idx)]
        if d.empty:
            return
        K, n = len(self.teams), len(d)
        ti = d["team_id"].map(self.idx).values
        ui = d["opp_id"].map(self.idx).values
        H = np.zeros((n, 2 * K + 1))
        H[np.arange(n), ti] = 1
        H[np.arange(n), K + ui] = 1
        H[:, -1] = 1
        y = d["off_eff"].values - self.hca * d["loc"].values
        self.m, self.P = self._update(self.m, self.P, H, y, self.hyper["sigma"] ** 2)

        g = d[d["loc"] >= 0].drop_duplicates("game_id")
        gi, go = g["team_id"].map(self.idx).values, g["opp_id"].map(self.idx).values
        Ht = np.zeros((len(g), K + 1))
        Ht[np.arange(len(g)), gi] += 1
        Ht[np.arange(len(g)), go] += 1
        Ht[:, -1] = 1
        self.mt, self.Pt = self._update(self.mt, self.Pt, Ht, g["poss"].values,
                                        self.hyper["sigma_t"] ** 2)

    def run(self, tg: pd.DataFrame):
        for _, day in tg.sort_values("date").groupby("date", sort=True):
            self.update_day(day)
        return self

    # ------------------------------------------------------------ output
    @property
    def ratings(self) -> pd.DataFrame:
        K = len(self.teams)
        mu, sd = self.m[-1], np.sqrt(np.diag(self.P))
        em_var = (np.diag(self.P)[:K] + np.diag(self.P)[K:2 * K]
                  - 2 * self.P[np.arange(K), K + np.arange(K)])
        return pd.DataFrame({
            "team_id": self.teams,
            "adj_o": mu + self.m[:K], "adj_d": mu + self.m[K:2 * K],
            "adj_em": self.m[:K] - self.m[K:2 * K],
            "em_sd": np.sqrt(em_var), "o_sd": sd[:K], "d_sd": sd[K:2 * K],
            "adj_t": self.mt[-1] + self.mt[:K],
        })

    def predict(self, team_id, opp_id, loc):
        """Expected margin (points), win prob. Uncertainty = rating posterior + game noise."""
        K = len(self.teams)
        a = pd.Series(np.asarray(team_id)).map(self.idx)
        b = pd.Series(np.asarray(opp_id)).map(self.idx)
        ok = a.notna() & b.notna()
        a = a.fillna(0).astype(int).values
        b = b.fillna(0).astype(int).values
        loc = np.asarray(loc, dtype=float)
        o, d = self.m[:K], self.m[K:2 * K]
        diff100 = (o[a] + d[b]) - (o[b] + d[a]) + 2 * self.hca * loc
        # Var of (o_a + d_b - o_b - d_a) from the posterior covariance.
        ia, ib, ja, jb = a, K + b, b, K + a  # +, +, -, -
        P = self.P
        var = (P[ia, ia] + P[ib, ib] + P[ja, ja] + P[jb, jb]
               + 2 * P[ia, ib] - 2 * P[ia, ja] - 2 * P[ia, jb]
               - 2 * P[ib, ja] - 2 * P[ib, jb] + 2 * P[ja, jb])
        poss = self.mt[-1] + self.mt[:K][a] + self.mt[:K][b]
        margin = diff100 * poss / 100
        sd = np.sqrt(var * (poss / 100) ** 2 + self.hyper["margin_noise"] ** 2)
        margin = np.where(ok, margin, 0.0)
        return margin, norm.cdf(margin / sd)

    def sample(self, n: int, rng=None) -> np.ndarray:
        """Draw n joint samples of [o, d, mu] from the posterior (for simulation)."""
        rng = rng or np.random.default_rng()
        L = np.linalg.cholesky(self.P + 1e-9 * np.eye(len(self.m)))
        return self.m + rng.standard_normal((n, len(self.m))) @ L.T
