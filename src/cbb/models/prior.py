"""Preseason prior: predict each team's season rating before it plays a game.

Target: the team's full-season ridge rating (offense and defense, relative to
league average that season).

Features for team T in season s (all from season s-1 unless noted):
  prev_o, prev_d, prev_em      T's final ratings last season
  prev2_em                     two seasons ago
  conf_prev_em                 mean prev_em of T's conference *this* season
  ret_min                      share of T's minutes returning to T
  ret_gmsc                     share of T's game-score production returning
  tr_in_share                  incoming D1 transfers' minute shares (sum; 1 = a full-time player)
  tr_in_gmsc40                 their minutes-weighted game score per 40
  tr_in_src_em                 their minutes-weighted former-team rating (level of competition)
  new_d1                       1 if T was not D1 last season
  prev_x_ret, tr_x_src, tr_x_gmsc   interactions (returning production matters
                               more for teams far from average, etc.)

Rosters for season s are players who have appeared for T in s. For past seasons
that is the full roster; for the current season it fills in over the first
week or two. Teams with no games yet fall back to a model without roster terms.

The fitted model's cross-validated error is the prior's standard deviation,
so the data decides how hard the prior pulls.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

from cbb.config import PROCESSED_DIR, SEASONS
from cbb.models.ridge import fit_ridge

ROSTER_FEATURES = ["ret_min", "ret_gmsc", "tr_in_share", "tr_in_gmsc40", "tr_in_src_em",
                   "prev_x_ret", "tr_x_src", "tr_x_gmsc"]
BASE_FEATURES = ["prev_o", "prev_d", "prev_em", "prev2_em", "conf_prev_em", "new_d1"]
FULL_FEATURES = BASE_FEATURES + ROSTER_FEATURES
TARGETS = ["o_rel", "d_rel"]


# ------------------------------------------------------------- final ratings
def season_final_ratings(tg: pd.DataFrame, seasons=SEASONS, force=False) -> pd.DataFrame:
    """Full-season ridge ratings, relative to that season's league average."""
    path = PROCESSED_DIR / "final_ratings.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    out = []
    for s in seasons:
        m = fit_ridge(tg[tg["season"] == s])
        r = m.ratings.copy()
        r["o_rel"] = r["adj_o"] - m.mu
        r["d_rel"] = r["adj_d"] - m.mu
        r["em"] = r["adj_em"]
        r.insert(0, "season", s)
        out.append(r[["season", "team_id", "o_rel", "d_rel", "em", "adj_t"]])
    fr = pd.concat(out, ignore_index=True)
    fr.to_parquet(path, index=False)
    return fr


# ------------------------------------------------------------------ features
def build_features(final: pd.DataFrame, ps: pd.DataFrame, teams: pd.DataFrame,
                   seasons) -> pd.DataFrame:
    rows = []
    for s in seasons:
        t = teams.loc[teams["season"] == s, ["team_id", "conf_id", "conf"]].copy()
        prev = final[final["season"] == s - 1].set_index("team_id")
        prev2 = final[final["season"] == s - 2].set_index("team_id")
        t["prev_o"] = t["team_id"].map(prev["o_rel"])
        t["prev_d"] = t["team_id"].map(prev["d_rel"])
        t["prev_em"] = t["team_id"].map(prev["em"])
        t["prev2_em"] = t["team_id"].map(prev2["em"])
        t["new_d1"] = t["prev_em"].isna().astype(int)
        # Teams new to D1: start at the historical average of new teams (very low).
        t["prev_em"] = t["prev_em"].fillna(-15.0)
        t["prev_o"] = t["prev_o"].fillna(-7.5)
        t["prev_d"] = t["prev_d"].fillna(7.5)
        t["prev2_em"] = t["prev2_em"].fillna(t["prev_em"])
        t["conf_prev_em"] = t.groupby("conf_id")["prev_em"].transform("mean")

        last = ps[ps["season"] == s - 1]
        cur = ps[ps["season"] == s]
        last_team_tot = last.groupby("team_id")[["min", "gmsc"]].sum()
        # Returning: same athlete, same team, both seasons.
        ret = last.merge(cur[["athlete_id", "team_id"]], on=["athlete_id", "team_id"])
        ret_tot = ret.groupby("team_id")[["min", "gmsc"]].sum()
        t["ret_min"] = t["team_id"].map(ret_tot["min"] / last_team_tot["min"])
        t["ret_gmsc"] = t["team_id"].map(
            (ret_tot["gmsc"] / last_team_tot["gmsc"].clip(lower=1)).clip(-0.5, 1.5))
        # Transfers in: played for a different D1 team last season.
        src = last.sort_values("min", ascending=False).drop_duplicates("athlete_id")
        tr = cur[["athlete_id", "team_id"]].merge(
            src[["athlete_id", "team_id", "min_share", "gmsc40"]].rename(
                columns={"team_id": "src_team"}), on="athlete_id")
        tr = tr[tr["team_id"] != tr["src_team"]]
        tr["src_em"] = tr["src_team"].map(prev["em"]).fillna(-10)
        w = tr["min_share"]
        tr = tr.assign(w_gmsc=w * tr["gmsc40"], w_src=w * tr["src_em"])
        g = tr.groupby("team_id")
        share = g["min_share"].sum()
        t["tr_in_share"] = t["team_id"].map(share)
        t["tr_in_gmsc40"] = t["team_id"].map(g["w_gmsc"].sum() / share.clip(lower=1e-6))
        t["tr_in_src_em"] = t["team_id"].map(g["w_src"].sum() / share.clip(lower=1e-6))
        has_roster = t["team_id"].isin(cur["team_id"])
        t["has_roster"] = has_roster
        # No transfers in -> zero share; level features at neutral values.
        t.loc[has_roster, "tr_in_share"] = t.loc[has_roster, "tr_in_share"].fillna(0)
        t.loc[has_roster, "tr_in_gmsc40"] = t.loc[has_roster, "tr_in_gmsc40"].fillna(0)
        t.loc[has_roster, "tr_in_src_em"] = t.loc[has_roster, "tr_in_src_em"].fillna(0)
        t.loc[has_roster, "ret_min"] = t.loc[has_roster, "ret_min"].fillna(0)
        t.loc[has_roster, "ret_gmsc"] = t.loc[has_roster, "ret_gmsc"].fillna(0)
        t["tr_in_share"] = t["tr_in_share"].clip(upper=5)
        t["prev_x_ret"] = t["prev_em"] * t["ret_gmsc"]
        t["tr_x_src"] = t["tr_in_share"] * t["tr_in_src_em"]
        t["tr_x_gmsc"] = t["tr_in_share"] * t["tr_in_gmsc40"]
        t.insert(0, "season", s)
        rows.append(t)
    f = pd.concat(rows, ignore_index=True)
    return f.merge(final[["season", "team_id", "o_rel", "d_rel", "em"]],
                   on=["season", "team_id"], how="left")


# --------------------------------------------------------------------- model
@dataclass
class PriorModel:
    full: dict      # target -> fitted model using roster features
    base: dict      # target -> fitted model without roster features
    sd: dict        # target -> prior SD (cross-validated RMSE)

    def predict(self, f: pd.DataFrame) -> pd.DataFrame:
        out = f[["season", "team_id"]].copy()
        has = f["has_roster"].values
        for tgt in TARGETS:
            pred = self.base[tgt].predict(f[BASE_FEATURES])
            if has.any():
                pred[has] = self.full[tgt].predict(f.loc[has, FULL_FEATURES])
            out[f"prior_{tgt}"] = pred
        out["prior_em"] = out["prior_o_rel"] - out["prior_d_rel"]
        out["prior_sd_o"] = self.sd["o_rel"]
        out["prior_sd_d"] = self.sd["d_rel"]
        return out


def _make(kind: str):
    if kind == "linear":
        return RidgeCV(alphas=np.logspace(-2, 3, 20))
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3,
                                         min_samples_leaf=30, l2_regularization=1.0)


def fit_prior(train: pd.DataFrame, kind: str = "linear", sd=None) -> PriorModel:
    train = train.dropna(subset=TARGETS)
    full = {t: _make(kind).fit(train[FULL_FEATURES], train[t]) for t in TARGETS}
    base = {t: _make(kind).fit(train[BASE_FEATURES], train[t]) for t in TARGETS}
    return PriorModel(full, base, sd or {t: np.nan for t in TARGETS})


def leave_season_out(feats: pd.DataFrame, kind="linear", strict_past=False) -> pd.DataFrame:
    """Out-of-sample prior predictions per season. strict_past: train only on earlier seasons."""
    out = []
    feats = feats.dropna(subset=TARGETS)
    for s in sorted(feats["season"].unique()):
        tr = feats[(feats["season"] < s) if strict_past else (feats["season"] != s)]
        if tr["season"].nunique() < 2:
            continue
        m = fit_prior(tr, kind)
        p = m.predict(feats[feats["season"] == s])
        out.append(p.merge(feats[["season", "team_id", "o_rel", "d_rel", "em", "prev_em"]],
                           on=["season", "team_id"]))
    return pd.concat(out, ignore_index=True)
