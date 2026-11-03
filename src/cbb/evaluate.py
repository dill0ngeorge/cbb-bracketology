"""Walk-forward evaluation: refit weekly, predict the next week's games.

Any rating model with fit(team_games_so_far) -> obj.predict(team, opp, loc)
can be dropped in here, so the Bayesian model gets scored the same way.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def walk_forward(tg: pd.DataFrame, fit_fn, start=(12, 1), step_days: int = 7,
                 seasons=None) -> pd.DataFrame:
    """Return one row per predicted game with pred margin, win prob, actual result."""
    out = []
    seasons = seasons or sorted(tg["season"].unique())
    for s in seasons:
        st = tg[tg["season"] == s]
        if st.empty:
            continue
        cut = dt.date(s - 1 if start[0] >= 8 else s, *start)
        end = st["date"].max()
        while cut <= end:
            train = st[st["date"] < cut]
            nxt = cut + dt.timedelta(days=step_days)
            test = st[(st["date"] >= cut) & (st["date"] < nxt) & (st["loc"] >= 0)]
            test = test.drop_duplicates("game_id")
            if len(train) > 200 and len(test):
                model = fit_fn(train)
                m, p = model.predict(test["team_id"], test["opp_id"], test["loc"])
                out.append(test[["season", "date", "game_id", "team_id", "opp_id", "loc",
                                 "game_type", "margin", "win"]].assign(pred_margin=m, p_win=p))
            cut = nxt
    return pd.concat(out, ignore_index=True)


def score(preds: pd.DataFrame, by=None) -> pd.DataFrame:
    def _s(d):
        p = np.clip(d["p_win"], 1e-6, 1 - 1e-6)
        return pd.Series({
            "games": len(d),
            "margin_rmse": np.sqrt(np.mean((d["margin"] - d["pred_margin"]) ** 2)),
            "margin_mae": np.mean(np.abs(d["margin"] - d["pred_margin"])),
            "accuracy": np.mean((d["p_win"] > 0.5) == (d["win"] == 1)),
            "log_loss": -np.mean(d["win"] * np.log(p) + (1 - d["win"]) * np.log(1 - p)),
            "brier": np.mean((d["p_win"] - d["win"]) ** 2),
        })
    if by is None:
        return _s(preds).to_frame("all").T
    return preds.groupby(by).apply(_s, include_groups=False)
