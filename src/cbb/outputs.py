"""Produce the public outputs: top 40, projected bracket, bubble.

Everything runs from one simulation pass so the three outputs agree.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from cbb.config import ROOT
from cbb.sim.bracket import DEFAULT_ALPHA, last_in_first_out, projected_bracket, select_and_seed
from cbb.sim.resume import compute_resume

OUT_DIR = ROOT / "outputs"
QUAD_COLS = ["q1w", "q1l", "q2w", "q2l", "q3w", "q3l", "q4w", "q4l"]


def record_to_date(played: pd.DataFrame) -> pd.DataFrame:
    hw = played["home_score"] > played["away_score"]
    w = pd.concat([played.loc[hw, "home_id"], played.loc[~hw, "away_id"]]).value_counts()
    l = pd.concat([played.loc[~hw, "home_id"], played.loc[hw, "away_id"]]).value_counts()
    return pd.DataFrame({"w_now": w, "l_now": l}).fillna(0).astype(int).rename_axis("team_id")


def build_outputs(sim, model, played, remaining, today: dt.date, alpha: float = DEFAULT_ALPHA):
    arr = sim.arrays
    ids = arr["team_ids"]
    resume, quad = compute_resume(arr, played, remaining, model.idx, model.hca)
    inel = sim.summary.set_index("team_id").loc[ids, "ineligible"].values.astype(bool)
    sel = select_and_seed(arr, resume, inel, alpha)

    seed = sel["seed"].astype(float)
    seed[seed == 0] = np.nan
    with np.errstate(all="ignore"):
        mean_seed = np.nanmean(np.where(np.isnan(seed).all(0), 0, seed), 0)
    extra = pd.DataFrame({
        "team_id": ids, "resume": resume.mean(0), "mean_score": sel["score"].mean(0),
        "p_field": sel["field"].mean(0), "p_at_large": sel["at_large"].mean(0),
        "mean_seed": mean_seed, "p_top4_seed": ((sel["seed"] >= 1) & (sel["seed"] <= 4)).mean(0),
        **{c: quad[:, i] for i, c in enumerate(QUAD_COLS)},
    })
    summary = sim.summary.merge(extra, on="team_id").merge(
        record_to_date(played), left_on="team_id", right_index=True, how="left")
    summary[["w_now", "l_now"]] = summary[["w_now", "l_now"]].fillna(0).astype(int)
    summary.loc[summary["ineligible"], ["p_field", "p_at_large", "p_top4_seed"]] = 0.0

    field = projected_bracket(summary)
    last_in, first_out = last_in_first_out(summary, field)

    r = model.ratings.merge(summary[["team_id", "team", "conf", "w_now", "l_now", "wins",
                                     "losses", "p_field", "mean_seed"]], on="team_id")
    top40 = r.sort_values("adj_em", ascending=False).head(40).reset_index(drop=True)
    top40.insert(0, "rank", np.arange(1, 41))

    meta = {"as_of": today.isoformat(), "sims": int(arr["em_final"].shape[0]), "alpha": alpha,
            "games_played": int(len(played)), "field_size": int(len(field))}
    return {"summary": summary, "field": field, "last_in": last_in, "first_out": first_out,
            "top40": top40, "meta": meta}


def write_outputs(o: dict):
    OUT_DIR.mkdir(exist_ok=True)
    o["top40"][["rank", "team", "conf", "w_now", "l_now", "adj_em", "em_sd", "adj_o", "adj_d",
                "adj_t", "wins", "losses", "p_field", "mean_seed"]].round(3).to_csv(
        OUT_DIR / "top40.csv", index=False)
    o["field"][["overall", "seed", "region", "team", "conf", "bid", "opening_round",
                "opening_game", "w_now", "l_now", "wins", "losses", "em_now", "resume",
                "mean_score", "p_field", "p_auto", *QUAD_COLS]].round(3).to_csv(
        OUT_DIR / "bracket.csv", index=False)
    o["summary"].round(4).to_csv(OUT_DIR / "all_teams.csv", index=False)
    (OUT_DIR / "meta.json").write_text(json.dumps(o["meta"], indent=2))
