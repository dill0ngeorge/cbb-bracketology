"""End-to-end daily run: download -> clean -> preseason prior -> Bayesian ratings.

Usage:  python scripts/run_pipeline.py [--skip-download]
"""
import argparse

import pandas as pd

from cbb.config import CURRENT_SEASON, PROCESSED_DIR
from cbb.data.clean import build_all
from cbb.data.download import download_all
from cbb.data.players import build_player_seasons
from cbb.models.prior import build_features, fit_prior, season_final_ratings
from cbb.models.kalman import KalmanRatings
from cbb.models.ridge import rank_table



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-download", action="store_true")
    a = p.parse_args()

    if not a.skip_download:
        print("Downloading...")
        download_all()
    print("Cleaning...")
    out = build_all()

    tg = out["team_games"]
    ps = build_player_seasons()
    print("Fitting preseason prior...")
    final = season_final_ratings(tg, seasons=range(2015, CURRENT_SEASON), force=True)
    feats = build_features(final, ps, out["teams"], range(2016, CURRENT_SEASON + 1))
    prior = fit_prior(feats[feats["season"] < CURRENT_SEASON]).predict(
        feats[feats["season"] == CURRENT_SEASON])

    cur = tg[tg["season"] == CURRENT_SEASON]
    print(f"Running Bayesian filter over {cur['game_id'].nunique()} {CURRENT_SEASON} games...")
    prev = final[final["season"] == CURRENT_SEASON - 1].set_index("team_id")
    teams = out["teams"][out["teams"]["season"] == CURRENT_SEASON]
    mu0 = tg[(tg["season"] == CURRENT_SEASON - 1) & tg["has_box"]]["off_eff"].mean()
    model = (KalmanRatings(teams["team_id"].values)
             .init(prior, mu0=mu0, prev_tempo=prev["adj_t"] - prev["adj_t"].mean(),
                   mu_t0=prev["adj_t"].mean())
             .run(cur))
    table = rank_table(model, teams)
    table = table.merge(prior[["team_id", "prior_em"]], on="team_id", how="left")
    path = PROCESSED_DIR / f"ratings_{CURRENT_SEASON}.csv"
    table.to_csv(path, index=False)
    print(f"  wrote {path}")
    cols = ["rank", "team_short", "conf", "adj_o", "adj_d", "adj_em", "em_sd", "prior_em"]
    print(table[cols].head(15).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
