"""End-to-end daily run: download -> clean -> preseason prior -> ratings.

Usage:  python scripts/run_pipeline.py [--skip-download]
"""
import argparse

import pandas as pd

from cbb.config import CURRENT_SEASON, PROCESSED_DIR
from cbb.data.clean import build_all
from cbb.data.download import download_all
from cbb.data.players import build_player_seasons
from cbb.models.prior import build_features, fit_prior, season_final_ratings
from cbb.models.ridge import fit_ridge, rank_table

PRIOR_ALPHA = 8.0  # backtest-optimal; matches sigma_game^2 / sigma_prior^2


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
    print(f"Fitting ratings on {cur['game_id'].nunique()} {CURRENT_SEASON} games...")
    model = fit_ridge(cur, alpha=PRIOR_ALPHA, prior=prior)
    table = rank_table(model, out["teams"][out["teams"]["season"] == CURRENT_SEASON])
    table = table.merge(prior[["team_id", "prior_em"]], on="team_id", how="left")
    path = PROCESSED_DIR / f"ridge_ratings_{CURRENT_SEASON}.csv"
    table.to_csv(path, index=False)
    print(f"  wrote {path}")
    print(table.head(10).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
