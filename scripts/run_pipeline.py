"""End-to-end daily run: download -> clean -> baseline ratings.

Usage:  python scripts/run_pipeline.py [--skip-download]
"""
import argparse

import pandas as pd

from cbb.config import CURRENT_SEASON, PROCESSED_DIR
from cbb.data.clean import build_all
from cbb.data.download import download_all
from cbb.models.ridge import fit_ridge, rank_table


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
    cur = tg[tg["season"] == CURRENT_SEASON]
    n_games = cur["game_id"].nunique()
    print(f"Fitting baseline ratings on {n_games} {CURRENT_SEASON} games...")
    if n_games < 1000:
        print("  note: early season; ridge ratings are noisy until the Bayesian model "
              "adds preseason priors.")
    model = fit_ridge(cur)
    table = rank_table(model, out["teams"][out["teams"]["season"] == CURRENT_SEASON])
    path = PROCESSED_DIR / f"ridge_ratings_{CURRENT_SEASON}.csv"
    table.to_csv(path, index=False)
    print(f"  wrote {path}")
    print(table.head(10).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
