"""End-to-end daily run: download -> clean -> preseason prior -> Bayesian ratings
-> season + conference tournament simulation.

Usage:  python scripts/run_pipeline.py [--skip-download] [--sims 5000]
"""
import argparse
import datetime as dt

import numpy as np

from cbb.config import CURRENT_SEASON, PROCESSED_DIR
from cbb.data.clean import build_all
from cbb.data.download import download_all
from cbb.data.players import build_player_seasons
from cbb.models.current import build_model, load_tables
from cbb.models.prior import build_features, season_final_ratings
from cbb.models.ridge import rank_table
from cbb.sim.formats import load_formats
from cbb.sim.season import simulate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--sims", type=int, default=5000)
    a = p.parse_args()

    if not a.skip_download:
        print("Downloading...")
        download_all()
    print("Cleaning...")
    out = build_all()
    ps = build_player_seasons()

    print("Fitting preseason prior features...")
    final = season_final_ratings(out["team_games"], seasons=range(2015, CURRENT_SEASON), force=True)
    feats = build_features(final, ps, out["teams"], range(2016, CURRENT_SEASON + 1))
    feats.to_parquet(PROCESSED_DIR / "prior_features.parquet", index=False)

    tables = load_tables()
    today = dt.date.today()
    model, prior, teams = build_model(CURRENT_SEASON, through=today, tables=tables)
    table = rank_table(model, teams).merge(prior[["team_id", "prior_em"]], on="team_id", how="left")
    table.to_csv(PROCESSED_DIR / f"ratings_{CURRENT_SEASON}.csv", index=False)
    cols = ["rank", "team_short", "conf", "adj_o", "adj_d", "adj_em", "em_sd", "prior_em"]
    print(table[cols].head(15).round(1).to_string(index=False))

    print(f"\nSimulating rest of season ({a.sims} sims)...")
    g = tables["games"]
    played = g[(g["season"] == CURRENT_SEASON) & (g["date"] < today)]
    rem = tables["schedule_remaining"]
    rem = rem[rem["date"] >= today]
    sim = simulate(model, teams, played, rem, load_formats(CURRENT_SEASON), today, n_sims=a.sims)
    sim.summary.to_csv(PROCESSED_DIR / f"sim_summary_{CURRENT_SEASON}.csv", index=False)
    np.savez_compressed(PROCESSED_DIR / f"sim_{CURRENT_SEASON}.npz", **sim.arrays)
    cols = ["team", "conf", "wins", "losses", "p_conf_1seed", "p_auto", "p_field_pred"]
    print(sim.summary[cols].head(20).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
