"""Fit the full PyMC dynamic model on past seasons to learn sigma, tau and k.

Usage: python scripts/fit_hyperparams.py 2023 2025 2026 [--draws 300]

Each season takes ~30 s with nutpie on 2 CPUs. Copy the averaged posterior
means into kalman.DEFAULT_HYPER.
"""
import argparse
import json
import time

import arviz as az
import pandas as pd

from cbb.config import PROCESSED_DIR as D
from cbb.models import bayes_pymc
from cbb.models.prior import fit_prior

HYPER = ["mu", "sigma", "tau", "k"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", type=int, nargs="+")
    ap.add_argument("--draws", type=int, default=300)
    a = ap.parse_args()
    tg = pd.read_parquet(D / "team_games.parquet")
    f = pd.read_parquet(D / "prior_features.parquet")
    results = {}
    for s in a.seasons:
        prior = fit_prior(f[f.season < s]).predict(f[f.season == s])
        prior["prior_sd_o"], prior["prior_sd_d"] = 4.3, 4.1  # past-only CV RMSE
        t = time.time()
        idata, _ = bayes_pymc.fit(tg[tg.season == s], prior, draws=a.draws, tune=a.draws)
        summ = az.summary(idata, var_names=HYPER)
        print(f"season {s} ({time.time() - t:.0f}s)\n{summ.to_string()}\n")
        results[s] = {v: float(summ.loc[v, "mean"]) for v in HYPER}
    (D / "pymc_hyperparams.json").write_text(json.dumps(results, indent=2))
    print(pd.DataFrame(results).T.mean().round(3))


if __name__ == "__main__":
    main()
