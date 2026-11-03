"""Walk-forward backtest of the Bayesian (Kalman) model vs ridge baselines.

Refit weekly from Nov 15; preseason priors are trained only on earlier seasons.
Usage: python scripts/backtest.py
"""
import pandas as pd

from cbb.config import PROCESSED_DIR as D
from cbb.evaluate import score, walk_forward
from cbb.models.kalman import KalmanRatings
from cbb.models.prior import fit_prior
from cbb.models.ridge import fit_ridge

SEASONS = list(range(2018, 2027))


def main():
    tg = pd.read_parquet(D / "team_games.parquet")
    f = pd.read_parquet(D / "prior_features.parquet")
    teams = pd.read_parquet(D / "teams.parquet")
    final = pd.read_parquet(D / "final_ratings.parquet")
    setup = {}
    for s in SEASONS:
        pr = fit_prior(f[f.season < s]).predict(f[f.season == s])
        prev = final[final.season == s - 1].set_index("team_id")
        mu0 = tg[(tg.season == s - 1) & tg.has_box].off_eff.mean()
        setup[s] = (pr, prev.adj_t - prev.adj_t.mean(), mu0, prev.adj_t.mean())

    def kalman(train):
        s = int(train.season.iloc[0]); pr, pt, mu0, mt0 = setup[s]
        return (KalmanRatings(teams[teams.season == s].team_id.values)
                .init(pr, mu0=mu0, prev_tempo=pt, mu_t0=mt0).run(train))

    models = {
        "ridge (no prior)": lambda t: fit_ridge(t, alpha=1.0),
        "ridge + prior": lambda t: fit_ridge(t, alpha=8.0, prior=setup[int(t.season.iloc[0])][0]),
        "bayesian (kalman)": kalman,
    }
    months = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]
    overall, monthly = {}, {}
    for name, fn in models.items():
        p = walk_forward(tg, fn, seasons=SEASONS, start=(11, 15))
        p["month"] = pd.Categorical(pd.to_datetime(p.date).dt.month_name().str[:3], months)
        p.to_parquet(D / f"wf_{name.split()[0]}.parquet")
        overall[name] = score(p).iloc[0]
        monthly[name] = score(p, "month")[["margin_rmse", "log_loss"]]
    print(pd.DataFrame(overall).T.round(3).to_string())
    print(pd.concat(monthly, axis=1).round(3).to_string())
    p["bin"] = pd.cut(p.p_win, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    print("\nCalibration (bayesian):")
    print(p.groupby("bin", observed=True).agg(pred=("p_win", "mean"), actual=("win", "mean"),
                                               n=("win", "size")).round(3).to_string())


if __name__ == "__main__":
    main()
