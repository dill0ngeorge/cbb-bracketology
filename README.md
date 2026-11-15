# CBB Bracketology

A projected-Selection-Sunday NCAA tournament bracket built on **my own model of
team strength**, not a prediction of the committee.

## How it works (planned)

1. **Team ratings**: a dynamic hierarchical Bayesian model of offensive/defensive
   efficiency per possession, tempo, and home court, with preseason priors.
2. **Season simulation**: Monte Carlo over the remaining regular season and every
   conference tournament → P(auto bid), P(at-large), P(in field), seed distribution.
3. **Seeding score**: `α · predictive + (1 − α) · résumé`, where the résumé uses
   quadrant logic (NCAA location-based cutoffs) applied to **my** rankings, plus
   capped margin. α is a slider on the public site.
4. **Field**: auto bids (simulated conference champions) → best remaining at-large
   teams → bracketing rules.
5. **Secondary view**: a committee-behavior model, shown as "model vs. committee."

## Status

- [x] Data pipeline (2014-15 → current season, D1 only)
- [x] Ridge-regression baseline ratings + walk-forward evaluation
- [x] Preseason prior from roster continuity + transfers (player-level data)
- [x] Bayesian dynamic rating model (PyMC for hyperparameters, Kalman filter for exact daily posterior)
- [x] Season + conference tournament simulator (76-team field: 32 auto bids + 44 at-large)
- [ ] Résumé / quadrant scoring
- [ ] Bracket construction + bracketing rules
- [ ] Public site

## Data

ESPN men's college basketball schedules and team box scores, via
[sportsdataverse](https://github.com/sportsdataverse) (`hoopR-mbb-data`,
`sportsdataverse-data`). D1 teams are those ESPN assigns to a D1 conference;
games involving non-D1 opponents are dropped. Possessions are estimated as
`FGA − OREB + TO + 0.475·FTA`, averaged over both teams.

## Baseline results

Walk-forward (refit weekly from Dec 1, predict the next week), 2015–2026, 53k games:

| Model | Margin RMSE | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| Ridge adjusted efficiency (fixed home court) | 11.36 | 71.0% | 0.557 | 0.188 |

## Preseason prior

A linear model predicts each team's full-season offensive/defensive rating from
last season's ratings, conference strength, returning minutes and production
(Hollinger game score), and incoming transfers' production and level of
competition (ESPN athlete IDs persist across schools). Trained 2016–2026,
evaluated leave-one-season-out:

| Preseason projection | RMSE (efficiency margin) |
|---|---|
| Everyone average | 12.6 |
| Last season's rating | 7.6 |
| Last season, regressed 30% | 7.5 |
| Roster model (linear) | **6.2** |
| Roster model (gradient boosting) | 6.25 |

Ratings are then shrunk toward the prior instead of toward average (MAP estimate).
The backtest-optimal prior strength (α ≈ 8) matches the Bayesian value
σ²_game / σ²_prior ≈ (12 / 4.3)².

Walk-forward 2018–2026, refit weekly from Nov 15 (trained only on earlier seasons):

| Month | RMSE, no prior | RMSE, with prior | Log loss, no prior | Log loss, with prior |
|---|---|---|---|---|
| Nov | 13.32 | **11.85** | 0.639 | **0.495** |
| Dec | 12.18 | **11.56** | 0.512 | **0.476** |
| Jan | 11.41 | **11.14** | 0.584 | **0.568** |
| Feb | 11.12 | **11.02** | 0.567 | **0.563** |
| Mar | 11.14 | **11.11** | 0.567 | **0.566** |

## Bayesian dynamic model

Each team's offense and defense follow a weekly random walk starting from the
preseason prior; each team-game's efficiency is
`mu + o[team] + d[opp] + HCA·loc + Normal(0, σ)`.

- **PyMC (nutpie NUTS)** fits the full model on past seasons (~30 s/season) to
  learn the hyperparameters. Posterior means are stable across seasons:

  | Season | σ (game noise /100 poss) | τ (weekly drift) | k (prior scale) |
  |---|---|---|---|
  | 2023 | 12.58 | 0.68 | 0.76 |
  | 2025 | 12.58 | 0.73 | 0.79 |
  | 2026 | 12.79 | 0.62 | 0.88 |

  τ ≈ 0.7/week means true team quality drifts ±3 points of efficiency margin
  over a season; k < 1 means the roster prior is *more* reliable than its raw
  error suggests once in-season drift is accounted for.
- **Kalman filter** computes the exact filtering posterior of the same
  linear-Gaussian model (a full season in ~2 s), including the full covariance
  between teams, which the season simulator samples from.

Walk-forward 2018–2026 (45.5k games):

| Model | Margin RMSE | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| Ridge, no prior | 11.73 | 70.8% | 0.571 | 0.191 |
| Ridge + prior (α tuned) | 11.28 | 71.9% | 0.539 | 0.183 |
| **Bayesian (Kalman)** | **11.28** | **71.9%** | **0.538** | **0.182** |

The Bayesian model matches the tuned ridge on point accuracy with no tuning,
and adds calibrated uncertainty: predicted vs. actual win rates agree within
~1 point in every decile. That uncertainty is what the bracket simulation needs.

## Season simulator

Monte Carlo over the rest of the season (5,000 runs in ~10 s). Each run draws
ratings from the Bayesian posterior (full covariance), lets them keep drifting
weekly, plays every remaining game, builds conference standings (random
tiebreaks), and plays each conference tournament in its **actual 2027 format**
(`config/conf_tournaments_2027.yaml`, including stepladders and campus sites).
Outputs P(conference 1 seed), P(auto bid), and, for now, P(in field) under the
purely predictive view (α = 1); per-simulation game results are saved for the
résumé-based selection.

The 2027 NCAA tournament has **76 teams**: 32 automatic bids + 44 at-large; the
12 lowest-seeded at-large teams and 12 lowest-seeded automatic qualifiers play
in the Opening Round.

Backtest from Jan 15, 2023–2026 (past seasons use generic tournament formats):

| Quantity | Model | Naive baseline |
|---|---|---|
| Conference wins RMSE | 1.98 | 3.75 |
| P(conference 1 seed) log loss | 0.181 | 0.294 |
| P(auto bid) log loss | 0.215 | 0.300 |

Probabilities are calibrated across deciles; strong favorites' auto-bid odds run
slightly low in the backtest because generic formats ignore protective byes.

Known gaps: freshmen/recruiting classes are invisible to the prior; one prior SD
for every team (extreme projections may be overconfident).

## Usage

```bash
pip install -e ".[dev,bayes]"
python scripts/run_pipeline.py      # download, clean, baseline ratings
pytest
```

## Layout

```
src/cbb/
  config.py          seasons, paths, URLs
  data/download.py   fetch raw parquet files
  data/clean.py      teams / games / team_games / remaining schedule
  data/players.py    player-season table (minutes, game score, transfers)
  models/ridge.py    adjusted-efficiency ratings, optionally shrunk to a prior
  models/prior.py    preseason prior model
  models/bayes_pymc.py  full Bayesian model in PyMC (hyperparameter learning)
  models/kalman.py   exact posterior via Kalman filter (daily ratings, simulation draws)
  models/current.py  build the season model as of any date (pipeline, sim, backtests)
  sim/formats.py     conference tournament formats (validated)
  sim/season.py      season + conference tournament Monte Carlo
  evaluate.py        walk-forward evaluation, shared by every model
config/conf_tournaments_2027.yaml   tournament formats (edit me)
scripts/run_pipeline.py     daily: download, clean, prior, Bayesian ratings, simulation
scripts/fit_hyperparams.py  PyMC fits for sigma / tau / k
scripts/backtest.py         walk-forward comparison of all models
scripts/backtest_sim.py     simulator backtest (standings, 1 seeds, auto bids)
tests/
```
