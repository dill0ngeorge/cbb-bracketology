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
- [ ] Bayesian rating model (PyMC)
- [ ] Season + conference tournament simulator
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

Known gaps: freshmen/recruiting classes are invisible to the prior; one prior SD
for every team (extreme projections may be overconfident).

## Usage

```bash
pip install -e ".[dev]"
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
  evaluate.py        walk-forward evaluation, shared by every model
scripts/run_pipeline.py
tests/
```
