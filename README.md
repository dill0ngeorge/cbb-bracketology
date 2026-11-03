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
| Ridge adjusted efficiency | 11.47 | 70.9% | 0.560 | 0.190 |

Error is highest in December (RMSE 12.4) when the model has little data, which is
what the Bayesian model's preseason priors are meant to fix.

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
  models/ridge.py    baseline adjusted-efficiency ratings
  evaluate.py        walk-forward evaluation, shared by every model
scripts/run_pipeline.py
tests/
```
