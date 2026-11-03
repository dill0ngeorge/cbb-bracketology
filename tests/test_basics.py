import numpy as np
import pandas as pd

from cbb.data.clean import classify_game
from cbb.models.ridge import fit_ridge


def test_classify_game():
    base = {"season_type": 2, "conference_competition": False, "notes_headline": ""}
    assert classify_game(base) == "nonconf"
    assert classify_game({**base, "conference_competition": True}) == "conf_regular"
    assert classify_game({**base, "conference_competition": True,
                          "notes_headline": "SEC Tournament - Final"}) == "conf_tourney"
    assert classify_game({**base, "season_type": 3,
                          "notes_headline": "MEN'S BASKETBALL CHAMPIONSHIP - FINAL FOUR"}) == "ncaa"
    assert classify_game({**base, "season_type": 3, "notes_headline": "NIT"}) == "post_other"


def _synthetic_games(n_teams=30, n_games=900, seed=0):
    """Simulate games from known ratings; ridge should recover their ordering."""
    rng = np.random.default_rng(seed)
    o, d = rng.normal(0, 6, n_teams), rng.normal(0, 6, n_teams)
    rows = []
    for g in range(n_games):
        a, b = rng.choice(n_teams, 2, replace=False)
        poss = 68.0
        pa = 105 + o[a] + d[b] + 2.5 + rng.normal(0, 8)
        pb = 105 + o[b] + d[a] - 2.5 + rng.normal(0, 8)
        for t, u, pt, pu, loc in ((a, b, pa, pb, 1), (b, a, pb, pa, -1)):
            rows.append(dict(game_id=g, team_id=t, opp_id=u, loc=loc, has_box=True,
                             poss=poss, off_eff=pt, def_eff=pu,
                             margin=(pt - pu) * poss / 100))
    return pd.DataFrame(rows), o - d


def test_ridge_recovers_ratings():
    tg, true_em = _synthetic_games()
    m = fit_ridge(tg, hca=2.5)
    est = m.ratings.sort_values("team_id")["adj_em"].values
    assert np.corrcoef(est, true_em)[0, 1] > 0.95


def test_prior_pulls_ratings():
    """With a strong prior and little data, ratings stay near the prior."""
    tg, _ = _synthetic_games(n_games=20)
    teams = np.unique(tg["team_id"])
    prior = pd.DataFrame({"team_id": teams, "prior_o_rel": 10.0, "prior_d_rel": 0.0})
    prior.loc[0, "prior_o_rel"] = 20.0
    m = fit_ridge(tg, alpha=1e6, hca=2.5, prior=prior)
    r = m.ratings.set_index("team_id")
    assert abs((r.loc[teams[0], "adj_o"] - r.loc[teams[1], "adj_o"]) - 10.0) < 0.5
