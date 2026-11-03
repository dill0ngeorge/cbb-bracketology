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
    m = fit_ridge(tg)
    est = m.ratings.sort_values("team_id")["adj_em"].values
    assert np.corrcoef(est, true_em)[0, 1] > 0.95
    assert 1.5 < m.hca < 3.5
