"""Résumé score, computed inside every simulated season at Selection Sunday.

Quadrants use *our* end-of-season ranking (from the Bayesian model in that
simulation) with the NCAA's location-based cutoffs:

              Q1        Q2        Q3         Q4
    home    1-30     31-75     76-160     161+
    neutral 1-50     51-100    101-200    201+
    away    1-75     76-135    136-240    241+

The quadrant system is a step-function approximation of one question: how hard
was it for a bubble-quality team to win this game? The score uses that question
directly, so a win over the #31 team at home isn't treated as completely
different from a win over #30:

    credit = (won - P(bubble team wins this game))
           + MARGIN_WEIGHT * (capped margin - bubble team's capped expected margin) / 30

summed over the season. The first term is wins above bubble (WAB); the second
adds a capped margin component (+-15 points). Quadrant W-L records are
reported alongside for display.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

BUBBLE_RANK = 50          # ~ the last at-large spot in a 76-team field
MARGIN_CAP = 15.0
MARGIN_WEIGHT = 0.25
GAME_SD = 11.0            # single-game margin SD (points)
POSS = 68.5

# (Q1, Q2, Q3) upper rank bounds by location: home, neutral, away
QUAD_CUTS = {1: (30, 75, 160), 0: (50, 100, 200), -1: (75, 135, 240)}


def _side_credit(em, rank, bub, t, u, loc, margin, hca):
    """Credit and quadrant for team t vs u. Arrays shaped (S, G); loc/t/u shaped (G,)."""
    exp_b = (bub[:, None] - em[:, u] + 2 * hca * loc) * POSS / 100
    p_b = ndtr(exp_b / GAME_SD)
    won = margin > 0
    credit = (won - p_b) + MARGIN_WEIGHT * (
        np.clip(margin, -MARGIN_CAP, MARGIN_CAP) - np.clip(exp_b, -MARGIN_CAP, MARGIN_CAP)) / 30
    r = rank[:, u]
    q = np.full(r.shape, 4, dtype=np.int8)
    for l, (c1, c2, c3) in QUAD_CUTS.items():
        m = loc == l
        rr = r[:, m]
        q[:, m] = np.where(rr <= c1, 1, np.where(rr <= c2, 2, np.where(rr <= c3, 3, 4)))
    return credit, q, won


def compute_resume(arr: dict, played: pd.DataFrame, remaining: pd.DataFrame, idx: dict,
                   hca: float, chunk: int = 500):
    """Returns résumé score (S, K) and mean quadrant records (K, 8: Q1W, Q1L, ... Q4L)."""
    em_all = arr["em_final"]
    S, K = em_all.shape

    pl = played[played["home_id"].isin(idx) & played["away_id"].isin(idx)]
    p_h = pl["home_id"].map(idx).values.astype(int)
    p_a = pl["away_id"].map(idx).values.astype(int)
    p_loc = np.where(pl["neutral"].values, 0, 1)
    p_m = (pl["home_score"].astype(float) - pl["away_score"].astype(float)).values

    rem = remaining.set_index("game_id").loc[arr["rem_game_ids"]]
    r_h = rem["home_id"].map(idx).values.astype(int)
    r_a = rem["away_id"].map(idx).values.astype(int)
    r_loc = np.where(rem["neutral"].values, 0, 1)

    tgames = arr["tourney_games"]
    resume = np.zeros((S, K), dtype=np.float32)
    quad = np.zeros((K, 8))

    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        em = em_all[s0:s1].astype(np.float64)
        n = s1 - s0
        rank = np.empty_like(em, dtype=np.int32)
        order = np.argsort(-em, axis=1)
        rank[np.arange(n)[:, None], order] = np.arange(1, K + 1)
        bub = np.sort(em, axis=1)[:, -BUBBLE_RANK]
        rows = np.arange(n)[:, None]

        def add(t, u, loc, margin):
            credit, q, won = _side_credit(em, rank, bub, t, u, loc, margin, hca)
            np.add.at(resume[s0:s1], (np.broadcast_to(rows, credit.shape),
                                      np.broadcast_to(t, credit.shape)), credit)
            for qi in range(4):
                mq = q == qi + 1
                np.add.at(quad[:, 2 * qi], np.broadcast_to(t, mq.shape)[mq & won], 1)
                np.add.at(quad[:, 2 * qi + 1], np.broadcast_to(t, mq.shape)[mq & ~won], 1)

        # played games: same result in every simulation
        pm = np.broadcast_to(p_m, (n, len(p_m)))
        add(p_h, p_a, p_loc, pm)
        add(p_a, p_h, -p_loc, -pm)
        # simulated regular season
        rm = arr["rem_home_margin"][s0:s1].astype(np.float64)
        add(r_h, r_a, r_loc, rm)
        add(r_a, r_h, -r_loc, -rm)
        # simulated conference tournaments (ragged: handle per row)
        tg = tgames[(tgames[:, 0] >= s0) & (tgames[:, 0] < s1)]
        if len(tg):
            sims = tg[:, 0] - s0
            for (t, u, loc, m) in ((tg[:, 1], tg[:, 2], tg[:, 3], tg[:, 4]),
                                   (tg[:, 2], tg[:, 1], -tg[:, 3], -tg[:, 4])):
                exp_b = (bub[sims] - em[sims, u] + 2 * hca * loc) * POSS / 100
                p_b = ndtr(exp_b / GAME_SD)
                won = m > 0
                credit = (won - p_b) + MARGIN_WEIGHT * (
                    np.clip(m, -MARGIN_CAP, MARGIN_CAP) - np.clip(exp_b, -MARGIN_CAP, MARGIN_CAP)) / 30
                np.add.at(resume[s0:s1], (sims, t), credit.astype(np.float32))
                r = rank[sims, u]
                cuts = np.array([QUAD_CUTS[int(x)] for x in loc])
                q = 1 + (r > cuts[:, 0]).astype(int) + (r > cuts[:, 1]) + (r > cuts[:, 2])
                for qi in range(4):
                    mq = q == qi + 1
                    np.add.at(quad[:, 2 * qi], t[mq & won], 1)
                    np.add.at(quad[:, 2 * qi + 1], t[mq & ~won], 1)
    return resume, quad / S
