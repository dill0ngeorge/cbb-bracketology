"""Field selection and seeding for the 76-team NCAA tournament.

Seeding score, per simulation, standardized across eligible D1 teams:
    score = alpha * z(predictive rating) + (1 - alpha) * z(résumé)

Selection: every conference champion (auto bid) + the 44 best-scoring
non-champions. The 76 are ordered 1..76 (the S-curve).

76-team seed lines (NCAA format from 2027): the 12 lowest-ranked at-large teams
and the 12 lowest-ranked automatic qualifiers play in the Opening Round, paired
with the team next to them on the S-curve. That leaves 52 direct teams + 12
Opening Round slots = 64 bracket slots; slots are filled in S-curve order, four
per seed line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_FIELD = 76
N_AT_LARGE = 44
N_OPENING_EACH = 12      # 12 at-large + 12 auto-bid teams play in the Opening Round
DEFAULT_ALPHA = 0.5


def blend_scores(em: np.ndarray, resume: np.ndarray, eligible: np.ndarray, alpha: float):
    """z-score each component across eligible teams within each simulation."""
    def z(x):
        e = x[:, eligible]
        return (x - e.mean(1, keepdims=True)) / e.std(1, keepdims=True)
    return alpha * z(em) + (1 - alpha) * z(resume)


def seed_lines(scurve_auto: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    scurve_auto: bool array (76,), S-curve order, True = automatic qualifier.
    Returns seed line per S-curve position (76,) and Opening Round game id (-1 if none).
    """
    n = len(scurve_auto)
    pos = np.arange(n)
    opening = np.full(n, -1)
    slot_pos = []        # (position used to order the slot, members)
    game = 0
    for is_auto in (False, True):
        group = pos[scurve_auto == is_auto][-N_OPENING_EACH:]
        for i in range(0, len(group), 2):
            pair = group[i:i + 2]
            opening[pair] = game
            game += 1
    direct = pos[opening < 0]
    slots = [(p, [p]) for p in direct]
    for g in range(game):
        members = pos[opening == g]
        slots.append((members.mean(), list(members)))
    slots.sort(key=lambda s: s[0])
    seed = np.zeros(n, dtype=int)
    for k, (_, members) in enumerate(slots):
        seed[members] = k // 4 + 1
    return seed, opening


def select_and_seed(arr: dict, resume: np.ndarray, ineligible: np.ndarray,
                    alpha: float = DEFAULT_ALPHA):
    """Per-simulation field and seeds. Returns dict of (S, K) arrays."""
    em, auto = arr["em_final"], arr["auto"]
    S, K = em.shape
    eligible = ~ineligible
    score = blend_scores(em.astype(np.float64), resume.astype(np.float64), eligible, alpha)
    pool = np.where(auto | ineligible[None, :], -np.inf, score)
    cut = np.sort(pool, axis=1)[:, -N_AT_LARGE][:, None]
    at_large = pool >= cut
    field = auto | at_large
    seed = np.zeros((S, K), dtype=np.int8)
    for s in range(S):
        teams = np.where(field[s])[0]
        order = teams[np.argsort(-score[s, teams])]
        lines, _ = seed_lines(auto[s, order])
        seed[s, order] = lines
    return {"score": score.astype(np.float32), "at_large": at_large, "field": field, "seed": seed}


def projected_bracket(summary: pd.DataFrame, alpha_label: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """
    One displayed bracket from the simulations: each conference's most likely champion
    takes its auto bid; the 44 at-large spots go to the best mean seeding scores among
    the rest; the S-curve orders all 76 by mean seeding score.
    summary needs: team, conf, p_auto, mean_score, ineligible.
    """
    s = summary[~summary["ineligible"]].copy()
    autos = s.loc[s.groupby("conf")["p_auto"].idxmax()].assign(bid="auto")
    rest = s[~s.index.isin(autos.index)].sort_values("mean_score", ascending=False)
    at_large = rest.head(N_AT_LARGE).assign(bid="at-large")
    field = pd.concat([autos, at_large]).sort_values("mean_score", ascending=False)
    field = field.reset_index(drop=True)
    lines, opening = seed_lines((field["bid"] == "auto").values)
    field.insert(0, "overall", np.arange(1, len(field) + 1))
    field.insert(1, "seed", lines)
    field["opening_round"] = opening >= 0
    field["opening_game"] = opening
    # Regions: snake the S-curve by seed line (no bracketing-rule checks yet).
    region_names = ["East", "West", "South", "Midwest"]
    region = []
    for line, grp in field.groupby("seed", sort=True):
        slots = []
        for _, r in grp.iterrows():
            key = ("g", r["opening_game"]) if r["opening_round"] else ("t", r["overall"])
            if key not in slots:
                slots.append(key)
        order = region_names if line % 2 else region_names[::-1]
        slot_region = {k: order[i % 4] for i, k in enumerate(slots)}
        for _, r in grp.iterrows():
            key = ("g", r["opening_game"]) if r["opening_round"] else ("t", r["overall"])
            region.append((r["overall"], slot_region[key]))
    field["region"] = field["overall"].map(dict(region))
    return field


def last_in_first_out(summary: pd.DataFrame, field: pd.DataFrame, n: int = 6):
    """Bubble view: lowest-scoring at-large teams in, best-scoring teams out."""
    at_large = field[field["bid"] == "at-large"].sort_values("mean_score")
    out = summary[~summary["ineligible"] & ~summary["team"].isin(field["team"])]
    return at_large.head(n), out.sort_values("mean_score", ascending=False).head(n)
