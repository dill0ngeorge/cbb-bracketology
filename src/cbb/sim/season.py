"""Monte Carlo simulation of the rest of the season and every conference tournament.

Each simulation:
  1. draws team ratings from the Bayesian posterior (with full covariance),
  2. lets them keep drifting week by week (the model's random walk, tau),
  3. plays every remaining regular-season game (margin = ratings + game noise),
  4. builds conference standings and seeds (random tiebreaks),
  5. plays each conference tournament in its real format -> automatic bids.

Outputs per team: expected record, P(regular-season title), seed distribution,
P(auto bid), and a provisional P(in field) under the purely predictive view
(alpha = 1: best 44 non-champions by rating). The résumé-based selection
replaces that last step later; per-simulation game results are saved for it.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cbb.sim.formats import FormatBook

N_AT_LARGE = 44  # 76-team field from 2027: 32 automatic bids + 44 at-large


@dataclass
class SimOutput:
    summary: pd.DataFrame         # one row per team
    arrays: dict                  # per-simulation detail for downstream steps


def _game_margin(R, tempo, mu_t, h, a, loc, hca, noise, rng, K):
    """R: (S, 2K) ratings; h/a: team indices (arrays, broadcast over S)."""
    o, d = R[..., :K], R[..., K:]
    diff100 = (np.take_along_axis(o, h, -1) + np.take_along_axis(d, a, -1)
               - np.take_along_axis(o, a, -1) - np.take_along_axis(d, h, -1)
               + 2 * hca * loc)
    poss = mu_t + tempo[h] + tempo[a]
    return diff100 * poss / 100 + noise * rng.standard_normal(diff100.shape)


def simulate(model, teams: pd.DataFrame, played: pd.DataFrame, remaining: pd.DataFrame,
             book: FormatBook, today: dt.date, n_sims: int = 5000, chunk: int = 500,
             seed: int = 0, n_at_large: int = N_AT_LARGE) -> SimOutput:
    """
    model:     KalmanRatings fitted through `today`
    teams:     D1 teams this season (team_id, team_short, conf)
    played:    completed D1 games this season (games table rows, before today)
    remaining: regular-season D1 games still to play (date, home_id, away_id, neutral, conf_game)
    """
    rng = np.random.default_rng(seed)
    K = len(model.teams)
    idx = model.idx
    hyp = model.hyper
    tempo, mu_t = model.mt[:K], model.mt[-1]
    teams = teams.set_index("team_id").reindex(model.teams).reset_index()
    names = teams["team_short"].values
    conf = teams["conf"].values
    ineligible = np.isin(names, list(book.ncaa_ineligible))

    # ---- fixed inputs: records from games already played
    pl = played[played["home_id"].isin(idx) & played["away_id"].isin(idx)]
    ph = pl["home_id"].map(idx).values.astype(int)
    pa = pl["away_id"].map(idx).values.astype(int)
    hw = (pl["home_score"].astype(float) > pl["away_score"].astype(float)).values
    base_w = np.bincount(np.where(hw, ph, pa), minlength=K).astype(float)
    base_l = np.bincount(np.where(hw, pa, ph), minlength=K).astype(float)
    c = pl["conf_game"].values.astype(bool)
    base_cw = np.bincount(np.where(hw, ph, pa)[c], minlength=K).astype(float)
    base_cl = np.bincount(np.where(hw, pa, ph)[c], minlength=K).astype(float)

    rem = remaining[remaining["home_id"].isin(idx) & remaining["away_id"].isin(idx)]
    rem = rem.sort_values("date")
    gh, ga = rem["home_id"].map(idx).values, rem["away_id"].map(idx).values
    gloc = np.where(rem["neutral"].values, 0.0, 1.0)
    gconf = rem["conf_game"].values
    days = (pd.to_datetime(rem["date"]) - pd.Timestamp(today)).dt.days.clip(lower=0).values
    gweek = days // 7
    W = int(gweek.max()) + 2 if len(rem) else 1   # +1 week for conference tournaments

    conf_members = {cf: np.where((conf == cf) & ~ineligible)[0] for cf in pd.unique(conf)}

    keep = {k: [] for k in ["wins", "losses", "conf_wins", "conf_losses", "seed",
                            "auto", "em_final", "rem_home_won"]}
    tourney_rows = []

    for start in range(0, n_sims, chunk):
        S = min(chunk, n_sims - start)
        x0 = model.sample(S, rng)[:, :2 * K].astype(np.float32)
        steps = (hyp["tau"] * rng.standard_normal((S, W, 2 * K))).astype(np.float32)
        steps[:, 0] = 0
        path = x0[:, None, :] + np.cumsum(steps, axis=1)          # (S, W, 2K)

        # ---- remaining regular season
        home_won = np.zeros((S, len(rem)), dtype=bool)
        for w in np.unique(gweek):
            m = gweek == w
            R = path[:, w, :]
            h = np.broadcast_to(gh[m], (S, m.sum()))
            a = np.broadcast_to(ga[m], (S, m.sum()))
            home_won[:, m] = _game_margin(R, tempo, mu_t, h, a, gloc[m], model.hca,
                                          hyp["margin_noise"], rng, K) > 0
        win_t = np.where(home_won, gh, ga)
        lose_t = np.where(home_won, ga, gh)
        rows = np.arange(S)[:, None]
        wins = np.tile(base_w, (S, 1)); losses = np.tile(base_l, (S, 1))
        np.add.at(wins, (rows, win_t), 1); np.add.at(losses, (rows, lose_t), 1)
        cw = np.tile(base_cw, (S, 1)); cl = np.tile(base_cl, (S, 1))
        np.add.at(cw, (rows, win_t[:, gconf]), 1); np.add.at(cl, (rows, lose_t[:, gconf]), 1)

        # ---- standings, seeds, conference tournaments
        pct = cw / np.maximum(cw + cl, 1) + 1e-4 * rng.random((S, K))   # random tiebreak
        seed_arr = np.zeros((S, K), dtype=np.int16)
        auto = np.zeros((S, K), dtype=bool)
        R_end = path[:, -1, :]
        for cf, mem in conf_members.items():
            if len(mem) < 2:
                continue
            order = mem[np.argsort(-pct[:, mem], axis=1)]                # (S, n) team idx by seed
            seed_arr[rows, order] = np.arange(1, len(mem) + 1)
            fmt = book.get(cf, len(mem))
            champ = _play_tournament(fmt, order, R_end, tempo, mu_t, model.hca,
                                     hyp["margin_noise"], rng, K, start, tourney_rows)
            auto[np.arange(S), champ] = True

        keep["wins"].append(wins.astype(np.int16)); keep["losses"].append(losses.astype(np.int16))
        keep["conf_wins"].append(cw.astype(np.int16)); keep["conf_losses"].append(cl.astype(np.int16))
        keep["seed"].append(seed_arr); keep["auto"].append(auto)
        keep["em_final"].append((R_end[:, :K] - R_end[:, K:]).astype(np.float32))
        keep["rem_home_won"].append(home_won)

    arr = {k: np.concatenate(v) for k, v in keep.items()}
    arr["tourney_games"] = (np.concatenate(tourney_rows) if tourney_rows
                            else np.zeros((0, 5), dtype=np.int32))
    arr["team_ids"] = model.teams
    arr["rem_game_ids"] = rem["game_id"].values

    # ---- provisional field, predictive view (alpha = 1)
    score = np.where(arr["auto"] | ineligible[None, :], -np.inf, arr["em_final"])
    cut = np.sort(score, axis=1)[:, -n_at_large][:, None]
    at_large = score >= cut
    arr["at_large_pred"] = at_large

    seeds = arr["seed"].astype(float)
    seeds[seeds == 0] = np.nan
    with np.errstate(all="ignore"):
        mean_seed = np.nanmean(np.where(np.isnan(seeds).all(0), 0, seeds), 0)
    summary = pd.DataFrame({
        "team_id": model.teams, "team": names, "conf": conf,
        "wins": arr["wins"].mean(0), "losses": arr["losses"].mean(0),
        "conf_wins": arr["conf_wins"].mean(0), "conf_losses": arr["conf_losses"].mean(0),
        "p_conf_1seed": (arr["seed"] == 1).mean(0),
        "mean_conf_seed": mean_seed,
        "p_auto": arr["auto"].mean(0),
        "p_at_large_pred": at_large.mean(0),
        "p_field_pred": (arr["auto"] | at_large).mean(0),
        "em_now": model.ratings["adj_em"].values,
        "em_sd": model.ratings["em_sd"].values,
    })
    summary.loc[ineligible, ["p_auto", "p_at_large_pred", "p_field_pred"]] = 0.0
    summary["ineligible"] = ineligible
    return SimOutput(summary.sort_values("p_field_pred", ascending=False).reset_index(drop=True), arr)


def _play_tournament(fmt, order, R, tempo, mu_t, hca, noise, rng, K, sim_offset, log):
    """order: (S, n) team index by seed (column 0 = 1 seed). Returns champion team idx (S,)."""
    S = order.shape[0]
    rows = np.arange(S)[:, None]
    alive = np.zeros((S, 0), dtype=np.int16)   # seed numbers still alive
    rounds = [list(r) for r in fmt.entry]
    rnd = 0
    while True:
        entrants = rounds[rnd] if rnd < len(rounds) else []
        pool = np.concatenate([alive, np.broadcast_to(np.array(entrants, dtype=np.int16),
                                                      (S, len(entrants)))], axis=1)
        if pool.shape[1] == 1 and rnd >= len(rounds):
            break
        pool = np.sort(pool, axis=1)                              # best seed first
        n = pool.shape[1]
        hi, lo = pool[:, : n // 2], pool[:, ::-1][:, : n // 2]    # 1 v n, 2 v n-1, ...
        th = order[rows, hi - 1]
        tl = order[rows, lo - 1]
        loc = 1.0 if rnd < fmt.campus_rounds else 0.0
        hi_won = _game_margin(R, tempo, mu_t, th, tl, loc, hca, noise, rng, K) > 0
        alive = np.where(hi_won, hi, lo)
        log.append(np.stack([
            np.broadcast_to(np.arange(S)[:, None] + sim_offset, th.shape).ravel(),
            th.ravel(), tl.ravel(), np.full(th.size, int(loc)), hi_won.ravel()], 1).astype(np.int32))
        rnd += 1
        if alive.shape[1] == 1 and rnd >= len(rounds):
            break
    return order[np.arange(S), alive[:, 0] - 1]
