"""Player-season table from ESPN player box scores.

One row per (season, athlete, team) for D1 teams. ESPN athlete_id is stable
across schools, which lets us follow transfers.

Production metric: Hollinger Game Score, summed over the season. It is crude,
but it is box-score only and available for every player in every season.
`min_share` is the fraction of the team's available minutes the player played
(0..1; 1.0 = played every minute of every game).
"""
from __future__ import annotations

import pandas as pd

from cbb.config import PROCESSED_DIR, RAW_DIR, SEASONS

STATS = {
    "minutes": "min", "points": "pts", "field_goals_made": "fgm",
    "field_goals_attempted": "fga", "free_throws_made": "ftm",
    "free_throws_attempted": "fta", "offensive_rebounds": "orb",
    "defensive_rebounds": "drb", "assists": "ast", "steals": "stl",
    "blocks": "blk", "turnovers": "tov", "fouls": "pf",
}


def game_score(d: pd.DataFrame) -> pd.Series:
    return (d["pts"] + 0.4 * d["fgm"] - 0.7 * d["fga"] - 0.4 * (d["fta"] - d["ftm"])
            + 0.7 * d["orb"] + 0.3 * d["drb"] + d["stl"] + 0.7 * d["ast"]
            + 0.7 * d["blk"] - 0.4 * d["pf"] - d["tov"])


def build_player_seasons(seasons=SEASONS) -> pd.DataFrame:
    teams = pd.read_parquet(PROCESSED_DIR / "teams.parquet")
    out = []
    for s in seasons:
        d1 = set(teams.loc[teams["season"] == s, "team_id"])
        p = pd.read_parquet(RAW_DIR / f"player_box_{s}.parquet",
                            columns=["game_id", "athlete_id", "athlete_display_name",
                                     "team_id", *STATS])
        p = p.rename(columns=STATS).dropna(subset=["athlete_id", "min"])
        p = p[p["min"] > 0]
        p["team_id"] = pd.to_numeric(p["team_id"], errors="coerce")
        p = p[p["team_id"].isin(d1)]
        p["athlete_id"] = p["athlete_id"].astype(int)
        p["team_id"] = p["team_id"].astype(int)
        for c in STATS.values():
            p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0)

        agg = (p.groupby(["athlete_id", "team_id"])
                .agg(name=("athlete_display_name", "first"), games=("game_id", "nunique"),
                     **{c: (c, "sum") for c in STATS.values()})
                .reset_index())
        team_min = p.groupby("team_id")["min"].sum().rename("team_min")
        agg = agg.join(team_min, on="team_id")
        agg["min_share"] = (agg["min"] / (agg["team_min"] / 5)).clip(upper=1.0)
        agg["gmsc"] = game_score(agg)
        agg["gmsc40"] = 40 * agg["gmsc"] / agg["min"].clip(lower=1)
        agg.insert(0, "season", s)
        out.append(agg)
    ps = pd.concat(out, ignore_index=True)
    ps.to_parquet(PROCESSED_DIR / "player_seasons.parquet", index=False)
    return ps


if __name__ == "__main__":
    ps = build_player_seasons()
    print(ps.groupby("season").size())
