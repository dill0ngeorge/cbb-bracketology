"""Turn raw ESPN files into tidy tables.

Outputs (data/processed/):
  teams.parquet     one row per (season, team): D1 teams only, with conference
  games.parquet     one row per completed D1-vs-D1 game (home/away perspective)
  team_games.parquet  two rows per game (one per team) with possessions & efficiency
  schedule_remaining.parquet  unplayed D1-vs-D1 games for the current season
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cbb.config import CURRENT_SEASON, FT_FACTOR, PROCESSED_DIR, RAW_DIR, SEASONS

BOX_COLS = [
    "field_goals_made", "field_goals_attempted",
    "three_point_field_goals_made", "three_point_field_goals_attempted",
    "free_throws_made", "free_throws_attempted",
    "offensive_rebounds", "defensive_rebounds", "total_turnovers",
    "assists", "steals", "blocks", "fouls",
]


def load_raw(kind: str, season: int) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / f"{kind}_{season}.parquet")


# --------------------------------------------------------------------- teams
def build_teams(sched: pd.DataFrame, season: int) -> pd.DataFrame:
    """D1 teams = teams ESPN assigns to a D1 conference (incl. independents, id 43)."""
    parts = []
    for side in ("home", "away"):
        parts.append(
            sched[[f"{side}_id", f"{side}_conference_id", f"{side}_display_name",
                   f"{side}_location", f"{side}_abbreviation"]]
            .set_axis(["team_id", "conf_id", "team_name", "team_short", "abbr"], axis=1)
        )
    t = pd.concat(parts).dropna(subset=["conf_id"])
    t["team_id"] = t["team_id"].astype(int)
    t["conf_id"] = t["conf_id"].astype(int)
    # Membership is fixed within a season; take the mode to be safe.
    teams = (t.groupby("team_id")
               .agg(conf_id=("conf_id", lambda x: x.mode().iat[0]),
                    team_name=("team_name", "first"),
                    team_short=("team_short", "first"),
                    abbr=("abbr", "first"))
               .reset_index())
    conf_names = (sched.loc[sched["groups_is_conference"].fillna(False).astype(bool),
                            ["groups_id", "groups_short_name"]]
                  .dropna().drop_duplicates("groups_id"))
    conf_names = dict(zip(conf_names["groups_id"].astype(int), conf_names["groups_short_name"]))
    conf_names.setdefault(43, "Ind")
    teams["conf"] = teams["conf_id"].map(conf_names)
    teams.insert(0, "season", season)
    return teams


# --------------------------------------------------------------------- games
def classify_game(row) -> str:
    notes = row["notes_headline"] or ""
    if row["season_type"] == 3:
        return "ncaa" if "MEN'S BASKETBALL CHAMPIONSHIP" in notes.upper() else "post_other"
    if row["conference_competition"] and notes.strip():
        return "conf_tourney"
    if row["conference_competition"]:
        return "conf_regular"
    return "nonconf"


def build_games(sched: pd.DataFrame, teams: pd.DataFrame, season: int):
    d1 = set(teams["team_id"])
    g = sched.copy()
    g["home_id"] = pd.to_numeric(g["home_id"], errors="coerce")
    g["away_id"] = pd.to_numeric(g["away_id"], errors="coerce")
    g = g[g["home_id"].isin(d1) & g["away_id"].isin(d1)].copy()
    g["home_id"] = g["home_id"].astype(int)
    g["away_id"] = g["away_id"].astype(int)
    g["game_type"] = g.apply(classify_game, axis=1)
    g["date"] = pd.to_datetime(g["date"], utc=True).dt.tz_convert("America/New_York").dt.date
    g["neutral"] = g["neutral_site"].fillna(False).astype(bool)
    keep = ["game_id", "date", "home_id", "away_id", "neutral", "game_type",
            "conference_competition", "notes_headline", "home_score", "away_score",
            "status_type_completed"]
    g = g[keep].rename(columns={"conference_competition": "conf_game",
                                "notes_headline": "notes",
                                "status_type_completed": "completed"})
    g["completed"] = g["completed"].fillna(False).astype(bool)
    g["conf_game"] = g["conf_game"].fillna(False).astype(bool)
    g.insert(0, "season", season)
    g = g.drop_duplicates("game_id")
    played = g[g["completed"]].copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    # Drop the occasional forfeit / bad record.
    played = played[(played["home_score"] > 0) & (played["away_score"] > 0)]
    remaining = g[~g["completed"]].drop(columns=["home_score", "away_score"])
    return played, remaining


# ---------------------------------------------------------------- team-games
def build_team_games(box: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game with box stats, possessions and per-100 efficiency."""
    b = box[["game_id", "team_id", "team_home_away", "team_score", *BOX_COLS]].copy()
    b["team_id"] = pd.to_numeric(b["team_id"], errors="coerce")
    b = b.dropna(subset=["team_id"]).drop_duplicates(["game_id", "team_id"])
    b["team_id"] = b["team_id"].astype(int)

    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        r = games[["season", "game_id", "date", "neutral", "game_type", "conf_game"]].copy()
        r["team_id"] = games[f"{side}_id"]
        r["opp_id"] = games[f"{opp}_id"]
        r["pts"] = games[f"{side}_score"]
        r["opp_pts"] = games[f"{opp}_score"]
        r["loc"] = np.where(games["neutral"], 0, 1 if side == "home" else -1)
        rows.append(r)
    tg = pd.concat(rows, ignore_index=True)

    tg = tg.merge(b.drop(columns=["team_home_away", "team_score"]),
                  on=["game_id", "team_id"], how="left")
    opp_b = b[["game_id", "team_id", *BOX_COLS]].rename(
        columns={"team_id": "opp_id", **{c: f"opp_{c}" for c in BOX_COLS}})
    tg = tg.merge(opp_b, on=["game_id", "opp_id"], how="left")

    def poss(prefix=""):
        return (tg[f"{prefix}field_goals_attempted"] - tg[f"{prefix}offensive_rebounds"]
                + tg[f"{prefix}total_turnovers"] + FT_FACTOR * tg[f"{prefix}free_throws_attempted"])

    # Average the two teams' estimates; they should be ~equal.
    tg["poss"] = (poss() + poss("opp_")) / 2
    tg["has_box"] = tg["poss"].notna() & (tg["poss"] > 40) & (tg["poss"] < 120)
    tg.loc[~tg["has_box"], "poss"] = np.nan
    tg["off_eff"] = 100 * tg["pts"] / tg["poss"]
    tg["def_eff"] = 100 * tg["opp_pts"] / tg["poss"]
    tg["margin"] = tg["pts"] - tg["opp_pts"]
    tg["win"] = (tg["margin"] > 0).astype(int)
    return tg.sort_values(["season", "date", "game_id", "loc"], ascending=[True, True, True, False])


# ------------------------------------------------------------------- driver
def build_all(seasons=SEASONS) -> dict[str, pd.DataFrame]:
    teams_l, games_l, tg_l, rem_l = [], [], [], []
    for s in seasons:
        sched = load_raw("schedule", s)
        box = load_raw("team_box", s)
        teams = build_teams(sched, s)
        games, remaining = build_games(sched, teams, s)
        tg = build_team_games(box, games)
        teams_l.append(teams); games_l.append(games); tg_l.append(tg)
        if s == CURRENT_SEASON:
            rem_l.append(remaining)
        print(f"  {s}: {len(teams)} D1 teams, {len(games)} games played, "
              f"{tg['has_box'].mean():.1%} with box scores, {len(remaining)} unplayed")
    out = {
        "teams": pd.concat(teams_l, ignore_index=True),
        "games": pd.concat(games_l, ignore_index=True),
        "team_games": pd.concat(tg_l, ignore_index=True),
        "schedule_remaining": (pd.concat(rem_l, ignore_index=True) if rem_l
                               else pd.DataFrame()),
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
    return out


if __name__ == "__main__":
    build_all()
