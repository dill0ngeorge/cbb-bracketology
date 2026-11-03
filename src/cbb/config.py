"""Project-wide settings: seasons, paths, data URLs."""
from pathlib import Path

# Season is labeled by the year it ends in (2026-27 season == 2027).
FIRST_SEASON = 2015
CURRENT_SEASON = 2027
SEASONS = list(range(FIRST_SEASON, CURRENT_SEASON + 1))
NO_TOURNAMENT_SEASONS = {2020}  # COVID

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# sportsdataverse republishes ESPN men's college basketball data on GitHub.
TEAM_BOX_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "espn_mens_college_basketball_team_boxscores/team_box_{season}.parquet"
)
SCHEDULE_URL = (
    "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/"
    "mbb/schedules/parquet/mbb_schedule_{season}.parquet"
)

# Possession estimate: FGA - OREB + TO + FT_FACTOR * FTA
FT_FACTOR = 0.475
