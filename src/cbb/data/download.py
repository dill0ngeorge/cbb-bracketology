"""Download raw season files (schedules + team box scores) from sportsdataverse."""
from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

from cbb.config import CURRENT_SEASON, RAW_DIR, SCHEDULE_URL, SEASONS, TEAM_BOX_URL


def _fetch(url: str, dest: Path, retries: int = 3) -> None:
    tmp = dest.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                f.write(r.read())
            tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001
            if attempt == retries:
                raise RuntimeError(f"failed to download {url}: {e}") from e
            time.sleep(2 * attempt)


def download_season(season: int, force: bool = False) -> None:
    """Fetch one season. Past seasons are cached; the current season is always refreshed."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    refresh = force or season == CURRENT_SEASON
    for kind, template in (("schedule", SCHEDULE_URL), ("team_box", TEAM_BOX_URL)):
        dest = RAW_DIR / f"{kind}_{season}.parquet"
        if dest.exists() and not refresh:
            continue
        _fetch(template.format(season=season), dest)


def download_all(seasons=SEASONS, force: bool = False) -> None:
    for s in seasons:
        download_season(s, force=force)
        print(f"  downloaded {s}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="re-download cached seasons")
    p.add_argument("--season", type=int, nargs="*", default=SEASONS)
    a = p.parse_args()
    download_all(a.season, force=a.force)
