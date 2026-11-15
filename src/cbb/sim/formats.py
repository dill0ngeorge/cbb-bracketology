"""Load and validate conference tournament formats (config/conf_tournaments_*.yaml)."""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from cbb.config import ROOT


@dataclass
class TourneyFormat:
    conf: str
    entry: list[list[int]]          # seeds entering each round
    campus_rounds: int = 0          # rounds hosted by the higher seed (99 = all)
    confidence: str = "assumed"
    note: str = ""

    @property
    def n_qualifiers(self) -> int:
        return sum(len(r) for r in self.entry)

    def validate(self):
        seeds = sorted(s for r in self.entry for s in r)
        if seeds != list(range(1, len(seeds) + 1)):
            raise ValueError(f"{self.conf}: seeds must be 1..N exactly once, got {seeds}")
        alive = 0
        for i, r in enumerate(self.entry):
            pool = alive + len(r)
            if pool % 2:
                raise ValueError(f"{self.conf}: round {i + 1} has {pool} teams (odd)")
            alive = pool // 2
        while alive > 1:
            if alive % 2:
                raise ValueError(f"{self.conf}: later round has {alive} teams (odd)")
            alive //= 2


def default_format(conf: str, n: int) -> TourneyFormat:
    """All teams, standard bracket, byes to the top seeds, neutral site."""
    size = 1
    while size < n:
        size *= 2
    byes = size - n
    entry = [list(range(byes + 1, n + 1)), list(range(1, byes + 1))] if byes else [list(range(1, n + 1))]
    entry = [r for r in entry if r]
    return TourneyFormat(conf, entry, confidence="default")


@dataclass
class FormatBook:
    formats: dict[str, TourneyFormat]
    ncaa_ineligible: set[str] = field(default_factory=set)

    def get(self, conf: str, n_teams: int) -> TourneyFormat:
        f = self.formats.get(conf)
        if f is None or f.n_qualifiers > n_teams:
            return default_format(conf, n_teams)
        return f


def load_formats(season: int) -> FormatBook:
    path = ROOT / "config" / f"conf_tournaments_{season}.yaml"
    raw = yaml.safe_load(path.read_text())
    out = {}
    for conf, spec in raw["conferences"].items():
        site = spec.get("site", "neutral")
        if site == "neutral":
            campus = 0
        elif site == "campus":
            campus = 99
        else:
            campus = int(site["campus_rounds"])
        f = TourneyFormat(conf, spec["entry"], campus, spec.get("confidence", ""),
                          spec.get("note", ""))
        f.validate()
        out[conf] = f
    return FormatBook(out, set(raw.get("ncaa_ineligible", [])))
