"""Render outputs/index.html: projected bracket, bubble, and top 40.

Static HTML (everything readable at load), built from outputs/*.csv so the
daily pipeline can regenerate the page.
"""
from __future__ import annotations

import datetime as dt
import html
import json

import pandas as pd

from cbb.outputs import OUT_DIR

REGIONS = ["East", "West", "South", "Midwest"]

CSS = """
:root{
  /* Layout: masthead, four region columns (stack on phones), bubble strip, top-40 table. */
  --paper:#f4f6f8; --card:#ffffff; --ink:#122033; --muted:#5b6b7e; --line:#d9e0e8;
  --accent:#c2410c; --aq:#0f6e66; --aq-bg:#e1f3f0; --or:#8a5a00; --or-bg:#fbefd6;
  --display:"Barlow Condensed","Arial Narrow",sans-serif;
  --body:"IBM Plex Sans",system-ui,sans-serif;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --paper:#0e151e; --card:#16202c; --ink:#e6edf5; --muted:#93a3b5; --line:#2a3646;
  --accent:#f97b42; --aq:#5fd0c3; --aq-bg:#123532; --or:#f2c46d; --or-bg:#3a2c0e; color-scheme:dark; } }
:root[data-theme="dark"]{
  --paper:#0e151e; --card:#16202c; --ink:#e6edf5; --muted:#93a3b5; --line:#2a3646;
  --accent:#f97b42; --aq:#5fd0c3; --aq-bg:#123532; --or:#f2c46d; --or-bg:#3a2c0e; color-scheme:dark; }
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 var(--body)}
main{max-width:1180px;margin:0 auto;padding:28px 20px 48px;display:grid;gap:36px}
h1,h2{font-family:var(--display);font-weight:600;letter-spacing:.01em;margin:0;text-wrap:balance}
h1{font-size:44px;line-height:1}
h2{font-size:28px;line-height:1.1}
.sub{color:var(--muted);margin:6px 0 0;max-width:68ch}
.facts{display:flex;flex-wrap:wrap;gap:8px 20px;margin-top:14px;font-size:13px;color:var(--muted)}
.facts b{color:var(--ink);font-weight:600}
.regions{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}
@media (max-width:980px){.regions{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:560px){.regions{grid-template-columns:1fr}}
.region{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 12px 8px}
.region h3{font-family:var(--display);font-size:22px;font-weight:600;margin:0 0 6px;color:var(--accent)}
.slot{display:grid;grid-template-columns:24px minmax(0,1fr);gap:6px;padding:5px 0;border-top:1px solid var(--line);font-size:13.5px}
.slot:first-of-type{border-top:0}
.seed{font-family:var(--display);font-size:17px;font-weight:600;color:var(--muted);text-align:right;line-height:1.2}
.team{display:flex;align-items:baseline;gap:6px;min-width:0}
.team .n{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.team .c{color:var(--muted);font-size:11.5px;white-space:nowrap}
.chip{font-size:10.5px;font-weight:600;padding:0 5px;border-radius:4px;line-height:16px;white-space:nowrap}
.aq{color:var(--aq);background:var(--aq-bg)}
.or{color:var(--or);background:var(--or-bg)}
.pair .team + .team{margin-top:1px}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--muted);margin-top:10px;align-items:center}
.bubble{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}
.box{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.box h3{font-size:13px;font-weight:600;margin:0 0 6px;text-transform:none;color:var(--muted)}
.box ol{margin:0;padding-left:20px;font-size:14px}
.box li{padding:2px 0}
.box li span{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
.tablewrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th{font-size:12px;font-weight:600;color:var(--muted);background:var(--card);position:sticky;top:0}
td.l,th.l{text-align:left}
tr:last-child td{border-bottom:0}
.pm{color:var(--muted);font-size:11.5px}
.bar{display:inline-block;height:6px;border-radius:3px;background:var(--accent);vertical-align:middle;margin-left:6px;opacity:.85}
.notes{font-size:13.5px;color:var(--muted);max-width:80ch;display:grid;gap:6px}
.notes p{margin:0}
.notes a{color:var(--accent)}
"""


def _e(x) -> str:
    return html.escape(str(x))


def _team(r, show_chip=True) -> str:
    chip = '<span class="chip aq" title="Automatic bid">AQ</span>' if (show_chip and r["bid"] == "auto") else ""
    return (f'<div class="team"><span class="n">{_e(r["team"])}</span>'
            f'<span class="c">{_e(r["conf"])}</span>{chip}</div>')


def _region_html(field: pd.DataFrame, region: str) -> str:
    rows = []
    reg = field[field["region"] == region].sort_values(["seed", "overall"])
    for seed, grp in reg.groupby("seed", sort=True):
        games = grp.groupby(grp["opening_game"].where(grp["opening_round"], -1 - grp["overall"]))
        for _, g in games:
            if len(g) == 2:
                body = (f'<div class="pair">{_team(g.iloc[0])}{_team(g.iloc[1])}'
                        f'<span class="chip or">Opening Round</span></div>')
            else:
                body = _team(g.iloc[0])
            rows.append(f'<div class="slot"><div class="seed">{seed}</div>{body}</div>')
    return f'<section class="region"><h3>{region}</h3>{"".join(rows)}</section>'


def render(out_dir=OUT_DIR) -> str:
    field = pd.read_csv(out_dir / "bracket.csv")
    top = pd.read_csv(out_dir / "top40.csv")
    allt = pd.read_csv(out_dir / "all_teams.csv")
    meta = json.loads((out_dir / "meta.json").read_text())
    as_of = dt.date.fromisoformat(meta["as_of"]).strftime("%B %-d, %Y")

    at_large = field[field["bid"] == "at-large"].sort_values("mean_score")
    last_in = at_large.head(6)
    out = allt[~allt["ineligible"] & ~allt["team"].isin(field["team"])].sort_values(
        "mean_score", ascending=False).head(6)
    opening = field[field["opening_round"]].sort_values("overall")

    def li(df):
        return "".join(f'<li>{_e(r.team)} <span>{_e(r.conf)} · {r.p_field:.0%} in field</span></li>'
                       for r in df.itertuples())

    og = []
    for gid, g in opening.groupby("opening_game", sort=False):
        a, b = g.iloc[0], g.iloc[1]
        og.append(f'<li>({a.seed}) {_e(a.team)} vs. {_e(b.team)} <span>{a.region}</span></li>')

    trs = []
    for r in top.itertuples():
        seed = "" if pd.isna(r.mean_seed) else f"{r.mean_seed:.1f}"
        trs.append(
            f'<tr><td>{r.rank}</td><td class="l">{_e(r.team)}</td><td class="l">{_e(r.conf)}</td>'
            f'<td>{r.w_now}-{r.l_now}</td><td>{r.adj_em:+.1f} <span class="pm">±{r.em_sd:.1f}</span></td>'
            f'<td>{r.adj_o:.1f}</td><td>{r.adj_d:.1f}</td><td>{r.adj_t:.1f}</td>'
            f'<td>{r.wins:.0f}-{r.losses:.0f}</td><td>{seed}</td>'
            f'<td>{r.p_field:.0%}<span class="bar" style="width:{round(r.p_field * 50)}px"></span></td></tr>')

    regions = "".join(_region_html(field, r) for r in REGIONS)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>2027 Bracketology</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>{CSS}</style></head>
<body><main>
<header>
  <h1>2027 Bracketology</h1>
  <p class="sub">Projected field for Selection Sunday, March 14, 2027, from {meta['sims']:,} simulations of the rest of the season. Seeds blend team strength and résumé equally.</p>
  <div class="facts"><span>Updated <b>{as_of}</b></span><span><b>{meta['games_played']:,}</b> D1 games played</span>
  <span>Field <b>{meta['field_size']}</b> (32 automatic, 44 at-large)</span><span>Blend <b>α = {meta['alpha']}</b></span></div>
</header>

<section>
  <h2>Projected bracket</h2>
  <div class="legend"><span><span class="chip aq">AQ</span> projected conference champion</span>
  <span><span class="chip or">Opening Round</span> plays in on Tuesday or Wednesday</span>
  <span>Regions follow the S-curve; bracketing rules not yet applied.</span></div>
  <div class="regions" style="margin-top:12px">{regions}</div>
</section>

<section class="bubble">
  <div class="box"><h3>Last six in</h3><ol>{li(last_in)}</ol></div>
  <div class="box"><h3>First six out</h3><ol>{li(out)}</ol></div>
  <div class="box"><h3>Opening Round games</h3><ol>{"".join(og)}</ol></div>
</section>

<section>
  <h2>Top 40</h2>
  <p class="sub">Ranked by current team strength: adjusted efficiency margin per 100 possessions, with its posterior standard deviation.</p>
  <div class="tablewrap" style="margin-top:12px"><table>
  <thead><tr><th>#</th><th class="l">Team</th><th class="l">Conf.</th><th>Record</th><th>Rating</th><th>Off.</th><th>Def.</th><th>Tempo</th><th>Proj. record</th><th>Avg. seed</th><th>In field</th></tr></thead>
  <tbody>{"".join(trs)}</tbody></table></div>
</section>

<section class="notes">
  <h2 style="color:var(--ink)">How this works</h2>
  <p>Team ratings come from a dynamic Bayesian model of points per possession, starting from a preseason projection built on returning production and transfers, and updated after every game.</p>
  <p>Each simulation plays out the remaining schedule and every conference tournament in its 2027 format. Conference champions take the automatic bids. The other 44 spots go to the best seeding scores, an even blend of projected team strength and a résumé score (wins above a bubble team against the same schedule, quadrant-weighted by location, plus capped margin).</p>
  <p>The displayed bracket gives each conference's auto bid to its most likely champion and orders the field by average seeding score. Early in the season these projections lean heavily on the preseason prior.</p>
  <p>Updated every morning. Data: <a href="bracket.csv">bracket.csv</a>, <a href="top40.csv">top40.csv</a>, <a href="all_teams.csv">all_teams.csv</a>. Code and methodology: <a href="https://github.com/dill0ngeorge/cbb-bracketology">github.com/dill0ngeorge/cbb-bracketology</a>.</p>
</section>
</main></body></html>"""


def write_report(out_dir=OUT_DIR):
    path = out_dir / "index.html"
    path.write_text(render(out_dir))
    return path


if __name__ == "__main__":
    print(write_report())
