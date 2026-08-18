"""Render a job's results as a self-contained HTML leaderboard.

Design notes worth keeping, because they are constraints rather than taste:

- **Rank ranges, not ranks.** Models whose intervals overlap are shown as tied.
  Printing 1, 2, 3 for statistically indistinguishable rows invents an ordering
  the data does not support.
- **The interval is drawn, not just printed.** A whisker under each bar makes
  "these two do not separate" visible at a glance, which a number in a column
  does not.
- **One hue.** Success rate is a single measure across agents -- magnitude, not
  identity -- so it gets one validated series colour. The task grid is a
  sequential ramp of that same hue.
- **Colour never carries meaning alone.** Every heatmap cell prints its n/N (the
  sub-3:1 light steps of a sequential ramp require that relief, and it also
  distinguishes a genuine zero from no-data). The harness-fault callout pairs
  its status colour with an icon and a label.
- **Baselines are marked by form, not hue.** Oracle and null are reference rows,
  not competitors; a status colour would imply good/bad.

No external requests: the page must render from a file:// URL with no network.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from harness.eval.results import FailureMode
from harness.eval.stats import rank_interval

# --- validated palette (see the data-viz reference instance) ----------------
# series hue passes every check in both modes; the sequential ramp is monotone,
# single-hue, with visible per-cell labels as the relief for its light steps.
_RAMP_LIGHT = ["#e8f1fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"]
_RAMP_DARK = ["#12233a", "#173556", "#1c4a7c", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]

# A cell label sits ON its fill, so it must contrast with the fill rather than
# with the page. Each ramp step therefore carries its own ink: dark ink on the
# pale steps, light ink on the saturated ones. Without this the brightest cells
# in either mode render near-invisible text.
# Which ink each step takes is measured, not assumed: white-on-#3987e5 is only
# 3.6:1 while black on the same fill is 5.4:1, so the mid-ramp step keeps dark
# ink even though it looks "dark enough" for white. tests/test_report.py
# recomputes every pair and fails below 4.5:1.
_RAMP_INK_LIGHT = ["#0b0b0b", "#0b0b0b", "#0b0b0b", "#0b0b0b", "#0b0b0b", "#ffffff", "#ffffff"]
_RAMP_INK_DARK = ["#ffffff", "#ffffff", "#ffffff", "#ffffff", "#0b0b0b", "#0b0b0b", "#0b0b0b"]

#: Rows that are references rather than competitors. "scripted" is a solvability
#: probe that is told what to do, so it belongs here too -- ranking it against
#: models would compare a model with something handed the answer.
_BASELINE_IDS = ("oracle", "scripted", "scripted_pick_place", "null")

#: Anything that establishes a task is solvable at all, in preference order.
_SOLVABILITY_IDS = ("oracle", "scripted", "scripted_pick_place")


def _e(v) -> str:
    return html.escape("" if v is None else str(v))


def _pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{100 * float(v):.1f}%"


def _num(v, nd: int = 3) -> str:
    return "-" if v is None else f"{float(v):.{nd}f}"


def _ramp_index(frac: Optional[float], n: int) -> int:
    if frac is None:
        return -1
    return min(n - 1, max(0, int(round(frac * (n - 1)))))


def _bar_svg(rate: float, lo: float, hi: float, *, width: int = 190, muted: bool = False) -> str:
    """A magnitude bar with its 95% interval drawn beneath it.

    Bar: 4px rounded data-end, anchored at the baseline. Whisker: 2px line with
    end caps. Both in one hue; the interval is what makes overlap legible.
    """
    h_bar, h_gap, h_whisk = 9, 5, 7
    height = h_bar + h_gap + h_whisk
    pad = 1.5  # keep the interval end caps inside the viewBox
    x = lambda f: pad + f * (width - 2 * pad)  # noqa: E731
    fill = "var(--viz-series-muted)" if muted else "var(--viz-series)"
    bw = 0.0 if rate <= 0 else max(2.0, rate * width)
    cy = h_bar + h_gap + h_whisk / 2
    return f"""<svg class="bar" viewBox="0 0 {width} {height}" width="{width}" height="{height}"
 role="img" aria-label="success {100 * rate:.1f} percent, 95% interval {100 * lo:.1f} to {100 * hi:.1f}">
  <rect x="0" y="0" width="{width}" height="{h_bar}" rx="2" fill="var(--viz-track)"/>
  <rect x="0" y="0" width="{bw:.1f}" height="{h_bar}" rx="2" fill="{fill}"/>
  <line x1="{x(lo):.1f}" y1="{cy}" x2="{x(hi):.1f}" y2="{cy}"
        stroke="var(--viz-whisker)" stroke-width="2"/>
  <line x1="{x(lo):.1f}" y1="{cy - 3}" x2="{x(lo):.1f}" y2="{cy + 3}"
        stroke="var(--viz-whisker)" stroke-width="2"/>
  <line x1="{x(hi):.1f}" y1="{cy - 3}" x2="{x(hi):.1f}" y2="{cy + 3}"
        stroke="var(--viz-whisker)" stroke-width="2"/>
</svg>"""


_CSS = """
:root {
  --bg:#fcfcfb; --panel:#ffffff; --panel-2:#f4f4f1;
  --line:#e3e3de; --line-strong:#c8c8c1;
  --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#78776f;
  --viz-series:#2a78d6; --viz-series-muted:#9ba8b8;
  --viz-track:#ececea; --viz-whisker:#78776f;
  --warn:#fab219; --warn-ink:#7a5300; --warn-bg:#fdf3dd;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:#1a1a19; --panel:#212120; --panel-2:#282827;
    --line:#33332f; --line-strong:#4a4a45;
    --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#8f8e85;
    --viz-series:#3987e5; --viz-series-muted:#5d6a78;
    --viz-track:#2e2e2c; --viz-whisker:#8f8e85;
    --warn:#fab219; --warn-ink:#fab219; --warn-bg:#332a12;
  }
}
:root[data-theme="dark"] {
  --bg:#1a1a19; --panel:#212120; --panel-2:#282827;
  --line:#33332f; --line-strong:#4a4a45;
  --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#8f8e85;
  --viz-series:#3987e5; --viz-series-muted:#5d6a78;
  --viz-track:#2e2e2c; --viz-whisker:#8f8e85;
  --warn:#fab219; --warn-ink:#fab219; --warn-bg:#332a12;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
     font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:74rem;margin:0 auto;padding:2.5rem 1.5rem 5rem}
h1{font-family:var(--mono);font-size:1.7rem;font-weight:600;letter-spacing:-.02em;margin:0 0 .3rem}
h2{font-family:var(--mono);font-size:1.02rem;font-weight:600;margin:2.6rem 0 .3rem;
   padding-top:1.4rem;border-top:1px solid var(--line)}
.sub{color:var(--ink-2);margin:0 0 .2rem}
.meta{font-family:var(--mono);font-size:.74rem;color:var(--ink-3);letter-spacing:.03em}
.note{color:var(--ink-2);font-size:.88rem;margin:.4rem 0 1rem;max-width:52rem}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(9.5rem,1fr));gap:1px;
       background:var(--line);border:1px solid var(--line);border-radius:4px;
       overflow:hidden;margin:1.6rem 0}
.tile{background:var(--panel);padding:.85rem 1rem}
.tile dt{font-family:var(--mono);font-size:.63rem;letter-spacing:.1em;text-transform:uppercase;
         color:var(--ink-3);margin:0 0 .3rem}
.tile dd{margin:0;font-family:var(--mono);font-size:1.5rem;letter-spacing:-.02em;
         font-variant-numeric:tabular-nums}
.tile dd small{font-size:.72rem;color:var(--ink-3)}

.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:4px;background:var(--panel)}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:.83rem}
th,td{padding:.55rem .8rem;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
thead th{font-family:var(--mono);font-size:.65rem;letter-spacing:.07em;text-transform:uppercase;
         color:var(--ink-3);font-weight:500;background:var(--panel-2);position:sticky;top:0}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--panel-2)}
td.n,th.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
.agent{font-family:var(--mono);font-weight:600}
.rank{font-family:var(--mono);color:var(--ink-2)}
.bar{display:block}
.ci{font-family:var(--mono);font-size:.72rem;color:var(--ink-3);white-space:nowrap}
/* A reference row is recessive on the page surface -- but a heatmap label sits
   on its own fill, so muting it there fights the fill instead of the page and
   the cell goes unreadable. The `reference` chip already marks the row; ink
   stays keyed to whatever is behind it. */
tr.ref td:not(.cell){color:var(--ink-2)}
.chip{display:inline-block;font-family:var(--mono);font-size:.63rem;letter-spacing:.04em;
      padding:.1em .45em;border-radius:3px;border:1px solid var(--line-strong);color:var(--ink-2)}
.chip.warn{background:var(--warn-bg);color:var(--warn-ink);border-color:var(--warn)}

.grid-tbl th.ch{text-align:center}
.grid-tbl td.cell{text-align:center;font-family:var(--mono);font-size:.74rem;
                  font-variant-numeric:tabular-nums;position:relative;min-width:4.6rem}
.grid-tbl td.cell span{position:relative;z-index:1}
.grid-tbl td.cell .fillbg{position:absolute;inset:1px;border-radius:2px}
.grid-tbl td.nodata{color:var(--ink-3)}
.legend{display:flex;align-items:center;gap:.5rem;margin:.7rem 0 0;
        font-family:var(--mono);font-size:.68rem;color:var(--ink-3)}
.legend .steps{display:flex;gap:2px}
.legend i{width:1.15rem;height:.55rem;border-radius:1px;display:block}

.callout{border:1px solid var(--line-strong);border-left:3px solid var(--warn);
         background:var(--panel);border-radius:4px;padding:.9rem 1.1rem;margin:1.2rem 0}
.callout .tag{font-family:var(--mono);font-size:.63rem;letter-spacing:.1em;text-transform:uppercase;
              color:var(--warn-ink);display:block;margin-bottom:.35rem}
.callout p{margin:0;font-size:.88rem;color:var(--ink-2)}
details{border:1px solid var(--line);border-radius:4px;background:var(--panel);
        padding:.75rem 1rem;margin:1.2rem 0}
summary{cursor:pointer;font-family:var(--mono);font-size:.8rem;font-weight:600}
details dl{display:grid;grid-template-columns:auto 1fr;gap:.3rem .9rem;margin:.8rem 0 0;font-size:.82rem}
details dt{font-family:var(--mono);font-size:.72rem;color:var(--ink-3);white-space:nowrap}
details dd{margin:0;color:var(--ink-2)}
footer{margin-top:3rem;padding-top:1.2rem;border-top:1px solid var(--line);
       font-size:.78rem;color:var(--ink-3)}
[data-tip]{cursor:help}
#tip{position:fixed;z-index:50;pointer-events:none;opacity:0;transition:opacity .08s;
     background:var(--ink);color:var(--bg);font-family:var(--mono);font-size:.7rem;
     padding:.35rem .5rem;border-radius:3px;max-width:20rem;white-space:pre-line}
#tip.on{opacity:1}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
"""

_JS = """
(function(){
  var tip=document.getElementById('tip');
  function show(e){
    var t=e.target.closest('[data-tip]'); if(!t){return}
    tip.textContent=t.getAttribute('data-tip');
    tip.classList.add('on');
    var r=t.getBoundingClientRect();
    var x=Math.min(window.innerWidth-tip.offsetWidth-8, r.left);
    var y=r.top-tip.offsetHeight-6;
    tip.style.left=Math.max(8,x)+'px';
    tip.style.top=(y<8? r.bottom+6 : y)+'px';
  }
  function hide(e){ if(e.target.closest('[data-tip]')){tip.classList.remove('on')} }
  document.addEventListener('mouseover',show);
  document.addEventListener('mouseout',hide);
  document.addEventListener('focusin',show);
  document.addEventListener('focusout',hide);
})();
"""


def render_report(
    summary: dict,
    records: Sequence[dict],
    *,
    title: str = "",
) -> str:
    """Build the HTML page for one job."""
    models: dict = (summary.get("leaderboard") or {}).get("models") or {}
    job = summary.get("job_name") or "job"
    name = title or f"{job} leaderboard"

    ids = list(models)
    scores = [models[m]["success_rate"] for m in ids]
    ivs = [tuple(models[m]["success_ci_95"]) for m in ids]
    ranks = rank_interval(scores, ivs) if ids else []
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))

    # Columns come from the configured task set unioned with whatever actually
    # produced records, so a task that never ran shows up as an explicit gap
    # rather than silently vanishing from the grid.
    tasks = sorted(set(summary.get("tasks") or [])
                   | {str(r.get("env_name") or "?") for r in records})
    total_cost = sum(
        (models[m].get("cost_usd") or 0.0) for m in ids if models[m].get("cost_usd")
    )
    n_scored = sum(1 for m in ids if m not in _BASELINE_IDS)

    # ---- KPI tiles: single headline numbers, no chart --------------------
    tiles = [
        ("trials", f"{summary.get('trials', len(records))}"),
        ("agents", f"{len(ids)} <small>({n_scored} scored)</small>"),
        ("tasks", f"{len(tasks)}"),
        ("seeds / task", f"{max((models[m]['episodes'] // max(len(tasks),1)) for m in ids) if ids else 0}"),
        ("total cost", f"${total_cost:.4f}" if total_cost else "&mdash;"),
    ]
    tiles_html = "".join(
        f'<div class="tile"><dt>{_e(k)}</dt><dd>{v}</dd></div>' for k, v in tiles
    )

    # ---- leaderboard ----------------------------------------------------
    rows = []
    for i in order:
        m = ids[i]
        e = models[m]
        lo, hi = e["success_ci_95"]
        is_ref = m in _BASELINE_IDS
        rank = f"{ranks[i][0]}&ndash;{ranks[i][1]}" if ranks[i][0] != ranks[i][1] else str(ranks[i][0])

        nm = sum(
            v for k, v in (e.get("failure_modes") or {}).items()
            if k in FailureMode.NOT_MODEL_FAULT
        )
        flags = ""
        if is_ref:
            flags = '<span class="chip">reference</span>'
        elif nm:
            flags = (f'<span class="chip warn" data-tip="These are harness or provider '
                     f'faults, not capability failures. Fix them and re-run before '
                     f'reading this row.">&#9888; {nm} not model fault</span>')

        cost = e.get("cost_usd")
        cps = e.get("cost_per_success_usd")
        tip = (f"{m}\n{e['episodes']} trials over {e['tasks']} tasks\n"
               f"success {_pct(e['success_rate'])}  95% CI [{_pct(lo)}, {_pct(hi)}]\n"
               f"interval width {e['ci_width_pp']} pp")
        rows.append(f"""<tr class="{'ref' if is_ref else ''}">
 <td class="rank">{rank}</td>
 <td class="agent">{_e(m)} {flags}</td>
 <td data-tip="{_e(tip)}">{_bar_svg(e['success_rate'], lo, hi, muted=is_ref)}</td>
 <td class="n">{_pct(e['success_rate'])}<div class="ci">[{_pct(lo)}, {_pct(hi)}]</div></td>
 <td class="n">{e['ci_width_pp']}</td>
 <td class="n">{_num(e.get('pass_hat_2'), 2)}</td>
 <td class="n">{_num(e.get('score_mean'), 2)}</td>
 <td class="n">{_num(e.get('steps_vs_oracle'), 2)}</td>
 <td class="n">{_num(e.get('soft_spl'), 2)}</td>
 <td class="n">{('$' + format(cost, '.4f')) if cost else '&mdash;'}</td>
 <td class="n">{('$' + format(cps, '.4f')) if cps else '&mdash;'}</td>
</tr>""")

    # ---- task x agent grid: sequential heatmap, every cell labelled ------
    grid_rows = []
    for m in [ids[i] for i in order]:
        per = models[m].get("per_task") or {}
        cells = []
        for t in tasks:
            v = per.get(t)
            if not v or not v.get("trials"):
                cells.append('<td class="cell nodata" data-tip="no trials recorded">&mdash;</td>')
                continue
            s, n = v["successes"], v["trials"]
            frac = s / n
            idx = _ramp_index(frac, len(_RAMP_LIGHT))
            tip = f"{m} / {t}\n{s} of {n} succeeded ({_pct(frac)})"
            cells.append(
                f'<td class="cell" data-tip="{_e(tip)}">'
                f'<span class="fillbg" style="background:var(--ramp-{idx})"></span>'
                f'<span>{s}/{n}</span></td>'
            )
        grid_rows.append(
            f'<tr class="{"ref" if m in _BASELINE_IDS else ""}">'
            f'<td class="agent">{_e(m)}</td>{"".join(cells)}</tr>'
        )

    ramp_vars_light = ";".join(
        [f"--ramp-{i}:{c}" for i, c in enumerate(_RAMP_LIGHT)]
        + [f"--ramp-ink-{i}:{c}" for i, c in enumerate(_RAMP_INK_LIGHT)]
    )
    ramp_vars_dark = ";".join(
        [f"--ramp-{i}:{c}" for i, c in enumerate(_RAMP_DARK)]
        + [f"--ramp-ink-{i}:{c}" for i, c in enumerate(_RAMP_INK_DARK)]
    )
    legend_steps = "".join(
        f'<i style="background:var(--ramp-{i})"></i>' for i in range(len(_RAMP_LIGHT))
    )

    # ---- audit callouts --------------------------------------------------
    audit = []
    ref_id = next((i for i in _SOLVABILITY_IDS if i in models), None)
    oracle = models.get(ref_id) if ref_id else None
    null = models.get("null")
    if oracle is not None:
        broken = [t for t, v in (oracle.get("per_task") or {}).items()
                  if v.get("trials") and v["successes"] < v["trials"]]
        if broken:
            audit.append(
                f"The solvability reference (<code>{_e(ref_id)}</code>) does not solve "
                + ", ".join(f"<code>{_e(t)}</code>" for t in broken)
                + ". Those tasks may be unsolvable as specified, and every agent's score on "
                "them is uninterpretable until that is resolved -- a zero cannot be "
                "attributed to the agent when the reference scores zero too.")
    if null is not None and null.get("success_rate", 0) > 0:
        vac = [t for t, v in (null.get("per_task") or {}).items() if v.get("successes")]
        audit.append(
            "The null agent passes " + ", ".join(f"<code>{_e(t)}</code>" for t in vac)
            + " without taking any action, so those success checks are vacuous.")
    ran = {str(r.get("env_name")) for r in records}
    never_ran = [t for t in (summary.get("tasks") or []) if t not in ran]
    if never_ran:
        audit.append(
            "No trials were recorded for "
            + ", ".join(f"<code>{_e(t)}</code>" for t in never_ran)
            + ". These tasks are in the job configuration but produced nothing, so "
            "the board is incomplete -- check for a crashed environment rather than "
            "reading the remaining columns as the whole grid.")
    if oracle is None or null is None:
        audit.append(
            "This job has no " + ("solvability reference (oracle or scripted probe)"
                                  if oracle is None else "null baseline")
            + ", so " + ("task solvability" if oracle is None
                         else "success-check validity") + " is unverified.")
    audit_html = "".join(
        f'<div class="callout"><span class="tag">&#9888; audit</span><p>{a}</p></div>'
        for a in audit
    )

    rule = summary.get("reporting_rule") or {}
    rule_html = "".join(
        f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in rule.items() if k != "version"
    )

    infra = summary.get("infra_failures")
    infra_html = ""
    if infra:
        infra_html = (
            f'<div class="callout"><span class="tag">&#9888; infrastructure</span><p>'
            f'{infra.get("environment_failures", 0)} environment and '
            f'{infra.get("ambiguous_failures", 0)} ambiguous failure(s): '
            f'{_e(json.dumps(infra.get("reasons") or {}))}. '
            f'These are reported separately and do <strong>not</strong> change the '
            f'denominator.</p></div>')

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(name)}</title>
<style>{_CSS}
:root{{{ramp_vars_light}}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{{ramp_vars_dark}}}}}
:root[data-theme="dark"]{{{ramp_vars_dark}}}
</style></head>
<body>
<div id="tip" role="tooltip"></div>
<div class="wrap">

<h1>{_e(name)}</h1>
<p class="sub">Success rate with 95% credible intervals, reliability, and cost &mdash;
   audited against scripted baselines.</p>
<p class="meta">{_e(summary.get('job_dir') or '')} &middot; generated {stamp}</p>

<dl class="tiles">{tiles_html}</dl>

{audit_html}{infra_html}

<h2>Leaderboard</h2>
<p class="note">Ranked on success rate, but the <strong>rank column is a range</strong>:
   agents whose intervals overlap are not separated by this data, and a single
   number would imply an ordering the sample size does not support. The bar shows
   the point estimate; the line beneath it is the 95% interval.</p>
<div class="scroll"><table>
<thead><tr>
 <th>Rank</th><th>Agent</th><th>Success</th><th class="n">Rate &amp; CI</th>
 <th class="n" title="interval width in percentage points">Width</th>
 <th class="n" title="probability that both of 2 sampled trials succeed">pass^2</th>
 <th class="n" title="successes count 1.0; failures contribute partial progress">Score</th>
 <th class="n" title="steps taken relative to the oracle">St/Orc</th>
 <th class="n" title="graded score weighted by step efficiency">SoftSPL</th>
 <th class="n">Cost</th><th class="n">$/success</th>
</tr></thead>
<tbody>{"".join(rows) or '<tr><td colspan="11">no results</td></tr>'}</tbody>
</table></div>

<h2>Per-task breakdown</h2>
<p class="note">Successes over trials per cell. The aggregate hides where agents
   actually differ &mdash; and a task every agent fails is usually a defect rather
   than a hard task, which is what the baseline rows are for.</p>
<div class="scroll"><table class="grid-tbl">
<thead><tr><th>Agent</th>{"".join(f'<th class="ch">{_e(t)}</th>' for t in tasks)}</tr></thead>
<tbody>{"".join(grid_rows) or '<tr><td>no results</td></tr>'}</tbody>
</table></div>
<div class="legend"><span>0%</span><span class="steps">{legend_steps}</span><span>100%</span>
  <span>&middot; every cell is labelled, so colour is never the only encoding</span></div>

<details><summary>Reporting rule &mdash; how these numbers were computed</summary>
<p class="note">The reporting convention itself moves scores: whether errored trials
   count or are excluded has been measured to account for whole percentage points
   of a reported gap. It is published here so the numbers above are interpretable.</p>
<dl>{rule_html or "<dt>&mdash;</dt><dd>not recorded</dd>"}</dl>
</details>

<footer>Generated by the robotics harness. Intervals are Beta(k+1, n&minus;k+1)
  credible intervals; pass^k is averaged per task, skipping tasks with fewer than
  k trials. Overlapping intervals do not by themselves prove equivalence, and
  non-overlapping ones are not a substitute for a paired test.</footer>
</div>
<script>{_JS}</script>
</body></html>"""


def write_report(
    job_dir,
    out: Optional[str] = None,
    *,
    title: str = "",
) -> Path:
    """Render a job directory to HTML. Returns the written path."""
    from harness.eval.job import JobConfig, build_summary, load_job

    d = Path(job_dir)
    records = load_job(d)
    if not records:
        raise FileNotFoundError(f"no results under {d}")
    cfg = JobConfig(job_name=d.name, log_dir=str(d.parent))
    summary = build_summary(cfg, records)
    path = Path(out) if out else d / "report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(summary, records, title=title), encoding="utf-8")
    return path
