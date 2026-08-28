"""Weekly GitHub star counter.

Fetches every stargazer of a repository together with the timestamp of the
star, buckets them into ISO weeks (Monday -> Sunday, UTC) and reports the
new stars, the running total, the week-over-week delta and the growth
percentages -- as a console table, CSV/JSON exports and a self-contained
HTML report with charts.

Examples
--------
    python scripts/weekly_stars.py roboflow/trackers
    python scripts/weekly_stars.py roboflow/trackers --weeks 12
    python scripts/weekly_stars.py roboflow/trackers --cache stars.json \
        --csv weekly_stars.csv --html weekly_stars.html
    python scripts/weekly_stars.py --from-json stars.json --html report.html

A token is optional but strongly recommended: unauthenticated requests are
capped at 60/hour, authenticated ones at 5000/hour.  The token is read from
``--token``, ``$GITHUB_TOKEN`` or ``$GH_TOKEN``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Sequence

API_ROOT = "https://api.github.com"
PER_PAGE = 100
# GitHub refuses to paginate past 400 pages on the stargazers endpoint.
MAX_PAGES = 400
USER_AGENT = "weekly-stars-counter"
REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

class GitHubError(RuntimeError):
    pass


def _request(url: str, token: Optional[str]) -> tuple[bytes, dict]:
    headers = {
        "Accept": "application/vnd.github.star+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request) as response:
            return response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        headers = dict(error.headers or {})
        if error.code in (403, 429) and headers.get("X-RateLimit-Remaining") == "0":
            reset = int(headers.get("X-RateLimit-Reset", "0"))
            wait = max(0, reset - int(time.time())) + 1
            raise RateLimited(wait) from error
        body = error.read().decode("utf-8", "replace")[:400]
        raise GitHubError(f"HTTP {error.code} for {url}\n{body}") from error


class RateLimited(RuntimeError):
    def __init__(self, wait_seconds: int) -> None:
        super().__init__(f"rate limited, resets in {wait_seconds}s")
        self.wait_seconds = wait_seconds


def _next_link(link_header: str) -> Optional[str]:
    for part in link_header.split(","):
        section = part.split(";")
        if len(section) < 2:
            continue
        if section[1].strip() == 'rel="next"':
            return section[0].strip().lstrip("<").rstrip(">")
    return None


def fetch_star_dates(
    repo: str,
    token: Optional[str],
    wait_on_rate_limit: bool = True,
    progress: bool = True,
) -> List[dt.datetime]:
    """Return the ``starred_at`` timestamp of every stargazer of ``repo``."""
    url = f"{API_ROOT}/repos/{repo}/stargazers?per_page={PER_PAGE}"
    dates: List[dt.datetime] = []
    for page in range(1, MAX_PAGES + 1):
        while True:
            try:
                payload, headers = _request(url, token)
                break
            except RateLimited as limited:
                if not wait_on_rate_limit:
                    raise
                _log(
                    progress,
                    f"  rate limited; sleeping {limited.wait_seconds}s "
                    "(pass a token to avoid this)",
                )
                time.sleep(limited.wait_seconds)

        batch = json.loads(payload)
        if not batch:
            break
        for item in batch:
            starred_at = item.get("starred_at") if isinstance(item, dict) else None
            if starred_at is None:
                raise GitHubError(
                    "the API returned stargazers without timestamps -- the "
                    "'application/vnd.github.star+json' Accept header was lost"
                )
            dates.append(_parse_iso(starred_at))
        _log(progress, f"  page {page}: {len(dates)} stars so far")

        following = _next_link(headers.get("Link", ""))
        if not following:
            break
        url = following

    dates.sort()
    return dates


def _parse_iso(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        dt.timezone.utc
    )


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

@dataclass
class Week:
    week_start: str          # Monday of the week, YYYY-MM-DD (UTC)
    week_end: str            # Sunday of the week, YYYY-MM-DD (UTC)
    new_stars: int           # stars gained during the week
    total_stars: int         # cumulative total at the end of the week
    delta_vs_prev: int       # new_stars this week - new_stars last week
    wow_pct: Optional[float]      # % change of new_stars vs last week
    growth_pct: Optional[float]   # new_stars as % of the total at week start
    partial: bool            # True for a week that has not finished yet


def _week_start(moment: dt.datetime) -> dt.date:
    day = moment.date()
    return day - dt.timedelta(days=day.weekday())


def aggregate_weekly(
    dates: Sequence[dt.datetime], until: Optional[dt.date] = None
) -> List[Week]:
    """Bucket star timestamps into consecutive weeks with no gaps."""
    if not dates:
        return []

    counts: dict[dt.date, int] = {}
    for moment in dates:
        start = _week_start(moment)
        counts[start] = counts.get(start, 0) + 1

    today = until or dt.datetime.now(dt.timezone.utc).date()
    current_week = today - dt.timedelta(days=today.weekday())
    cursor = min(counts)
    last = max(max(counts), current_week)

    weeks: List[Week] = []
    total = 0
    previous_new: Optional[int] = None
    while cursor <= last:
        new_stars = counts.get(cursor, 0)
        opening_total = total
        total += new_stars
        weeks.append(
            Week(
                week_start=cursor.isoformat(),
                week_end=(cursor + dt.timedelta(days=6)).isoformat(),
                new_stars=new_stars,
                total_stars=total,
                delta_vs_prev=new_stars - previous_new if previous_new is not None else 0,
                wow_pct=(
                    (new_stars - previous_new) / previous_new * 100
                    if previous_new
                    else None
                ),
                growth_pct=(
                    new_stars / opening_total * 100 if opening_total else None
                ),
                partial=cursor == current_week,
            )
        )
        previous_new = new_stars
        cursor += dt.timedelta(days=7)

    return weeks


# --------------------------------------------------------------------------
# output: console + files
# --------------------------------------------------------------------------

def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{value:+.1f}%"


def _fmt_signed(value: int) -> str:
    return f"{value:+d}" if value else "0"


def render_console(weeks: Sequence[Week], repo: str) -> str:
    header = ("Week (Mon)", "New", "Total", "Delta", "WoW %", "Growth %")
    rows = [
        (
            week.week_start + (" *" if week.partial else "  "),
            f"{week.new_stars:,}",
            f"{week.total_stars:,}",
            _fmt_signed(week.delta_vs_prev),
            _fmt_pct(week.wow_pct),
            _fmt_pct(week.growth_pct),
        )
        for week in weeks
    ]
    widths = [
        max(len(header[i]), *(len(row[i]) for row in rows)) if rows else len(header[i])
        for i in range(len(header))
    ]

    def line(cells: Sequence[str]) -> str:
        first = cells[0].ljust(widths[0])
        rest = "  ".join(cell.rjust(widths[i + 1]) for i, cell in enumerate(cells[1:]))
        return f"{first}  {rest}"

    out = [f"Weekly stars -- {repo}", "", line(header), "-" * (sum(widths) + 2 * len(widths))]
    out.extend(line(row) for row in rows)
    if any(week.partial for week in weeks):
        out.append("")
        out.append("* current week, still in progress")
    return "\n".join(out)


def render_markdown(weeks: Sequence[Week], repo: str) -> str:
    out = [
        f"### Weekly stars -- {repo}",
        "",
        "| Week (Mon) | New | Total | Delta | WoW % | Growth % |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for week in weeks:
        label = week.week_start + (" \\*" if week.partial else "")
        out.append(
            f"| {label} | {week.new_stars:,} | {week.total_stars:,} | "
            f"{_fmt_signed(week.delta_vs_prev)} | {_fmt_pct(week.wow_pct)} | "
            f"{_fmt_pct(week.growth_pct)} |"
        )
    if any(week.partial for week in weeks):
        out.extend(["", "\\* current week, still in progress"])
    return "\n".join(out)


def write_csv(weeks: Sequence[Week], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "week_start",
                "week_end",
                "new_stars",
                "total_stars",
                "delta_vs_prev",
                "wow_pct",
                "growth_pct",
                "partial",
            ],
        )
        writer.writeheader()
        for week in weeks:
            row = asdict(week)
            for key in ("wow_pct", "growth_pct"):
                row[key] = "" if row[key] is None else f"{row[key]:.2f}"
            writer.writerow(row)


# --------------------------------------------------------------------------
# output: self-contained HTML report
# --------------------------------------------------------------------------

# Palette roles are declared once as CSS custom properties and referenced by
# role, so the light/dark values swap in a single place.
_CSS = """
:root {
  color-scheme: light;
  --page:           #f9f9f7;
  --surface-1:      #fcfcfb;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;
  --gridline:       #e1e0d9;
  --baseline:       #c3c2b7;
  --border:         rgba(11,11,11,0.10);
  --series-1:       #2a78d6;
  --up:             #006300;
  --down:           #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page:           #0d0d0d;
    --surface-1:      #1a1a19;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --gridline:       #2c2c2a;
    --baseline:       #383835;
    --border:         rgba(255,255,255,0.10);
    --series-1:       #3987e5;
    --up:             #0ca30c;
    --down:           #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page:           #0d0d0d;
  --surface-1:      #1a1a19;
  --text-primary:   #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted:     #898781;
  --gridline:       #2c2c2a;
  --baseline:       #383835;
  --border:         rgba(255,255,255,0.10);
  --series-1:       #3987e5;
  --up:             #0ca30c;
  --down:           #e66767;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 20px 64px;
  background: var(--page);
  color: var(--text-primary);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 26px; line-height: 1.2; margin: 0 0 6px; letter-spacing: -0.01em; }
.sub { color: var(--text-secondary); margin: 0 0 28px; font-size: 14px; }
.card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 22px 16px;
  margin-bottom: 20px;
}
h2 { font-size: 15px; margin: 0 0 2px; font-weight: 600; }
.hint { color: var(--text-muted); font-size: 12.5px; margin: 0 0 14px; }

.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 20px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
.tile .label { color: var(--text-secondary); font-size: 12.5px; margin-bottom: 4px; }
.tile .value { font-size: 27px; line-height: 1.1; letter-spacing: -0.02em; }
.tile .note { font-size: 12.5px; color: var(--text-muted); margin-top: 3px; }
.up { color: var(--up); }
.down { color: var(--down); }

figure { margin: 0; position: relative; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
.grid { stroke: var(--gridline); stroke-width: 1; }
.axis { stroke: var(--baseline); stroke-width: 1; }
.tick { fill: var(--text-muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.bar { fill: var(--series-1); }
.line { fill: none; stroke: var(--series-1); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.dot { fill: var(--series-1); stroke: var(--surface-1); stroke-width: 2; }
.hit { fill: transparent; }
.hit { cursor: crosshair; }
.crosshair { stroke: var(--baseline); stroke-width: 1; stroke-dasharray: 3 3; opacity: 0; }

.tip {
  position: absolute; pointer-events: none; opacity: 0; transition: opacity .08s;
  background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 10px; font-size: 12.5px; line-height: 1.45;
  box-shadow: 0 6px 18px rgba(0,0,0,.14); white-space: nowrap; z-index: 5;
}
.tip .k { color: var(--text-secondary); }
.tip b { font-variant-numeric: tabular-nums; }

.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px; text-align: right; white-space: nowrap; }
th { color: var(--text-secondary); font-weight: 600; font-size: 12.5px; border-bottom: 1px solid var(--baseline); }
td { border-bottom: 1px solid var(--gridline); }
th:first-child, td:first-child { text-align: left; }
tbody tr:hover td { background: var(--gridline); }
.muted { color: var(--text-muted); }
footer { color: var(--text-muted); font-size: 12.5px; margin-top: 22px; }
"""

_JS = """
document.querySelectorAll('figure[data-chart]').forEach(function (fig) {
  var tip = fig.querySelector('.tip');
  var cross = fig.querySelector('.crosshair');
  fig.querySelectorAll('.hit').forEach(function (hit) {
    hit.addEventListener('mouseenter', function () {
      tip.innerHTML = hit.dataset.tip;
      tip.style.opacity = 1;
      if (cross) { cross.setAttribute('x1', hit.dataset.cx); cross.setAttribute('x2', hit.dataset.cx); cross.style.opacity = 1; }
    });
    hit.addEventListener('mousemove', function (event) {
      var box = fig.getBoundingClientRect();
      var x = event.clientX - box.left;
      var y = event.clientY - box.top;
      tip.style.left = Math.min(Math.max(8, x + 14), box.width - tip.offsetWidth - 8) + 'px';
      tip.style.top = Math.max(4, y - tip.offsetHeight - 12) + 'px';
    });
    hit.addEventListener('mouseleave', function () {
      tip.style.opacity = 0;
      if (cross) { cross.style.opacity = 0; }
    });
  });
});
"""

_W, _H = 960.0, 300.0
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 52.0, 12.0, 14.0, 30.0


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _nice_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** (len(str(int(value))) - 1)
    for step in (1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10):
        candidate = step * magnitude
        if candidate >= value:
            return candidate
    return 10 * magnitude


def _y_ticks(maximum: float, count: int = 4) -> List[float]:
    return [maximum * i / count for i in range(count + 1)]


def _x_label_indices(total: int, wanted: int = 8) -> set:
    if total <= wanted:
        return set(range(total))
    step = max(1, round(total / wanted))
    indices = set(range(0, total, step))
    indices.add(total - 1)
    return indices


def _short_date(iso: str) -> str:
    day = dt.date.fromisoformat(iso)
    return day.strftime("%d %b %y")


def _chart_frame(weeks: Sequence[Week], maximum: float) -> List[str]:
    plot_h = _H - _PAD_T - _PAD_B
    parts: List[str] = []
    for tick in _y_ticks(maximum):
        y = _PAD_T + plot_h - (tick / maximum) * plot_h
        parts.append(
            f'<line class="grid" x1="{_PAD_L}" y1="{y:.1f}" x2="{_W - _PAD_R}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{_PAD_L - 8}" y="{y + 4:.1f}" text-anchor="end">'
            f"{int(round(tick)):,}</text>"
        )
    parts.append(
        f'<line class="axis" x1="{_PAD_L}" y1="{_PAD_T + plot_h}" '
        f'x2="{_W - _PAD_R}" y2="{_PAD_T + plot_h}"/>'
    )

    shown = _x_label_indices(len(weeks))
    slot = (_W - _PAD_L - _PAD_R) / max(1, len(weeks))
    for index, week in enumerate(weeks):
        if index not in shown:
            continue
        x = _PAD_L + slot * (index + 0.5)
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{_H - _PAD_B + 16:.1f}" '
            f'text-anchor="middle">{_short_date(week.week_start)}</text>'
        )
    return parts


def _tooltip_html(week: Week) -> str:
    rows = [
        f"<b>{_esc(week.week_start)} &rarr; {_esc(week.week_end)}</b>",
        f'<span class="k">New stars</span> <b>{week.new_stars:,}</b>',
        f'<span class="k">Total</span> <b>{week.total_stars:,}</b>',
        f'<span class="k">vs prev week</span> <b>{_fmt_signed(week.delta_vs_prev)}</b>'
        f" ({_fmt_pct(week.wow_pct)})",
        f'<span class="k">Growth</span> <b>{_fmt_pct(week.growth_pct)}</b>',
    ]
    if week.partial:
        rows.append('<span class="k">week still in progress</span>')
    return _esc("<br>".join(rows))


def _bar_chart(weeks: Sequence[Week]) -> str:
    plot_h = _H - _PAD_T - _PAD_B
    maximum = _nice_ceiling(max((w.new_stars for w in weeks), default=1))
    slot = (_W - _PAD_L - _PAD_R) / max(1, len(weeks))
    bar_w = max(2.0, slot - 2.0)          # 2px surface gap between bars
    radius = min(4.0, bar_w / 2)

    parts = _chart_frame(weeks, maximum)
    for index, week in enumerate(weeks):
        height = (week.new_stars / maximum) * plot_h
        x = _PAD_L + slot * index + (slot - bar_w) / 2
        y = _PAD_T + plot_h - height
        if height > 0:
            parts.append(
                f'<rect class="bar" x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" '
                f'height="{height:.2f}" rx="{min(radius, height / 2):.2f}"/>'
            )
    for index, week in enumerate(weeks):
        x = _PAD_L + slot * index
        parts.append(
            f'<rect class="hit" x="{x:.2f}" y="{_PAD_T}" width="{slot:.2f}" '
            f'height="{plot_h:.2f}" data-cx="{x + slot / 2:.2f}" '
            f'data-tip="{_tooltip_html(week)}"/>'
        )
    return "\n".join(parts)


def _line_chart(weeks: Sequence[Week]) -> str:
    plot_h = _H - _PAD_T - _PAD_B
    maximum = _nice_ceiling(max((w.total_stars for w in weeks), default=1))
    slot = (_W - _PAD_L - _PAD_R) / max(1, len(weeks))

    points = [
        (
            _PAD_L + slot * (index + 0.5),
            _PAD_T + plot_h - (week.total_stars / maximum) * plot_h,
        )
        for index, week in enumerate(weeks)
    ]
    parts = _chart_frame(weeks, maximum)
    parts.append(
        f'<line class="crosshair" x1="0" y1="{_PAD_T}" x2="0" y2="{_PAD_T + plot_h}"/>'
    )
    path = " ".join(
        ("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}" for i, (x, y) in enumerate(points)
    )
    parts.append(f'<path class="line" d="{path}"/>')
    if points:
        x, y = points[-1]
        parts.append(f'<circle class="dot" cx="{x:.2f}" cy="{y:.2f}" r="4.5"/>')
    for index, week in enumerate(weeks):
        x = _PAD_L + slot * index
        parts.append(
            f'<rect class="hit" x="{x:.2f}" y="{_PAD_T}" width="{slot:.2f}" '
            f'height="{plot_h:.2f}" data-cx="{x + slot / 2:.2f}" '
            f'data-tip="{_tooltip_html(week)}"/>'
        )
    return "\n".join(parts)


def _figure(title: str, hint: str, body: str) -> str:
    return (
        f'<section class="card"><h2>{_esc(title)}</h2>'
        f'<p class="hint">{_esc(hint)}</p>'
        f'<figure data-chart><svg viewBox="0 0 {int(_W)} {int(_H + 8)}" '
        f'role="img" aria-label="{_esc(title)}">{body}</svg>'
        f'<div class="tip"></div></figure></section>'
    )


def _tiles(weeks: Sequence[Week]) -> str:
    complete = [w for w in weeks if not w.partial]
    latest = complete[-1] if complete else weeks[-1]
    total = weeks[-1].total_stars
    as_of = min(
        dt.date.fromisoformat(weeks[-1].week_end),
        dt.datetime.now(dt.timezone.utc).date(),
    ).isoformat()
    recent = complete[-12:] or [latest]
    average = sum(w.new_stars for w in recent) / len(recent)
    best = max(weeks, key=lambda w: w.new_stars)
    trend = "up" if latest.delta_vs_prev > 0 else "down" if latest.delta_vs_prev < 0 else "muted"

    def tile(label: str, value: str, note: str, cls: str = "") -> str:
        return (
            f'<div class="tile"><div class="label">{_esc(label)}</div>'
            f'<div class="value {cls}">{value}</div>'
            f'<div class="note">{_esc(note)}</div></div>'
        )

    return (
        '<div class="tiles">'
        + tile("Total stars", f"{total:,}", f"as of {as_of}")
        + tile(
            "Last complete week",
            f"{latest.new_stars:,}",
            f"week of {latest.week_start}",
        )
        + tile(
            "vs previous week",
            f"{_fmt_signed(latest.delta_vs_prev)} ({_fmt_pct(latest.wow_pct)})",
            "change in new stars",
            trend,
        )
        + tile(
            "Average / week",
            f"{average:,.0f}",
            f"last {len(recent)} complete weeks",
        )
        + "</div>"
    )


def _table(weeks: Sequence[Week]) -> str:
    rows = []
    for week in reversed(weeks):
        trend = "up" if week.delta_vs_prev > 0 else "down" if week.delta_vs_prev < 0 else "muted"
        label = _esc(week.week_start)
        if week.partial:
            label += ' <span class="muted">(in progress)</span>'
        rows.append(
            f"<tr><td>{label}</td>"
            f"<td>{week.new_stars:,}</td>"
            f"<td>{week.total_stars:,}</td>"
            f'<td class="{trend}">{_fmt_signed(week.delta_vs_prev)}</td>'
            f'<td class="{trend}">{_fmt_pct(week.wow_pct)}</td>'
            f"<td>{_fmt_pct(week.growth_pct)}</td></tr>"
        )
    return (
        '<section class="card"><h2>Week by week</h2>'
        '<p class="hint">Newest first. "Delta" and "WoW %" compare new stars against '
        'the previous week; "Growth %" is the week\'s new stars over the total it '
        "started from.</p>"
        '<div class="scroll"><table><thead><tr>'
        "<th>Week (Mon, UTC)</th><th>New stars</th><th>Total</th>"
        "<th>Delta</th><th>WoW %</th><th>Growth %</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></section>"
    )


def render_html(weeks: Sequence[Week], repo: str, full_page: bool = True) -> str:
    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = (
        '<div class="wrap">'
        f"<h1>{_esc(repo)} &mdash; weekly stars</h1>"
        f'<p class="sub">{len(weeks)} weeks, {weeks[-1].total_stars:,} stars total. '
        f"Weeks run Monday&ndash;Sunday in UTC.</p>"
        + _tiles(weeks)
        + _figure(
            "New stars per week",
            "How many people starred the repository during each week.",
            _bar_chart(weeks),
        )
        + _figure(
            "Total stars",
            "Cumulative star count at the end of each week.",
            _line_chart(weeks),
        )
        + _table(weeks)
        + f'<footer>Generated {generated} from the GitHub stargazers API.</footer>'
        "</div>"
    )
    head = f"<title>{_esc(repo)} weekly stars</title><style>{_CSS}</style>"
    page = f"{head}{body}<script>{_JS}</script>"
    if not full_page:
        return page
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"{head}</head><body>{body}<script>{_JS}</script></body></html>"
    )


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def load_cache(path: str) -> tuple[str, List[dt.datetime]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        repo = payload.get("repo", "unknown/unknown")
        raw = payload["starred_at"]
    else:                                   # bare list of timestamps
        repo, raw = "unknown/unknown", payload
    return repo, sorted(_parse_iso(value) for value in raw)


def save_cache(path: str, repo: str, dates: Iterable[dt.datetime]) -> None:
    payload = {
        "repo": repo,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "starred_at": [moment.isoformat() for moment in dates],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Weekly star counter for a GitHub repository.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "repo", nargs="?", help="repository as owner/name, e.g. roboflow/trackers"
    )
    parser.add_argument("--token", help="GitHub token (default: $GITHUB_TOKEN/$GH_TOKEN)")
    parser.add_argument(
        "--weeks", type=int, help="only report the most recent N weeks"
    )
    parser.add_argument("--cache", help="write the fetched star timestamps here")
    parser.add_argument(
        "--from-json", dest="from_json", help="read star timestamps from a cache file"
    )
    parser.add_argument("--csv", help="write the weekly table as CSV")
    parser.add_argument("--json", dest="json_out", help="write the weekly table as JSON")
    parser.add_argument("--markdown", help="write the weekly table as Markdown")
    parser.add_argument("--html", help="write the HTML report with charts")
    parser.add_argument(
        "--html-fragment",
        action="store_true",
        help="with --html, emit a body fragment instead of a full document",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.from_json:
        repo, dates = load_cache(args.from_json)
        if args.repo:
            repo = args.repo
    else:
        if not args.repo:
            print("error: pass a repo (owner/name) or --from-json", file=sys.stderr)
            return 2
        if not REPO_RE.match(args.repo):
            print(f"error: '{args.repo}' is not owner/name", file=sys.stderr)
            return 2
        repo = args.repo
        token = args.token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not token:
            _log(not args.quiet, "no token found; limited to 60 requests/hour")
        _log(not args.quiet, f"fetching stargazers of {repo} ...")
        try:
            dates = fetch_star_dates(repo, token, progress=not args.quiet)
        except GitHubError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        if args.cache:
            save_cache(args.cache, repo, dates)
            _log(not args.quiet, f"cached {len(dates)} timestamps in {args.cache}")

    if not dates:
        print(f"{repo} has no stars yet.")
        return 0

    weeks = aggregate_weekly(dates)
    if args.weeks:
        weeks = weeks[-args.weeks :]

    print(render_console(weeks, repo))

    if args.csv:
        write_csv(weeks, args.csv)
        _log(not args.quiet, f"wrote {args.csv}")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(
                {"repo": repo, "weeks": [asdict(week) for week in weeks]},
                handle,
                indent=2,
            )
        _log(not args.quiet, f"wrote {args.json_out}")
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(render_markdown(weeks, repo) + "\n")
        _log(not args.quiet, f"wrote {args.markdown}")
    if args.html:
        with open(args.html, "w", encoding="utf-8") as handle:
            handle.write(render_html(weeks, repo, full_page=not args.html_fragment))
        _log(not args.quiet, f"wrote {args.html}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
