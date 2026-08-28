# scripts

## `weekly_stars.py`

Weekly star counter for a GitHub repository. It pulls every stargazer with its
`starred_at` timestamp, buckets them into ISO weeks (Monday → Sunday, UTC) and
reports, per week:

| column | meaning |
| --- | --- |
| `new_stars` | stars gained during the week |
| `total_stars` | cumulative total at the end of the week |
| `delta_vs_prev` | `new_stars` this week − `new_stars` last week |
| `wow_pct` | that delta as a percentage of last week's `new_stars` |
| `growth_pct` | the week's new stars over the total it started from |

Pure standard library — no dependencies.

### Usage

```bash
# console table
python scripts/weekly_stars.py roboflow/trackers

# last 12 weeks, full HTML report with charts, plus CSV
python scripts/weekly_stars.py roboflow/trackers --weeks 12 \
    --html weekly_stars.html --csv weekly_stars.csv

# fetch once, re-render offline as often as you like
python scripts/weekly_stars.py roboflow/trackers --cache stars.json --quiet
python scripts/weekly_stars.py --from-json stars.json --html report.html
```

A token is optional but recommended — unauthenticated requests are capped at 60
per hour, which is not enough for a repo with more than ~6k stars. It is read
from `--token`, `$GITHUB_TOKEN` or `$GH_TOKEN`; only public read scope is
needed. The script waits out rate limits rather than failing.

Other outputs: `--markdown` (a Markdown table), `--json` (the weekly rows),
`--html-fragment` (body-only HTML, for embedding).

The HTML report contains stat tiles, a bar chart of new stars per week, a line
chart of the cumulative total, and the full table. It is self-contained — no
CDN, no fonts to fetch — and follows the viewer's light/dark theme.

### Limits

GitHub's stargazers endpoint stops paginating after 400 pages, so history is
complete up to 40,000 stars. The current week is marked as in progress; its
counts are partial by definition.
