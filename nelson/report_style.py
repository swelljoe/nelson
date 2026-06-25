"""Shared visual language for every Nelson HTML report.

One stylesheet, two themes. Historically each report shipped its own ``<style>``: the
product reports in :mod:`nelson.html_report` were dark-navy, the promptlab experiment
scripts were light, the shadow scripts were dark. This module unifies them on a single
token-driven sheet whose **light** palette derives from the promptlab look and whose
**dark** palette derives from the shadow look, with a persistent light/dark toggle.

How to use it:

- ``THEME_VARS`` + ``BASE_CSS`` go inside a ``<style>`` (in that order); a report's own
  bespoke component CSS (scatter plots, matrices, …) follows as a third chunk, but its
  colors should reference the shared ``var(--…)`` tokens so it themes in both modes.
- ``THEME_HEAD`` is a tiny pre-paint ``<script>`` for ``<head>`` that resolves the theme
  from ``localStorage`` else the OS ``prefers-color-scheme`` before paint (no FOUC).
- ``THEME_TOGGLE`` is a fixed-position button (self-contained inline JS) that flips and
  persists the theme.
- ``page(title, body, …)`` assembles a complete document from all of the above — the
  easy path for a new report. Existing reports that hand-build their shell can instead
  interpolate the four constants directly (keeps diffs small).

Class vocabulary is a **superset** of what every current report uses, so migrating a
report is a ``<style>`` swap, not a markup rewrite.
"""

from __future__ import annotations

from html import escape

# --- Theme tokens -----------------------------------------------------------
# Light (default) derives from the promptlab report; dark from the shadow report.
# A report should only ever reference these var(--…) names, never raw hex, so both
# themes stay consistent.
THEME_VARS = """
:root{
  --bg:#ffffff; --surface:#fafafa; --surface-2:#f1f2f4; --border:#dcdee2;
  --text:#1a1a1a; --text-muted:#646a73;
  --accent:#b5530a; --blue:#2563a8; --cyan:#0f7b8a;
  --good:#1c6b2c; --amber:#8a6d00; --bad:#9b1c1c; --yellow:#8a6d00;
  --good-bg:#dff5e1; --amber-bg:#fff6da; --bad-bg:#fde8e8; --muted-bg:#eef0f2;
  --callout-bg:#fff8f0; --callout-border:#f0c890; --callout-edge:#e08a2b;
  --hover:rgba(37,99,168,.06); --shadow:0 1px 2px rgba(0,0,0,.06);
  /* back-compat aliases for pre-unification token names (resolve per-theme) */
  --red:var(--bad); --green:var(--good); --surface2:var(--surface-2); --orange:#c2410c;
}
:root[data-theme="dark"]{
  --bg:#15171c; --surface:#1b1e24; --surface-2:#23272f; --border:#2c2f36;
  --text:#e8e8ea; --text-muted:#9aa0aa;
  --accent:#d98a4a; --blue:#6ea8d8; --cyan:#56cfe1;
  --good:#5fbf7f; --amber:#d6a24a; --bad:#d2685c; --yellow:#d6a24a;
  --good-bg:rgba(95,191,127,.15); --amber-bg:rgba(214,162,74,.15);
  --bad-bg:rgba(210,104,92,.15); --muted-bg:rgba(154,160,170,.12);
  --callout-bg:#241d14; --callout-border:#5a4424; --callout-edge:#d6a24a;
  --hover:rgba(110,168,216,.08); --shadow:0 1px 2px rgba(0,0,0,.35);
  --orange:#e0934a;
}
"""

# --- Base component CSS (a superset of every report's class vocabulary) ------
BASE_CSS = """
*{box-sizing:border-box}
html{color-scheme:light dark}
body{
  font:16px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:var(--bg); color:var(--text);
  margin:0 auto; padding:2.2rem 1.3rem; max-width:1100px;
}
h1{font-size:1.7rem; line-height:1.25; margin:0 0 .3rem; color:var(--text)}
h2{font-size:1.25rem; margin:2rem 0 .7rem; color:var(--blue);
   border-bottom:1px solid var(--border); padding-bottom:.3rem}
h3{font-size:1.05rem; margin:1.2rem 0 .5rem; color:var(--cyan)}
p{margin:.55rem 0}
a{color:var(--blue); text-decoration:none}
a:hover{text-decoration:underline}
.sub,.subtitle,.lede,.intro{color:var(--text-muted); margin:.15rem 0 1.2rem}
.muted{color:var(--text-muted); font-size:.82rem}
small{color:var(--text-muted)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:.82rem}
.big{font-size:1.1rem; font-weight:700}
b,strong{color:var(--text)}

/* tables: clean ruled rows in a rounded surface */
table{border-collapse:collapse; width:100%; margin:.85rem 0; font-size:.9rem;
  background:var(--surface); border:1px solid var(--border); border-radius:8px;
  overflow:hidden}
th,td{padding:.45rem .65rem; text-align:left; border-bottom:1px solid var(--border)}
th{background:var(--surface-2); color:var(--text-muted); font-weight:600;
  font-size:.74rem; text-transform:uppercase; letter-spacing:.03em}
tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--hover)}
td.num,th.num{text-align:right; font-variant-numeric:tabular-nums}
td.l,th.l{text-align:left}
td.c,th.c{text-align:center}
/* opt-in spreadsheet grid (bordered + centered cells) for dense matrices */
table.grid th,table.grid td{border:1px solid var(--border); text-align:center}
table.grid td.l,table.grid th.l{text-align:left}

/* filled status cells (experiment-report vocabulary); centered for matrix legibility */
.hit{background:var(--good-bg); color:var(--good); font-weight:600; text-align:center}
.miss{background:var(--bad-bg); color:var(--bad); text-align:center}
.part{background:var(--amber-bg); color:var(--amber); font-weight:600;
  text-align:center}
.nd{background:var(--muted-bg); color:var(--text-muted); text-align:center}
.good{color:var(--good)} .amber{color:var(--amber)} .bad{color:var(--bad)}

/* badges + tags */
.badge{display:inline-block; padding:.12rem .5rem; border-radius:999px;
  font-size:.74rem; font-weight:600;
  background:var(--surface-2); color:var(--text-muted)}
.badge-high,.badge-confirmed,.badge-error{background:var(--bad-bg); color:var(--bad)}
.badge-medium,.badge-needs_review,.badge-running,.badge-pending{
  background:var(--amber-bg); color:var(--amber)}
.badge-low,.badge-unreviewed{background:var(--muted-bg); color:var(--text-muted)}
.badge-false_positive,.badge-resolved,.badge-complete{
  background:var(--good-bg); color:var(--good)}
.tag{display:inline-block; font-size:.7rem; padding:.05rem .45rem; border-radius:999px;
  background:var(--surface-2); color:var(--text-muted); margin-left:.4rem}
.tag.hard{background:var(--bad-bg); color:var(--bad)}

/* cards */
.card{background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:.9rem 1.1rem; margin:.6rem 0}
.cards{display:flex; gap:.9rem; flex-wrap:wrap; margin:.6rem 0}
.card .v,.cards .v{font-size:1.6rem; font-weight:700}
.card .l,.cards .l{color:var(--text-muted); font-size:.78rem}
.stats-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  gap:.75rem; margin:.8rem 0}
.stat-card{background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:.8rem; text-align:center}
.stat-value{font-size:1.7rem; font-weight:700}
.stat-label{font-size:.78rem; color:var(--text-muted)}

/* callout */
.callout{background:var(--callout-bg); border:1px solid var(--callout-border);
  border-left:5px solid var(--callout-edge); border-radius:8px;
  padding:.9rem 1.1rem; margin:1.1rem 0}
.callout b{color:var(--callout-edge)}

/* code */
code{background:var(--surface-2); padding:1px 5px; border-radius:4px; font-size:.85em;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.code-block{background:var(--surface-2); padding:.6rem .8rem; margin:.5rem 0;
  border:1px solid var(--border); border-radius:6px; overflow-x:auto;
  white-space:pre-wrap; font-size:.82rem;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}

/* findings (scan report) */
.finding{border-left:3px solid var(--border); background:var(--surface);
  border-radius:6px; padding:.8rem 1rem; margin:.6rem 0}
.finding-high{border-left-color:var(--bad)}
.finding-medium{border-left-color:var(--amber)}
.finding-low{border-left-color:var(--text-muted)}
.finding-header{display:flex; gap:.7rem; align-items:center; flex-wrap:wrap}
.review-box{margin-top:.5rem; padding:.5rem .75rem; background:var(--surface-2);
  border-radius:6px; font-size:.88rem}
.file-group{margin:1rem 0}
.file-name{font-family:ui-monospace,monospace; font-size:.9rem; color:var(--cyan);
  padding:.5rem; background:var(--surface-2); border-radius:6px 6px 0 0}

footer{margin-top:2.2rem; padding-top:1rem; border-top:1px solid var(--border);
  color:var(--text-muted); font-size:.8rem}

/* theme toggle */
.theme-toggle{position:fixed; top:.7rem; right:.7rem; z-index:100;
  background:var(--surface); color:var(--text-muted); border:1px solid var(--border);
  border-radius:999px; padding:.3rem .75rem; font:inherit; font-size:.8rem;
  cursor:pointer; box-shadow:var(--shadow)}
.theme-toggle:hover{color:var(--text); border-color:var(--text-muted)}
@media print{.theme-toggle{display:none}}
"""

# --- Theme bootstrapping (pre-paint) + toggle -------------------------------
# Runs synchronously in <head> before the body paints, so data-theme is set first
# (no flash of the wrong theme). Falls back to the OS preference, then light.
THEME_HEAD = (
    "<script>(function(){try{var t=localStorage.getItem('nelson-theme');"
    "if(t!=='light'&&t!=='dark'){t=window.matchMedia&&"
    "window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';}"
    "document.documentElement.setAttribute('data-theme',t);}catch(e){}})();</script>"
)

# Self-contained: flips data-theme on <html> and persists the choice.
THEME_TOGGLE = (
    "<button class=\"theme-toggle\" type=\"button\" "
    "onclick=\"(function(){var r=document.documentElement;"
    "var t=r.getAttribute('data-theme')==='dark'?'light':'dark';"
    "r.setAttribute('data-theme',t);"
    "try{localStorage.setItem('nelson-theme',t);}catch(e){}})()\">"
    "◐ theme</button>"
)


def page(
    title: str,
    body: str,
    *,
    subtitle: str = "",
    extra_css: str = "",
    max_width: str = "1100px",
) -> str:
    """Assemble a complete themed HTML document.

    ``title``/``subtitle`` are escaped here; ``body`` and ``extra_css`` are inserted
    verbatim (callers own their markup/CSS). ``extra_css`` is appended after the shared
    sheet so a report can add or override component styles using the shared tokens.
    """
    sub = f'<p class="subtitle">{escape(subtitle)}</p>' if subtitle else ""
    width_css = f"body{{max-width:{max_width}}}" if max_width != "1100px" else ""
    return (
        f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{escape(title)}</title>{THEME_HEAD}"
        f"<style>{THEME_VARS}{BASE_CSS}{extra_css}{width_css}</style></head>"
        f"<body>{THEME_TOGGLE}\n<h1>{escape(title)}</h1>{sub}\n{body}\n</body></html>"
    )
