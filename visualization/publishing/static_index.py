"""Self-contained, escaped HTML index for a static visualization publication."""

from __future__ import annotations

import html
from collections.abc import Iterable

from visualization.contracts import FigureData


def build_static_index(figures: Iterable[FigureData], *, target_scope: str) -> str:
    cards = []
    for figure in figures:
        title = html.escape(figure.title, quote=True)
        caption = html.escape(figure.caption, quote=True)
        limitations = "".join(
            f"<li>{html.escape(item, quote=True)}</li>" for item in figure.limitations
        )
        sources = ", ".join(
            html.escape(binding.source_id, quote=True)
            for binding in figure.source_bindings
        )
        stem = html.escape(figure.figure_id, quote=True)
        cards.append(
            "\n".join(
                (
                    '<article class="figure-card">',
                    f"<h2>{stem}: {title}</h2>",
                    (
                        f'<a href="figures/{stem}.svg">'
                        f'<img src="figures/{stem}.svg" alt="{title}"></a>'
                    ),
                    f"<p>{caption}</p>",
                    f'<p class="sources">Sources: {sources}</p>',
                    f"<ul>{limitations}</ul>",
                    (
                        '<p class="formats">'
                        f'<a href="figures/{stem}.svg">SVG</a> '
                        f'<a href="figures/{stem}.pdf">PDF</a> '
                        f'<a href="figures/{stem}.png">PNG</a></p>'
                    ),
                    "</article>",
                )
            )
        )
    escaped_scope = html.escape(target_scope, quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CVI research visualizations</title>
<style>
:root {{ color-scheme: light; font-family: "DejaVu Sans", sans-serif; color: #17212b; background: #eeeae1; }}
body {{ margin: 0 auto; max-width: 76rem; padding: 2rem; }}
header {{ border-bottom: 3px solid #2667a9; margin-bottom: 2rem; }}
.scope {{ color: #61707d; text-transform: uppercase; letter-spacing: .08em; }}
.figure-card {{ background: #faf8f3; border: 1px solid #d8dee3; margin: 1.5rem 0; padding: 1.25rem; }}
.figure-card img {{ display: block; height: auto; max-width: 100%; width: 100%; }}
.sources {{ color: #61707d; font-size: .85rem; }}
.formats a {{ margin-right: 1rem; }}
a {{ color: #2667a9; }}
</style>
</head>
<body>
<header><p class="scope">{escaped_scope} scope</p><h1>Research visualizations</h1></header>
<main>{"".join(cards)}</main>
</body>
</html>
"""
