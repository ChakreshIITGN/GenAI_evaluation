# Design tokens — Roland Berger palette for figures

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).

Colors extracted from the live rolandberger.com stylesheet (css/bundle.css,
fetched 2026-08-19), restricted per author instruction to blues, greens and
black — no red. Categorical order validated with a colorblind-safety checker
(lightness band, chroma floor, CVD ΔE separation): all checks pass; the two
light accents carry visible value labels as contrast relief.

## Categorical series order (fixed, never cycled)

| Slot | Hex | Name | Typical role |
|---|---|---|---|
| 1 | `#156C9C` | RB blue | human side / first series |
| 2 | `#72B656` | RB green | AI side / second series |
| 3 | `#00AAC9` | RB turquoise | third series |
| 4 | `#2B8D28` | RB dark green | fourth series |

## Non-data ink

| Hex | Name | Role |
|---|---|---|
| `#000000` | black | text, axis labels, ticks |
| `#8D9399` | RB gray | reference lines (e.g. 50% coin-flip) |
| `#DEE0E3` | RB light gray | grid lines |
| `#FFFFFF` | white | figure background |

Also on brand but unused here (too dark/light for marks on white):
`#000082` deep navy, `#004775` dark blue, `#9CDDF2` light blue.

Used by `src/plots.py`; slides and the app must reuse these tokens.
