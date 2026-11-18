"""Shared visual style for all episodes: colours, typography and logo placement.

Fonts: Fraunces (display serif) and Inter (sans), both SIL Open Font License, bundled in ./fonts.
If a font is unavailable, matplotlib falls back to the next family in each list.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent

for _f in (HERE / 'fonts').glob('*.ttf'):
    font_manager.fontManager.addfont(str(_f))

# ---- colours ----
BG = '#ffffff'
INK = '#0b1f3a'       # navy: headings
TEXT = '#334155'      # body text
MUTED = '#64748b'     # secondary text
COPPER = '#b87333'    # accent
FRAME = '#d6dde6'     # panel borders

# ---- typography ----
_FALLBACK = ['Cambria', 'Segoe UI Symbol', 'DejaVu Sans']   # Greek letters and arrows
FONT = {k: [v] + _FALLBACK for k, v in dict(
    display='Fraunces Display SemiBold', display_it='Fraunces Display Italic',
    serif='Fraunces Display Regular', serif_it='Fraunces Display Italic',
    sans='Inter Regular', sans_med='Inter Medium', sans_semi='Inter SemiBold').items()}
SIZE = dict(title=36, sub=19, head=24)

AUTHOR = 'Hamidreza Toloueinasab'


def apply():
    plt.rcParams.update({
        'font.family': FONT['sans'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': FONT['serif'][0],
        'mathtext.it': f"{FONT['serif_it'][0]}:italic",
        'mathtext.bf': FONT['display'][0],
    })


def spaced(text):
    """Letter-spaced capitals for series labels."""
    return ' '.join(text)   # thin spaces


def rich(fig, x, y, parts, ha='left'):
    """Draw text runs with different fonts/colours on one baseline. parts: (text, family, size, color)."""
    r = fig.canvas.get_renderer()
    W = fig.get_figwidth() * fig.dpi
    arts, total = [], 0.0
    for text, fam, size, color in parts:
        t = fig.text(0, y, text, fontfamily=fam, fontsize=size, color=color, va='baseline')
        w = t.get_window_extent(renderer=r).width / W
        arts.append((t, w))
        total += w
    x0 = x - total if ha == 'right' else x
    for t, w in arts:
        t.set_x(x0)
        x0 += w
    return total

