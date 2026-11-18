"""The HT streamline monogram, drawn as vectors so it stays crisp at any size.

H and T are drawn as three parallel flow lines: the H crossbar rises to become the T bar,
and the lines spread apart and warm from navy to copper ("fluid carrying heat").

The logo is the author's personal mark and is not covered by the repository's MIT licence.
"""
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch

INK = '#0b1f3a'
COPPER = '#b87333'
COPPER_LIGHT = '#d9a066'

MARK_XLIM = (-0.08, 1.58)     # axes limits that frame draw_mark()
MARK_YLIM = (-0.08, 1.10)
BADGE_XLIM = (-0.24, 1.72)    # axes limits that frame draw_badge()
BADGE_YLIM = (-0.24, 1.24)


def _mix(c0, c1, s):
    a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    return tuple(a * (1 - s) + b * s)


def _smoothstep(t):
    return t * t * (3 - 2 * t)


def _monogram(ax, cx, cy, s, lw, c0, c1, n=3, gap=0.05, w=2.0, spread=1.0,
              hw=0.44, tx=1.12, bar=(0.80, 1.48), ytop=0.94):
    xmax = bar[1]

    def g_at(x):                                   # line spacing grows along the flow
        return gap * (1 + spread * np.clip(x / xmax, 0, 1))

    def draw(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        pts = np.column_stack([cx + s * x, cy + s * y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        xm = 0.5 * (x[:-1] + x[1:])
        ax.add_collection(LineCollection(segs, colors=[_mix(c0, c1, np.clip(v / xmax, 0, 1)) for v in xm],
                                         linewidths=lw * w, capstyle='round', joinstyle='round', zorder=2))

    offs = [(k - (n - 1) / 2) for k in range(n)]
    for k, o in enumerate(offs):
        for x0 in (0.0, hw):                       # H stems
            xs = x0 + k * g_at(x0)
            draw([xs, xs], [0, 1])
        xa = np.linspace(0.0, hw + (n - 1) * gap, 160)         # crossbar
        ya = 0.5 + o * g_at(xa)
        xb = np.linspace(xa[-1], bar[0], 160)                   # rise
        t = (xb - xb[0]) / (xb[-1] - xb[0])
        yb = 0.5 + o * g_at(xb) + (ytop - 0.5) * _smoothstep(t)
        xc = np.linspace(bar[0], bar[1], 80)                    # T bar
        yc = ytop + o * g_at(xc)
        draw(np.concatenate([xa, xb[1:], xc[1:]]), np.concatenate([ya, yb[1:], yc[1:]]))
        xt = tx + o * g_at(tx) + (n - 1) / 2 * gap              # T stem
        draw([xt, xt], [0, ytop + offs[0] * g_at(tx) - 0.02])


def draw_mark(ax, cx=0.0, cy=0.0, s=1.0, lw=1.0, light=False):
    """Monogram occupying [cx, cx + 1.48 s] x [cy, cy + s]. light=True for dark backgrounds."""
    if light:
        _monogram(ax, cx, cy, s, lw, '#ffffff', COPPER_LIGHT)
    else:
        _monogram(ax, cx, cy, s, lw, INK, COPPER)


def draw_badge(ax, cx=0.0, cy=0.0, s=1.0, lw=1.0):
    """Navy rounded tile with the light monogram."""
    pad = 0.24
    ax.add_patch(FancyBboxPatch((cx - pad * s, cy - pad * s), s * (1.48 + 2 * pad), s * (1 + 2 * pad),
                                boxstyle=f'round,pad=0,rounding_size={0.2 * s}', fc=INK, ec='none', zorder=1))
    draw_mark(ax, cx, cy, s, lw, light=True)


def logo_axes(fig, rect, kind='mark'):
    """Add an invisible axes at rect = [x, y, w, h] (figure coords) and draw the mark or badge in it."""
    ax = fig.add_axes(rect)
    xlim, ylim = (MARK_XLIM, MARK_YLIM) if kind == 'mark' else (BADGE_XLIM, BADGE_YLIM)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal'); ax.axis('off')
    return ax
