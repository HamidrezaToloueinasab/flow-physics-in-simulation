"""
Render the 1080x1080 video, cover options and validation figures for episode 04 (side-heated cavity).

Usage:
  python render.py --data run/movie --media ../media [--layout grid|chart|hero] [--cover_only]
  python render.py --data run --media ../media --static          # cover from the steady benchmark fields
  python render.py --data run --media ../media --validation_only
"""
import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from style import brand                               # noqa: E402
from style.brand import INK, TEXT, MUTED, COPPER, FRAME, FONT, SIZE, AUTHOR, rich, spaced  # noqa: E402
from style.logo import draw_badge, draw_mark, logo_axes  # noqa: E402

BG = '#ffffff'
CASES = [  # key, Ra label, short description, accent colour
    ('1000', '1,000', 'slow loop, heat mostly conducted', '#8a9bb3'),
    ('10000', '10,000', 'the flow starts to carry the heat', '#5f86b8'),
    ('100000', '100,000', 'thin layers along the hot and cold walls', '#2f6fae'),
    ('1e+06', '1,000,000', 'fast wall jets around a calm, layered core', COPPER),
]
RAS = [1e3, 1e4, 1e5, 1e6]
CMAP = LinearSegmentedColormap.from_list(
    'brand', ['#0b1f3a', '#2f5f96', '#c9d6e6', '#f7f3ee', '#ecc9a2', '#b87333', '#6e3a12'], N=512)
HOT, COLD = '#9a4f18', '#1f3f6b'
INSUL_FC, INSUL_EC, EDGE = '#9aa6b5', '#3e4a5c', '#0b1f3a'
L = 0.06
W = 0.045                     # wall thickness in cavity units (drawing only)
# reference mean Nusselt numbers: de Vahl Davis (1983) Table V; Le Quere (1991) Table 4 at Ra = 1e6
NU_DVD = {1e3: 1.118, 1e4: 2.243, 1e5: 4.519, 1e6: 8.800}
NU_LQ = {1e6: 8.825}


# ---------------------------------------------------------------- drawing helpers
def streamfunction(u):
    """psi(x, y) = integral_0^y u dy' on a uniform grid; arrays indexed [x, y]."""
    n = u.shape[1]
    return np.cumsum(u, axis=1) / n


def cavity_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(-W, 1 + W); ax.set_ylim(-W, 1 + W); ax.set_aspect('equal'); ax.axis('off')
    # walls: hot left, cold right, insulated (hatched, darker) top and bottom
    ax.add_patch(Rectangle((-W, -W), W, 1 + 2 * W, fc=HOT, ec='none', zorder=3))
    ax.add_patch(Rectangle((1, -W), W, 1 + 2 * W, fc=COLD, ec='none', zorder=3))
    for y0 in (-W, 1):
        ax.add_patch(Rectangle((0, y0), 1, W, fc=INSUL_FC, ec=INSUL_EC, hatch='////', lw=0.0, zorder=3))
    # crisp edges: fluid/wall interface and the outer outline of the walls
    ax.add_patch(Rectangle((0, 0), 1, 1, fc='none', ec=EDGE, lw=1.3, zorder=4))
    ax.add_patch(Rectangle((-W, -W), 1 + 2 * W, 1 + 2 * W, fc='none', ec=EDGE, lw=1.0, zorder=4))
    return ax


class CavityArt:
    """Temperature image + streamline contours with direction arrows for one cavity, updated per frame."""

    LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
    ARROW_LEVELS = (1,)              # index into LEVELS: one loop gets arrows on all four sides

    def __init__(self, ax, arrow_scale=1.0):
        self.ax = ax
        self.im = ax.imshow(np.zeros((4, 4)), origin='lower', cmap=CMAP, vmin=0, vmax=1, extent=[0, 1, 0, 1],
                            interpolation='bicubic', zorder=1)
        self.cs = None
        self.arrows = []
        self.arrow_scale = arrow_scale

    def update(self, T, u, v):
        self.im.set_data(T.astype(np.float32).T)
        if self.cs is not None:
            self.cs.remove()
            self.cs = None
        for a in self.arrows:
            a.remove()
        self.arrows = []
        u = u.astype(np.float32); v = v.astype(np.float32)
        psi = np.abs(streamfunction(u))
        m = psi.max()
        if m <= 1e-3:
            return
        n = psi.shape[0]
        g = (np.arange(n) + 0.5) / n
        self.cs = self.ax.contour(g, g, (psi / m).T, levels=self.LEVELS, colors=INK, linewidths=0.8,
                                  alpha=0.6, zorder=2)
        paths = self.cs.get_paths()
        for li in self.ARROW_LEVELS:
            if li >= len(paths):
                continue
            for poly in paths[li].to_polygons():
                if len(poly) < 20:
                    continue
                for pick in (np.argmax(poly[:, 1]), np.argmin(poly[:, 1]), np.argmin(poly[:, 0]), np.argmax(poly[:, 0])):
                    x, y = poly[pick]
                    i = min(int(x * n), n - 1); j = min(int(y * n), n - 1)
                    d = np.array([u[i, j], v[i, j]])
                    nd = np.hypot(*d)
                    if nd < 1e-6:
                        continue
                    d = d / nd * 0.03 * self.arrow_scale
                    a = FancyArrowPatch((x - d[0], y - d[1]), (x + d[0], y + d[1]), arrowstyle='-|>',
                                        mutation_scale=8 * self.arrow_scale, lw=0, color=INK, alpha=0.9,
                                        zorder=2.5)
                    self.ax.add_patch(a)
                    self.arrows.append(a)


def header(fig, title_a, title_b, sub):
    fig.text(L, 0.940, spaced('FLOW PHYSICS IN SIMULATION') + '   ·   ' + spaced('Nº 04'), color=COPPER,
             fontsize=12.5, fontfamily=FONT['sans_semi'])
    rich(fig, L, 0.872, [(title_a, FONT['display'], SIZE['title'], INK),
                         (title_b, FONT['display_it'], SIZE['title'], COPPER)])
    fig.text(L, 0.832, sub, color=MUTED, fontsize=SIZE['sub'], fontfamily=FONT['serif_it'])
    bw = 0.072; bh = bw * 1.48 / 1.96
    draw_badge(logo_axes(fig, [1 - L - bw, 0.958 - bh, bw, bh], kind='badge'), lw=0.55)


def footer(fig):
    fig.add_artist(plt.Line2D([L, 1 - L], [0.042, 0.042], color=FRAME, lw=1))
    mh = 0.024; mw = mh * 1.66 / 1.18
    draw_mark(logo_axes(fig, [L, 0.010, mw, mh], kind='mark'), lw=0.45)
    fig.text(L + mw + 0.010, 0.016, AUTHOR, color=INK, fontsize=13, fontfamily=FONT['sans_semi'])
    fig.text(1 - L, 0.016, 'Thermal lattice Boltzmann method in Python', color=MUTED, fontsize=12,
             ha='right', fontfamily=FONT['sans'])


def legend_row(fig, y, x0=L, with_time=True):
    """Temperature colour bar, wall key and (optionally) the time readout."""
    cax = fig.add_axes([x0 + 0.112, y, 0.20, 0.014])
    cax.imshow(np.linspace(0, 1, 256)[None, :], cmap=CMAP, aspect='auto', extent=[0, 1, 0, 1])
    cax.set_xticks([]); cax.set_yticks([])
    for s in cax.spines.values():
        s.set_visible(False)
    fig.text(x0, y + 0.007, 'Temperature', fontsize=11.5, color=INK, va='center', fontfamily=FONT['sans_med'])
    fig.text(x0 + 0.112, y - 0.013, 'cold wall', fontsize=9.5, color=TEXT, va='center')
    fig.text(x0 + 0.312, y - 0.013, 'hot wall', fontsize=9.5, color=TEXT, va='center', ha='right')
    # wall key
    kx = x0 + 0.345
    for i, (fc, hatch, txt) in enumerate(((HOT, None, 'hot wall'), (COLD, None, 'cold wall'),
                                          (INSUL_FC, '////', 'insulated wall'))):
        fig.add_artist(Rectangle((kx, y - 0.002), 0.016, 0.018, transform=fig.transFigure, fc=fc,
                                 ec=INSUL_EC if hatch else 'none', hatch=hatch, lw=0))
        fig.text(kx + 0.022, y + 0.007, txt, fontsize=10, color=TEXT, va='center')
        kx += 0.022 + 0.012 * len(txt) * 0.62 + 0.03
    fig.add_artist(plt.Line2D([kx, kx + 0.03], [y + 0.007] * 2, color=INK, lw=0.9, alpha=0.6,
                              transform=fig.transFigure))
    fig.add_artist(FancyArrowPatch((kx + 0.008, y + 0.007), (kx + 0.024, y + 0.007), arrowstyle='-|>',
                                   mutation_scale=9, lw=0, color=INK, transform=fig.transFigure))
    fig.text(kx + 0.038, y + 0.007, 'streamlines, flow direction', fontsize=10, color=TEXT, va='center')
    t_txt = fig.text(1 - L, y + 0.007, '', fontsize=11.5, color=INK, va='center', ha='right',
                     fontfamily=FONT['sans_med']) if with_time else None
    return t_txt


# ---------------------------------------------------------------- layouts
TITLE = ('Heated from the side, air ', 'never rests')
SUB = 'A square box of air, hot wall left, cold wall right, at four Rayleigh numbers'


def build_grid(fig):
    """Option A: 2 x 2 cavities, Nu under each."""
    header(fig, *TITLE, SUB)
    s = 0.27
    xs = [L + 0.03, 0.545]
    ys = [0.478, 0.135]
    arts, nu_txt = [], []
    for k, (key, lab, desc, col) in enumerate(CASES):
        x0, y0 = xs[k % 2], ys[k // 2]
        rich(fig, x0, y0 + s + 0.036, [('Ra', FONT['display_it'], 16, INK), (f' = {lab}', FONT['display'], 16, INK)])
        fig.text(x0, y0 + s + 0.014, desc, fontsize=10.5, color=TEXT)
        ax = cavity_axes(fig, [x0, y0, s, s])
        arts.append(CavityArt(ax))
        nu_txt.append(fig.text(x0 + s, y0 + s + 0.036, '', ha='right', fontsize=16, color=col,
                               fontfamily=FONT['display']))
    t_txt = legend_row(fig, 0.088)
    footer(fig)
    return arts, nu_txt, t_txt, None


def build_chart(fig):
    """Option B: four cavities in a row, Nu vs Ra chart with benchmark points underneath."""
    header(fig, *TITLE, SUB)
    s = 0.205
    gap = (1 - 2 * L - 4 * s) / 3
    arts, nu_txt = [], []
    y0 = 0.535
    for k, (key, lab, desc, col) in enumerate(CASES):
        x0 = L + k * (s + gap)
        rich(fig, x0, y0 + s + 0.030, [('Ra', FONT['display_it'], 17, INK), (f' = {lab}', FONT['display'], 17, INK)])
        ax = cavity_axes(fig, [x0, y0, s, s])
        arts.append(CavityArt(ax, arrow_scale=1.5))
        nu_txt.append(fig.text(x0, y0 - 0.028, '', fontsize=17, color=col, fontfamily=FONT['display']))
    legend_row(fig, 0.455, with_time=False)
    # chart
    fig.text(L, 0.395, 'Heat transfer across the box: Nusselt number vs Rayleigh number', fontsize=13, color=INK,
             fontfamily=FONT['sans_semi'])
    t_txt = fig.text(1 - L, 0.395, '', fontsize=12, color=INK, ha='right', fontfamily=FONT['sans_med'])
    cx = fig.add_axes([L + 0.06, 0.110, 1 - 2 * L - 0.08, 0.255])
    cx.set_xscale('log'); cx.set_yscale('log')
    cx.set_xlim(5e2, 2e6); cx.set_ylim(0.8, 14)
    cx.set_xticks(RAS); cx.set_xticklabels(['10³', '10⁴', '10⁵', '10⁶'], fontsize=11, color=TEXT)
    cx.set_yticks([1, 2, 5, 10]); cx.set_yticklabels(['1', '2', '5', '10'], fontsize=11, color=TEXT)
    cx.minorticks_off()
    cx.set_xlabel('Rayleigh number, Ra', fontsize=11, color=MUTED)
    cx.set_ylabel('Nusselt number, Nu', fontsize=11, color=MUTED)
    for sp in ('top', 'right'):
        cx.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        cx.spines[sp].set_color(FRAME)
    cx.grid(alpha=0.25)
    cx.axhline(1, color=MUTED, lw=1, ls=(0, (3, 3)))
    cx.text(1.9e6, 0.93, 'Nu = 1: conduction only', fontsize=9.5, color=MUTED, ha='right', va='top')
    cx.plot(RAS, [NU_DVD[r] for r in RAS], 's', mfc='none', mec=INK, ms=11, mew=1.4,
            label='de Vahl Davis (1983), steady state')
    dots = [cx.plot([r], [1], 'o', color=col, ms=8, zorder=3)[0] for r, (_, _, _, col) in zip(RAS, CASES)]
    cx.plot([], [], 'o', color=MUTED, ms=8, label='Simulation')
    cx.legend(frameon=False, fontsize=10, loc='upper left')
    footer(fig)
    return arts, nu_txt, t_txt, dots


def build_hero(fig):
    """Option C: one large Ra = 1e6 cavity with the three lower Ra stacked beside it."""
    header(fig, *TITLE, SUB)
    big = 0.52
    xb, yb = L + 0.0, 0.165
    small = 0.175
    xs = 1 - L - small
    arts, nu_txt = [None] * 4, [None] * 4
    key, lab, desc, col = CASES[3]
    rich(fig, xb, yb + big + 0.035, [('Ra', FONT['display_it'], 22, INK), (f' = {lab}', FONT['display'], 22, INK)])
    fig.text(xb, yb + big + 0.012, desc, fontsize=12, color=TEXT)
    arts[3] = CavityArt(cavity_axes(fig, [xb, yb, big, big]))
    nu_txt[3] = fig.text(xb + big, yb + big + 0.035, '', ha='right', fontsize=22, color=col,
                         fontfamily=FONT['display'])
    for k in range(3):
        key, lab, desc, col = CASES[k]
        y0 = yb + (2 - k) * (small + 0.017)
        rich(fig, xb + big + 0.03, y0 + small - 0.025, [('Ra', FONT['display_it'], 15, INK),
                                                   (f' = {lab}', FONT['display'], 15, INK)])
        nu_txt[k] = fig.text(xb + big + 0.03, y0 + small - 0.060, '', fontsize=15, color=col,
                             fontfamily=FONT['display'])
        arts[k] = CavityArt(cavity_axes(fig, [xs, y0, small, small]))
    t_txt = legend_row(fig, 0.088)
    footer(fig)
    return arts, nu_txt, t_txt, None


LAYOUTS = dict(grid=build_grid, chart=build_chart, hero=build_hero)


# ---------------------------------------------------------------- data
def load_movie(data):
    d = np.load(data / 'movie.npz')
    keys = [k for k, *_ in CASES]
    return (d['times'], [d[f'T_{k}'] for k in keys], [d[f'u_{k}'] for k in keys], [d[f'v_{k}'] for k in keys],
            [d[f'Nu_{k}'] for k in keys])


def load_static(data, Lg=256):
    js = {r['Ra']: r for r in json.loads((data / f'bench_L{Lg}.json').read_text())['runs']}
    T, U, V, N = [], [], [], []
    for r in RAS:
        f = np.load(data / f'fields_L{Lg}_Ra{r:g}.npz')
        T.append(f['T'][None, ::2, ::2]); U.append(f['u'][None, ::2, ::2]); V.append(f['v'][None, ::2, ::2])
        N.append(np.array([js[r]['Nu_avg']]))
    return np.array([np.nan]), T, U, V, N


def set_frame(k, times, Ts, Us, Vs, Ns, arts, nu_txt, t_txt, dots, note=''):
    for i in range(4):
        arts[i].update(Ts[i][k], Us[i][k], Vs[i][k])
        nu_txt[i].set_text(f'Nu = {Ns[i][k]:.2f}')
        if dots is not None:
            dots[i].set_ydata([Ns[i][k]])
    if t_txt is not None:
        t = times[k]
        txt = '' if np.isnan(t) else (f'Time: {t:.1f} L/U' if t < 20 else f'Time: {t:.0f} L/U')
        t_txt.set_text(txt + (f'   ·   {note}' if note else ''))


# Video playback: piecewise-linear map from video time (s) to simulated time (L/U).
# The early transient (where the three lower-Ra cavities do all their changing) is shown in slow motion,
# the slower Ra = 1e6 approach is sped up, and the steady state is held at the end. The current playback
# speed is always shown next to the time readout.
PLAYBACK = [  # (video seconds, sim start, sim end, label)
    (6.0, 0.1, 12.0, 'slow motion'),
    (4.0, 12.0, 60.0, None),
]
HOLD_S = 2.0


def playback_sequence(times, fps=30):
    """List of (stored-frame index, label) for every video frame."""
    base = (PLAYBACK[0][2] - PLAYBACK[0][1]) / PLAYBACK[0][0]
    seq = []
    for dur, t0, t1, label in PLAYBACK:
        n = int(round(dur * fps))
        speed = (t1 - t0) / dur / base
        lab = label or f'{speed:.0f}× faster'
        for j in range(n):
            ts = t0 + (t1 - t0) * j / n
            seq.append((int(np.argmin(np.abs(times - ts))), lab))
    seq += [(len(times) - 1, 'steady state')] * int(round(HOLD_S * fps))
    return seq


def render(data, media, layout, cover_only, static, cover_name, cover_t):
    brand.apply()
    times, Ts, Us, Vs, Ns = load_static(data) if static else load_movie(data)
    fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor=BG)
    arts, nu_txt, t_txt, dots = LAYOUTS[layout](fig)
    kc = len(times) - 1 if (static or cover_t is None) else int(np.argmin(np.abs(times - cover_t)))
    if cover_only or static:
        set_frame(kc, times, Ts, Us, Vs, Ns, arts, nu_txt, t_txt, dots)
        fig.savefig(media / cover_name, dpi=100, facecolor=BG); plt.close(fig); return
    w = imageio.get_writer(media / 'cavity.mp4', fps=30, codec='libx264', quality=8,
                           pixelformat='yuv420p', macro_block_size=8)
    # t = 0 is skipped (uniform fluid, sudden wall heating: Nu is a meaningless spike); see PLAYBACK
    for k, note in playback_sequence(times):
        set_frame(k, times, Ts, Us, Vs, Ns, arts, nu_txt, t_txt, dots, note)
        fig.canvas.draw()
        w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    w.close()
    set_frame(kc, times, Ts, Us, Vs, Ns, arts, nu_txt, t_txt, dots)
    fig.savefig(media / cover_name, dpi=100, facecolor=BG); plt.close(fig)


# peak velocities (kappa/L): de Vahl Davis (1983) Table V; Le Quere (1991) Table 4 converted from (kappa/L) Ra^0.5
U_DVD = {1e3: 3.649, 1e4: 16.178, 1e5: 34.73, 1e6: 64.63}
V_DVD = {1e3: 3.697, 1e4: 19.617, 1e5: 68.59, 1e6: 219.36}
U_LQ, V_LQ = {1e6: 64.834}, {1e6: 220.56}


def render_validation(data, media, stacked=False):
    """(a) mean Nu vs Ra against both references, with velocity checks; (b) grid convergence at Ra = 1e6."""
    brand.apply()
    k = 1.45 if stacked else 1.0
    plt.rcParams.update({'font.size': 10 * k, 'axes.labelsize': 10 * k, 'xtick.labelsize': 9.5 * k,
                         'ytick.labelsize': 9.5 * k})
    runs = {}
    for Lg in (64, 128, 256, 384):
        f = data / f'bench_L{Lg}.json'
        if f.exists():
            for r in json.loads(f.read_text())['runs']:
                runs[(Lg, r['Ra'])] = r
    fin = {ra: runs[(384, ra)] if (384, ra) in runs else runs[(256, ra)] for ra in RAS}
    if stacked:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.8, 13.5), dpi=100, facecolor='white')
    else:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=100, facecolor='white')

    # (a) mean Nusselt number vs Ra
    a1.plot(RAS, [NU_DVD[r] for r in RAS], 's', mfc='none', mec=INK, ms=10 * k, mew=1.4,
            label='de Vahl Davis (1983)')
    a1.plot([1e6], [NU_LQ[1e6]], 'D', mfc='none', mec='#2f6fae', ms=8 * k, mew=1.4, label='Le Quéré (1991)')
    a1.plot(RAS, [fin[r]['Nu_avg'] for r in RAS], 'o', color=COPPER, ms=6 * k, label='simulation, finest grid')
    a1.set_xscale('log'); a1.set_yscale('log'); a1.minorticks_off()
    a1.set_xticks(RAS); a1.set_xticklabels(['10³', '10⁴', '10⁵', '10⁶'])
    a1.set_yticks([1, 2, 5, 10]); a1.set_yticklabels(['1', '2', '5', '10'])
    a1.set_xlim(5e2, 2e6); a1.set_ylim(0.9, 13)
    a1.set_xlabel('Rayleigh number, Ra'); a1.set_ylabel('Mean Nusselt number on the hot wall')
    a1.set_title('(a) Heat transfer and peak velocities vs. published benchmarks', fontsize=11.5 * k, color=INK,
                 loc='left')
    a1.legend(frameon=False, fontsize=9.5 * k, loc='upper left')
    pct = lambda a, b: f'{100 * (a / b - 1):+5.2f}'.replace('-0.00', ' 0.00').replace('+0.00', ' 0.00')
    rows = ['difference from de Vahl Davis (%)', '    Ra    Nu   u_max  v_max']
    for r in RAS:
        q = fin[r]
        rows.append(f'{r:6.0e}'.replace('e+0', 'e') + f' {pct(q["Nu_avg"], NU_DVD[r])} {pct(q["u_max"], U_DVD[r])}  {pct(q["v_max"], V_DVD[r])}')
    q = fin[1e6]
    rows += ['vs Le Quéré at Ra = 1e6:',
             f'        {100 * (q["Nu_avg"] / NU_LQ[1e6] - 1):+5.2f} {100 * (q["u_max"] / U_LQ[1e6] - 1):+5.2f}  '
             f'{100 * (q["v_max"] / V_LQ[1e6] - 1):+5.2f}']
    a1.text(0.98, 0.04, '\n'.join(rows), transform=a1.transAxes, ha='right', va='bottom', fontsize=8.5 * k,
            family='monospace', color=INK, bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=FRAME))
    a1.grid(alpha=0.25)

    # (b) grid convergence of Nu at Ra = 1e6
    grids = [g for g in (64, 128, 256, 384) if (g, 1e6) in runs]
    nu = [runs[(g, 1e6)]['Nu_avg'] for g in grids]
    a2.axhline(NU_LQ[1e6], color='#2f6fae', lw=1.2, label=f'Le Quéré (1991): {NU_LQ[1e6]:.3f}')
    a2.axhline(NU_DVD[1e6], color=INK, lw=1.2, ls=(0, (4, 2)), label=f'de Vahl Davis (1983): {NU_DVD[1e6]:.3f}')
    a2.plot(grids, nu, 'o-', color=COPPER, ms=7 * k, lw=1.4, label='simulation')
    for g, v in zip(grids, nu):
        a2.annotate(f'{v:.3f}', (g, v), xytext=(6, 6), textcoords='offset points', fontsize=9 * k, color=TEXT)
    a2.set_xscale('log', base=2); a2.minorticks_off(); a2.set_ylim(8.78, 9.13)
    a2.set_xticks(grids); a2.set_xticklabels([str(g) for g in grids])
    a2.set_xlabel('Grid cells per side'); a2.set_ylabel('Mean Nusselt number, Ra = 10⁶')
    a2.set_title('(b) Grid convergence at Ra = 10⁶', fontsize=11.5 * k, color=INK, loc='left')
    a2.legend(frameon=False, fontsize=9.5 * k, loc='upper right')
    a2.grid(alpha=0.25)

    fig.suptitle('Validation: side-heated cavity, air (Pr = 0.71)' if stacked else
                 'Validation: side-heated square cavity, air (Pr = 0.71), vs. published benchmark solutions',
                 fontfamily=FONT['serif'][0], fontsize=14 * (1.3 if stacked else 1), color=INK)
    fig.text(0.99, 0.01, f'{AUTHOR} | thermal lattice Boltzmann (D2Q9 + D2Q5), Python', ha='right', fontsize=9 * k,
             color=MUTED)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3 if stacked else None)
    fig.savefig(media / ('validation_comment.png' if stacked else 'validation.png')); plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault); brand.apply()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--media', required=True)
    ap.add_argument('--layout', default='chart', choices=sorted(LAYOUTS))
    ap.add_argument('--cover_only', action='store_true')
    ap.add_argument('--static', action='store_true', help='use the steady benchmark fields (no video)')
    ap.add_argument('--cover_t', type=float, default=None, help='cover frame time in L/U (default: last frame)')
    ap.add_argument('--cover_name', default='cover.png')
    ap.add_argument('--validation_only', action='store_true', help='--data must hold the bench_L*.json files')
    a = ap.parse_args()
    data, media = Path(a.data), Path(a.media)
    media.mkdir(parents=True, exist_ok=True)
    if a.validation_only:
        render_validation(data, media); render_validation(data, media, stacked=True); return
    render(data, media, a.layout, a.cover_only, a.static, a.cover_name, a.cover_t)


if __name__ == '__main__':
    main()
