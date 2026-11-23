"""
Render the 1080x1080 video, cover frame and validation figure for episode 03 (Rayleigh-Benard convection).

Usage:  python render.py --data run --media ../media [--cover_only] [--theme light|dark] [--cmap brand|ember]
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
from matplotlib.patches import Rectangle              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from style import brand                               # noqa: E402
from style.brand import INK, TEXT, MUTED, COPPER, FRAME, FONT, SIZE, AUTHOR, rich, spaced  # noqa: E402
from style.logo import draw_badge, draw_mark, logo_axes  # noqa: E402

CASES = [  # key in movie.npz, Ra label, description, accent colour
    ('1500', '1,500', 'below onset: conduction only', '#8a9bb3'),
    ('10000', '10,000', 'steady convection rolls', '#2f6fae'),
    ('1e+06', '1,000,000', 'rising and sinking plumes', COPPER),
]
CMAPS = {
    'brand': ['#0b1f3a', '#2f5f96', '#c9d6e6', '#f7f3ee', '#ecc9a2', '#b87333', '#6e3a12'],
    'ember': ['#0b1f3a', '#1f3f6b', '#5b6f8e', '#b6a596', '#d9a066', '#b87333', '#7a3b10'],
}
THEMES = {
    'light': dict(bg='#ffffff', ink=INK, text=TEXT, muted=MUTED, frame=FRAME, track='#eef1f5',
                  cold='#1f3f6b', hot='#9a4f18'),
    'dark': dict(bg='#0b1f3a', ink='#f4f1ec', text='#d5dbe4', muted='#9fb0c6', frame='#2a3f5c', track='#16304f',
                 cold='#4a6fa5', hot='#d9a066'),
}
L = 0.06
NU_MAX = 12.0


def build_figure(theme, cmap, aspect):
    th = THEMES[theme]
    brand.apply()
    fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor=th['bg'])
    fig.text(L, 0.940, spaced('FLOW PHYSICS IN SIMULATION') + '   ·   ' + spaced('Nº 03'), color=COPPER,
             fontsize=12.5, fontfamily=FONT['sans_semi'])
    rich(fig, L, 0.872, [('When does heated air ', FONT['display'], SIZE['title'], th['ink']),
                         ('start to move?', FONT['display_it'], SIZE['title'], COPPER)])
    fig.text(L, 0.832, 'A layer of air heated from below, at three Rayleigh numbers', color=th['muted'],
             fontsize=SIZE['sub'], fontfamily=FONT['serif_it'])
    if theme == 'light':
        bw = 0.072; bh = bw * 1.48 / 1.96
        draw_badge(logo_axes(fig, [1 - L - bw, 0.958 - bh, bw, bh], kind='badge'), lw=0.55)
    else:
        mh = 0.052; mw = mh * 1.66 / 1.18
        draw_mark(logo_axes(fig, [1 - L - mw, 0.958 - mh, mw, mh], kind='mark'), lw=0.55, light=True)

    # ---- geometry: field panels on the left, aligned Nu bars on the right ----
    ph = 0.145                                   # field height (figure fraction; square pixels)
    pw = ph * aspect                             # field width
    plate = 0.013
    x_bar = L + pw + 0.045
    bar_w = 1 - L - x_bar
    fig.text(x_bar, 0.785, 'Heat transfer, Nu', color=th['ink'], fontsize=13.5, fontfamily=FONT['sans_semi'])
    fig.text(x_bar, 0.772, 'heat through the plates,\nrelative to conduction alone', color=th['muted'],
             fontsize=10, va='top', linespacing=1.3)
    row = 0.236
    tops = [0.785 - i * row for i in range(3)]
    images, bars, nu_txt = [], [], []
    for n, ((key, lab, desc, col), top) in enumerate(zip(CASES, tops)):
        rich(fig, L, top, [('Ra', FONT['display_it'], SIZE['head'], th['ink']),
                           (f' = {lab}', FONT['display'], SIZE['head'], th['ink']),
                           (f'    {desc}', FONT['sans'], 13, th['text'])])
        y_cold = top - 0.018 - plate
        y_field = y_cold - ph
        y_hot = y_field - plate
        ax = fig.add_axes([L, y_field, pw, ph])
        ax.axis('off')
        im = ax.imshow(np.zeros((8, 32)), origin='lower', cmap=cmap, vmin=0, vmax=1, aspect='auto',
                       interpolation='bicubic', extent=[0, aspect, 0, 1])
        images.append(im)
        for y0, c, word in ((y_hot, th['hot'], 'HOT PLATE'), (y_cold, th['cold'], 'COLD PLATE')):
            fig.add_artist(Rectangle((L, y0), pw, plate, transform=fig.transFigure, fc=c, ec='none'))
            if n == 0:
                fig.text(L + 0.010, y0 + plate / 2, spaced(word), fontsize=7.5, color='#ffffff', va='center',
                         fontfamily=FONT['sans_semi'], alpha=0.9)
        # Nu bar on a common 0..NU_MAX scale; the dashed line marks Nu = 1 (conduction only)
        yc = y_field + 0.5 * ph
        bax = fig.add_axes([x_bar, yc - 0.045, bar_w, 0.075])
        bax.set_xlim(0, NU_MAX); bax.set_ylim(0, 1); bax.axis('off')
        bax.add_patch(Rectangle((0, 0.30), NU_MAX, 0.26, fc=th['track'], ec='none'))
        b = Rectangle((0, 0.30), 1, 0.26, fc=col, ec='none')
        bax.add_patch(b); bars.append(b)
        bax.plot([1, 1], [0.22, 0.64], color=th['ink'], lw=1.0, ls=(0, (2, 2)), alpha=0.7)
        for v in (0, 1, 4, 8, 12):
            bax.text(v, 0.12, f'{v}', ha='center', va='center', fontsize=9.5, color=th['muted'])
        nu_txt.append(bax.text(0, 0.72, '', ha='left', va='bottom', fontsize=24, color=col,
                               fontfamily=FONT['display']))

    # ---- legend row: temperature colour bar, time, fluid ----
    y_leg = 0.072
    cax = fig.add_axes([L + 0.120, y_leg, 0.24, 0.016])
    cax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap, aspect='auto', extent=[0, 1, 0, 1])
    cax.set_xticks([]); cax.set_yticks([])
    for s in cax.spines.values():
        s.set_visible(False)
    fig.text(L, y_leg + 0.008, 'Temperature', fontsize=12, color=th['ink'], va='center', fontfamily=FONT['sans_med'])
    fig.text(L + 0.120, y_leg - 0.014, 'cold plate', fontsize=10, color=th['text'], va='center')
    fig.text(L + 0.360, y_leg - 0.014, 'hot plate', fontsize=10, color=th['text'], va='center', ha='right')
    t_txt = fig.text(L + pw, y_leg + 0.008, '', fontsize=12, color=th['ink'], va='center', ha='right',
                     fontfamily=FONT['sans_med'])
    fig.text(x_bar, y_leg + 0.008, 'Fluid: air, Pr = 0.71', fontsize=11, color=th['muted'], va='center')

    # footer
    fig.add_artist(plt.Line2D([L, 1 - L], [0.042, 0.042], color=th['frame'], lw=1))
    mh = 0.024; mw = mh * 1.66 / 1.18
    draw_mark(logo_axes(fig, [L, 0.010, mw, mh], kind='mark'), lw=0.45, light=(theme == 'dark'))
    fig.text(L + mw + 0.010, 0.016, AUTHOR, color=th['ink'], fontsize=13, fontfamily=FONT['sans_semi'])
    fig.text(1 - L, 0.016, 'Thermal lattice Boltzmann method in Python', color=th['muted'], fontsize=12,
             ha='right', fontfamily=FONT['sans'])
    return fig, images, bars, nu_txt, t_txt


def make_cmap(name):
    return LinearSegmentedColormap.from_list(name, CMAPS[name], N=512)


def load(data):
    d = np.load(data / 'movie.npz')
    fields = [d[f'T_{k}'] for k, *_ in CASES]
    nus = [0.5 * (d[f'Nu_{k}'][:, 1] + d[f'Nu_{k}'][:, 2]) for k, *_ in CASES]   # through the plates
    res = json.loads((data / 'movie.json').read_text())
    return d['times'], fields, nus, res


def set_frame(k, times, fields, nus, images, bars, nu_txt, t_txt):
    for im, F, nu, b, tx in zip(images, fields, nus, bars, nu_txt):
        im.set_data(F[k].astype(np.float32).T)
        b.set_width(min(nu[k], NU_MAX))
        tx.set_text(f'Nu = {nu[k]:.1f}')
    t_txt.set_text(f'Time: {times[k]:.0f} H/U')


def render(data, media, theme, cmap_name, cover_only, cover_t, cover_name):
    times, fields, nus, res = load(data)
    aspect = res['aspect']
    fig, images, bars, nu_txt, t_txt = build_figure(theme, make_cmap(cmap_name), aspect)
    kc = int(np.argmin(np.abs(times - cover_t)))
    if cover_only:
        set_frame(kc, times, fields, nus, images, bars, nu_txt, t_txt)
        fig.savefig(media / cover_name, dpi=100, facecolor=fig.get_facecolor()); plt.close(fig); return
    w = imageio.get_writer(media / 'rayleigh_benard.mp4', fps=30, codec='libx264', quality=8,
                           pixelformat='yuv420p', macro_block_size=8)
    n = len(times)
    for k in list(range(n)) + [n - 1] * 30:            # play, then hold the last frame for 1 s
        set_frame(k, times, fields, nus, images, bars, nu_txt, t_txt)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        w.append_data(img)
        if k == kc:
            imageio.imwrite(media / cover_name, img)
    w.close(); plt.close(fig)


# Published reference values (rigid plates, 2D rolls of wavenumber 3.117, Pr = 0.71)
RAC_THEORY = 1707.762                          # critical Rayleigh number, linear stability theory
NU_CB = {2000: 1.212, 3000: 1.663, 5000: 2.116, 10000: 2.661}   # Clever & Busse (1974), Table 1


def render_validation(data, media, stacked=False):
    """stacked=True: 1080x1350 portrait version with larger text, for a LinkedIn comment."""
    brand.apply()
    k = 1.45 if stacked else 1.0
    plt.rcParams.update({'font.size': 10 * k, 'axes.labelsize': 10 * k, 'xtick.labelsize': 9.5 * k,
                         'ytick.labelsize': 9.5 * k})
    on = json.loads((data / 'onset_H96.json').read_text())
    grids = [json.loads((data / f'onset_H{h}.json').read_text()) for h in (32, 48, 64, 96)]
    rolls = json.loads((data / 'rolls_H128.json').read_text())
    if stacked:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.8, 13.5), dpi=100, facecolor='white')
    else:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=100, facecolor='white')

    # (a) growth rate of the convective mode vs Ra, extrapolated to zero
    ra = np.array([r['Ra'] for r in on['runs']]); s = np.array([r['growth_rate'] for r in on['runs']])
    xx = np.linspace(1630, 1820, 50)
    a1.axhline(0, color=FRAME, lw=1)
    a1.plot(xx, on['slope_per_Ra'] * (xx - on['Ra_c']), color=INK, lw=1.2, ls=(0, (4, 2)),
            label='straight-line fit')
    a1.plot(ra, s, 'o', color=COPPER, ms=7 * k, label='simulation', zorder=3)
    a1.axvline(RAC_THEORY, color='#2f6fae', lw=1.2, label=f'linear stability theory: Ra$_c$ = {RAC_THEORY:.1f}')
    a1.set_xlabel('Rayleigh number, Ra'); a1.set_ylabel('Growth rate of the first roll mode  [κ/H²]')
    a1.set_title('(a) Onset: small disturbances decay below Ra$_c$, grow above it',
                 fontsize=11.5 * k, color=INK, loc='left')
    a1.legend(frameon=False, fontsize=9.5 * k, loc='upper left')
    txt = 'Ra_c from simulation:\n' + '\n'.join(
        f'{g["H"]:>2d} cells across H:  {g["Ra_c"]:.1f}  ({g["error_pct"]:+.3f} %)' for g in grids)
    a1.text(0.98, 0.04, txt, transform=a1.transAxes, ha='right', va='bottom', fontsize=9 * k, color=INK,
            family='monospace', bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=FRAME))
    a1.grid(alpha=0.25)

    # (b) Nusselt number of steady rolls vs published values
    rr = np.array([r['Ra'] for r in rolls['runs']]); nu = np.array([r['Nu'] for r in rolls['runs']])
    rc = np.array(sorted(NU_CB)); nc = np.array([NU_CB[k] for k in rc])
    a2.plot(rc, nc, 's', mfc='none', mec=INK, ms=9 * k, mew=1.4, label='Clever & Busse (1974)')
    a2.plot(rr, nu, 'o', color=COPPER, ms=6 * k, label='simulation (128 cells across H)')
    for x, y, yc in zip(rr, nu, nc):
        a2.annotate(f'{100 * (y - yc) / yc:+.2f} %', (x, y), xytext=(8, -14), textcoords='offset points',
                    fontsize=9 * k, color=TEXT)
    a2.set_xlim(1000, 11000); a2.set_ylim(1.05, 2.85); a2.set_xticks([2000, 4000, 6000, 8000, 10000])
    a2.set_xticklabels(['2,000', '4,000', '6,000', '8,000', '10,000'])
    a2.text(0.98, 0.04, 'labels: difference from Clever & Busse', transform=a2.transAxes, ha='right', fontsize=9 * k, color=MUTED)
    a2.set_xlabel('Rayleigh number, Ra'); a2.set_ylabel('Nusselt number, Nu')
    a2.set_title('(b) Heat transfer of steady rolls (Pr = 0.71, wavenumber 3.117)', fontsize=11.5 * k, color=INK,
                 loc='left')
    a2.legend(frameon=False, fontsize=9.5 * k, loc='upper left')
    a2.grid(alpha=0.25)
    fig.suptitle('Validation: thermal lattice Boltzmann solver vs. theory and published results'
                 if not stacked else 'Validation against theory and published results',
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
    ap.add_argument('--cover_only', action='store_true')
    ap.add_argument('--validation_only', action='store_true')
    ap.add_argument('--theme', default='light', choices=sorted(THEMES))
    ap.add_argument('--cmap', default='brand', choices=sorted(CMAPS))
    ap.add_argument('--cover_t', type=float, default=30.0, help='cover frame time in H/U')
    ap.add_argument('--cover_name', default='cover.png')
    a = ap.parse_args()
    data, media = Path(a.data), Path(a.media)
    media.mkdir(parents=True, exist_ok=True)
    if a.validation_only:
        render_validation(data, media); render_validation(data, media, stacked=True); return
    render(data, media, a.theme, a.cmap, a.cover_only, a.cover_t, a.cover_name)
    if not a.cover_only:
        render_validation(data, media)
        render_validation(data, media, stacked=True)


if __name__ == '__main__':
    main()
