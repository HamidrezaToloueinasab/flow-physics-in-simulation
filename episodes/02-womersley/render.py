"""
Render the 1080x1080 video, cover frame and validation figure for episode 02 (Womersley flow).

Usage:  python render.py --data run --media media [--cover_only] [--vessel tissue|navy3d|cutaway]
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
from matplotlib.patches import FancyArrowPatch        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (for style/)
from style import brand                               # noqa: E402
from style.brand import BG, INK, TEXT, MUTED, COPPER, FRAME, FONT, SIZE, AUTHOR, rich, spaced  # noqa: E402
from style.logo import draw_badge, draw_mark, logo_axes  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from womersley import exact_u                         # noqa: E402

FWD, REV = COPPER, '#2f6fae'                          # forward / reversed flow
CASES = [  # alpha, label, description, curve colour
    (1.0, '1', 'small vessels: quasi-steady', '#8a9bb3'),
    (5.0, '5', '≈ common carotid artery', '#2f6fae'),
    (15.0, '15', '≈ ascending aorta', COPPER),
]
L = 0.06
WT = 0.17                     # wall thickness / lumen radius (drawing); arterial walls are roughly 0.1–0.2 R
XL = 1.32                     # panel half-width in profile units
VESSEL = 'tissue'             # 'tissue' | 'navy3d' | 'cutaway'
COVER_NAME = 'cover.png'

# wall layers (inner -> outer) and lumen tint for each style
STYLES = {
    'tissue': dict(lumen=('#fdf4f2', '#f6dcd8'), layers=[('#e9b2ad', 0.18), ('#d98f8a', 0.55), ('#c47f72', 0.27)],
                   edge='#8c4a45', opening='#b86a68'),
    'navy3d': dict(lumen=('#f7f9fc', '#e3e9f1'), layers=[('#9fb0c6', 0.15), ('#41597a', 0.70), ('#2a3f5c', 0.15)],
                   edge=INK, opening='#5d7393'),
    'cutaway': dict(lumen=('#ffffff', '#f3f1ef'), layers=[('#f0c9c2', 0.22), ('#dc9b92', 0.48), ('#efe3d0', 0.30)],
                    edge='#6b4a44', opening=None),
}


def _vgrad(ax, x0, x1, y0, y1, c0, c1, z, shade=True):
    """Vertical colour gradient from c0 (at y0) to c1 (at y1); optional cylindrical shading across it."""
    from matplotlib.colors import to_rgb
    n = 64
    t = np.linspace(0, 1, n)[:, None]
    rgb = np.array(to_rgb(c0)) * (1 - t) + np.array(to_rgb(c1)) * t
    if shade:
        rgb = rgb * (0.94 + 0.06 * np.sin(np.pi * t))
    img = np.repeat(rgb[:, None, :], 2, axis=1)
    ax.imshow(img, extent=[x0, x1, y0, y1], origin='lower', aspect='auto', zorder=z, interpolation='bicubic')


def draw_vessel(ax, style):
    """Straight artery segment: shaded lumen, layered wall (intima, media, adventitia), 3D end opening."""
    from matplotlib.patches import Ellipse
    S = STYLES[style]
    # lumen: lighter at the centre line, darker towards the wall (cylindrical shading)
    _vgrad(ax, -XL, XL, 0, 1, S['lumen'][0], S['lumen'][1], z=0, shade=False)
    _vgrad(ax, -XL, XL, -1, 0, S['lumen'][1], S['lumen'][0], z=0, shade=False)
    for s in (1, -1):
        y = 1.0
        for k, (col, frac) in enumerate(S['layers']):
            h = WT * frac
            lo, hi = (y, y + h) if s > 0 else (-(y + h), -y)
            dark = tuple(0.85 * np.array(matplotlib.colors.to_rgb(col)))
            _vgrad(ax, -XL, XL, lo, hi, *( (col, dark) if s > 0 else (dark, col) ), z=1)
            y += h
        ax.plot([-XL, XL], [s * 1.0] * 2, color=S['edge'], lw=1.6, zorder=2)            # endothelium
        ax.plot([-XL, XL], [s * (1 + WT)] * 2, color=S['edge'], lw=1.0, alpha=0.6, zorder=2)
    if style == 'cutaway':                                       # layer boundaries visible in section
        for s in (1, -1):
            y = 1.0
            for col, frac in S['layers'][:-1]:
                y += WT * frac
                ax.plot([-XL, XL], [s * y] * 2, color=S['edge'], lw=0.6, alpha=0.5, zorder=2)
    if S['opening']:                                             # downstream end seen in perspective
        ax.add_patch(Ellipse((XL, 0), 0.18, 2 * (1 + WT), fc=S['layers'][1][0], ec=S['edge'], lw=1.0, zorder=2))
        ax.add_patch(Ellipse((XL, 0), 0.14, 2.0, fc=S['opening'], ec=S['edge'], lw=0.8, zorder=2))


def build_figure(res):
    brand.apply()
    fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor=BG)
    fig.text(L, 0.940, spaced('FLOW PHYSICS IN SIMULATION') + '   ·   ' + spaced('Nº 02'), color=COPPER,
             fontsize=12.5, fontfamily=FONT['sans_semi'])
    rich(fig, L, 0.872, [('Why isn’t blood flow ', FONT['display'], SIZE['title'], INK),
                         ('parabolic?', FONT['display_it'], SIZE['title'], COPPER)])
    fig.text(L, 0.832, 'Oscillating flow in a tube, at three Womersley numbers', color=MUTED,
             fontsize=SIZE['sub'], fontfamily=FONT['serif_it'])
    bw = 0.072; bh = bw * 1.48 / 1.96
    draw_badge(logo_axes(fig, [1 - L - bw, 0.958 - bh, bw, bh], kind='badge'), lw=0.55)

    # three profile panels
    gap = 0.035
    pw = (1 - 2 * L - 2 * gap) / 3
    panels = []
    for i, (alpha, lab, desc, col) in enumerate(CASES):
        x0 = L + i * (pw + gap)
        rich(fig, x0, 0.765, [('α', FONT['display_it'], SIZE['head'], INK),
                              (f' = {lab}', FONT['display'], SIZE['head'], INK)])
        fig.text(x0, 0.738, desc, color=TEXT, fontsize=13, fontfamily=FONT['sans'])
        ax = fig.add_axes([x0, 0.385, pw, 0.335])
        ax.set_xlim(-XL, XL + 0.14); ax.set_ylim(-1.24, 1.24); ax.axis('off')
        draw_vessel(ax, VESSEL)
        ax.plot([0, 0], [-1, 1], color='#9aa5b1', lw=1.0, ls=(0, (3, 3)), zorder=2)   # station line
        panels.append(ax)

    # colour legend
    from matplotlib.patches import Rectangle
    lax = fig.add_axes([L, 0.338, 1 - 2 * L, 0.03]); lax.axis('off'); lax.set_xlim(0, 1); lax.set_ylim(0, 1)
    for x, col, txt in ((0.0, FWD, 'Forward flow'), (0.17, REV, 'Reversed flow')):
        lax.add_patch(Rectangle((x, 0.2), 0.022, 0.6, fc=col, alpha=0.45, ec=col, lw=1.2))
        lax.text(x + 0.032, 0.5, txt, va='center', fontsize=12, color=INK, fontfamily=FONT['sans_med'])
    lax.text(1.0, 0.5, 'Velocity profile across the vessel · each panel scaled to its own peak', va='center',
             ha='right', fontsize=11, color=MUTED, fontfamily=FONT['sans'])

    # time traces: driving pressure gradient and flow rate over one cycle
    fig.text(L, 0.300, 'Pressure gradient and flow rate over one cycle', color=INK, fontsize=13,
             fontfamily=FONT['sans_semi'])
    # legend row: each entry = line sample + label, laid out left to right by measured text width
    gax = fig.add_axes([L, 0.262, 0.62, 0.024]); gax.axis('off'); gax.set_xlim(0, 1); gax.set_ylim(0, 1)
    rend = fig.canvas.get_renderer()
    entries = [(INK, (0, (4, 2.5)), 'Driving pressure gradient −∂p/∂x', 1.6)]
    entries += [(col, '-', f'Flow rate, α = {lab}', 2.4) for alpha, lab, desc, col in CASES]
    x = 0.0
    for col, ls, txt, lw in entries:
        gax.plot([x, x + 0.036], [0.5, 0.5], color=col, lw=lw, ls=ls, solid_capstyle='butt')
        t = gax.text(x + 0.048, 0.5, txt, va='center', fontsize=10, color=INK, fontfamily=FONT['sans'])
        x += 0.048 + t.get_window_extent(renderer=rend).width / gax.get_window_extent(renderer=rend).width + 0.022
    tax = fig.add_axes([L + 0.03, 0.098, 0.575, 0.15])
    tax.set_xlim(0, 360); tax.set_ylim(-1.25, 1.25)
    for s in tax.spines.values():
        s.set_visible(False)
    tax.axhline(0, color=FRAME, lw=1)
    tax.set_xticks([0, 90, 180, 270, 360])
    tax.set_xticklabels(['0°', '90°', '180°', '270°', '360°'], fontsize=10.5, color=TEXT)
    tax.set_yticks([-1, 0, 1]); tax.set_yticklabels(['−1', '0', '1'], fontsize=10, color=TEXT)
    tax.tick_params(length=0)
    tax.set_xlabel('Phase within the cycle, ωt', fontsize=10.5, color=MUTED, labelpad=3)
    tax.set_ylabel('Scaled to own\namplitude', fontsize=10, color=MUTED, labelpad=4)
    ph = np.linspace(0, 360, 361)
    tax.plot(ph, np.cos(np.radians(ph)), color=INK, lw=1.6, ls=(0, (5, 3)))
    lags = {r['alpha']: r['flow_phase_lag_deg']['exact'] for r in res['results']}
    for alpha, lab, desc, col in CASES:
        tax.plot(ph, np.cos(np.radians(ph - lags[alpha])), color=col, lw=2.2)
    # lag chart (right): one aligned row per case, bar length = phase lag on a 0-90 degree scale
    bx0, bw_ = 0.715, 1 - L - 0.715
    fig.text(bx0, 0.300, 'Flow lags the pressure by', color=INK, fontsize=13, fontfamily=FONT['sans_semi'])
    bax = fig.add_axes([bx0 + 0.085, 0.105, bw_ - 0.085, 0.15])
    bax.set_xlim(0, 90); bax.set_ylim(-0.6, 2.6); bax.invert_yaxis()
    for s in bax.spines.values():
        s.set_visible(False)
    bax.set_yticks([]); bax.set_xticks([0, 45, 90]); bax.set_xticklabels(['0°', '45°', '90°'], fontsize=10.5, color=TEXT)
    bax.tick_params(length=0)
    for g in (45, 90):
        bax.axvline(g, color=FRAME, lw=1, zorder=0)
    for k, (alpha, lab, desc, col) in enumerate(CASES):
        bax.barh(k, 90, height=0.46, color='#eef1f5', zorder=1)                      # track
        bax.barh(k, lags[alpha], height=0.46, color=col, zorder=2)
        lab_tr = matplotlib.transforms.blended_transform_factory(fig.transFigure, bax.transData)
        bax.text(bx0, k, 'α', ha='left', va='center', fontsize=14, color=col,               # aligned columns:
                 fontfamily=FONT['serif_it'], transform=lab_tr, clip_on=False)            # α | = | value
        bax.text(bx0 + 0.022, k, '=', ha='left', va='center', fontsize=13, color=col,
                 fontfamily=FONT['sans_semi'], transform=lab_tr, clip_on=False)
        bax.text(bx0 + 0.045, k, lab, ha='left', va='center', fontsize=13, color=col,
                 fontfamily=FONT['sans_semi'], transform=lab_tr, clip_on=False)
        inside = lags[alpha] > 30
        bax.text(lags[alpha] - 2 if inside else lags[alpha] + 2, k, f'{lags[alpha]:.0f}°',
                 ha='right' if inside else 'left', va='center', fontsize=13,
                 color='white' if inside else col, fontfamily=FONT['display'], zorder=3)
    cursor = tax.axvline(0, color=INK, lw=1.0, alpha=0.6)
    cursor.now = tax.text(0, -1.22, '', fontsize=1)      # placeholder (no label; the line itself marks "now")

    # footer
    fig.add_artist(plt.Line2D([L, 1 - L], [0.042, 0.042], color=FRAME, lw=1))
    mh = 0.024; mw = mh * 1.66 / 1.18
    draw_mark(logo_axes(fig, [L, 0.010, mw, mh], kind='mark'), lw=0.45)
    fig.text(L + mw + 0.010, 0.016, AUTHOR, color=INK, fontsize=13, fontfamily=FONT['sans_semi'])
    fig.text(1 - L, 0.016, 'Unsteady Navier–Stokes, finite volumes in Python', color=MUTED, fontsize=12,
             ha='right', fontfamily=FONT['sans'])
    return fig, panels, tax, cursor


def draw_profiles(panels, r, profs, k):
    """Draw the k-th phase of each case; profs[i] has shape (samples, nr) and is scaled by its peak."""
    arts = []
    rr = np.concatenate([-r[::-1], r])
    for ax, P in zip(panels, profs):
        u = P[k]
        uu = np.concatenate([u[::-1], u])
        arts += ax.fill_betweenx(rr, 0, uu, where=uu >= 0, color=FWD, alpha=0.45, lw=0, interpolate=True, zorder=3),
        arts += ax.fill_betweenx(rr, 0, uu, where=uu < 0, color=REV, alpha=0.45, lw=0, interpolate=True, zorder=3),
        arts += ax.plot(uu, rr, color=INK, lw=2.0, zorder=4)
        for y in np.linspace(-0.8, 0.8, 7):                             # velocity arrows
            v = float(np.interp(abs(y), r, u))
            if abs(v) > 0.06:
                a = FancyArrowPatch((0, y), (v, y), arrowstyle='-|>', mutation_scale=11, lw=1.3,
                                    color=FWD if v > 0 else REV, shrinkA=0, shrinkB=0, zorder=4)
                ax.add_patch(a); arts.append(a)
    return arts


def load(data):
    d = np.load(data / 'profiles.npz')
    res = json.loads((data / 'results.json').read_text())
    r, t = d['r'], d['t']
    order = np.argsort(np.mod(t, 2 * np.pi))                            # phase 0..2pi
    profs = []
    for alpha, *_ in CASES:
        P = d[f'u_{alpha:g}'][order]
        profs.append(P / np.abs(P).max())
    phase = np.degrees(np.mod(t, 2 * np.pi))[order]
    return res, r, phase, profs


def render_video(data, media, cover_only=False):
    res, r, phase, profs = load(data)
    fig, panels, tax, cursor = build_figure(res)
    cover_k = int(np.argmin(np.abs(phase - 171)))       # core forward while near-wall flow has reversed (alpha 5, 15)
    if cover_only:
        draw_profiles(panels, r, profs, cover_k); cursor.set_xdata([phase[cover_k]] * 2); cursor.now.set_x(phase[cover_k])
        fig.savefig(media / COVER_NAME, dpi=100, facecolor=BG); plt.close(fig); return
    w = imageio.get_writer(media / 'womersley.mp4', fps=30, codec='libx264', quality=8,
                           pixelformat='yuv420p', macro_block_size=8)
    n = len(phase)
    for rep in range(3):                                # three cycles, 4 s each
        for k in range(n):
            arts = draw_profiles(panels, r, profs, k)
            cursor.set_xdata([phase[k]] * 2); cursor.now.set_x(phase[k])
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            w.append_data(img)
            if rep == 0 and k == cover_k:
                imageio.imwrite(media / 'cover.png', img)
            for a in arts:
                a.remove()
    w.close(); plt.close(fig)


def render_validation(data, media):
    res, r, phase, profs = load(data)
    brand.apply()
    d = np.load(data / 'profiles.npz')
    t = d['t']
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.6), dpi=100, facecolor='white', sharey=True)
    for ax, (alpha, lab, desc, col) in zip(axs, CASES):
        P = d[f'u_{alpha:g}']
        for k in range(0, len(t), 15):
            ax.plot(exact_u(alpha, r, t[k]), r, color=INK, lw=1.2)
            ax.plot(P[k][::25], r[::25], 'o', color=col, ms=3.5)
        ax.set_title(f'α = {lab}', fontfamily=FONT['display'], fontsize=15, color=INK)
        ax.set_xlabel('u / (G R² / 4μ)')
        ax.grid(alpha=0.25)
        e = [x for x in res['results'] if x['alpha'] == alpha][0]['max_profile_error_rel']
        ax.text(0.5, 0.04, f'max. error {100 * e:.4f} % of peak', transform=ax.transAxes, fontsize=10, color=INK, ha='center', bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=FRAME))
    axs[0].set_ylabel('r / R')
    fig.suptitle('Simulation (dots) vs. Womersley exact solution (lines), 8 phases of the cycle',
                 fontfamily=FONT['serif'][0], fontsize=14, color=INK)
    fig.text(0.99, 0.01, f'{AUTHOR} | finite volumes + Crank–Nicolson, Python', ha='right', fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(media / 'validation.png'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--media', required=True)
    ap.add_argument('--cover_only', action='store_true')
    ap.add_argument('--vessel', default=VESSEL, choices=sorted(STYLES))
    ap.add_argument('--cover_name', default='cover.png')
    a = ap.parse_args()
    globals()['VESSEL'] = a.vessel
    globals()['COVER_NAME'] = a.cover_name
    data, media = Path(a.data), Path(a.media)
    media.mkdir(parents=True, exist_ok=True)
    render_video(data, media, cover_only=a.cover_only)
    if not a.cover_only:
        render_validation(data, media)


if __name__ == '__main__':
    main()
