"""
Render the 1080x1080 video, the cover frame and the validation plot from cylinder_lbm.py output.

Usage:
  python render.py --data run --media media
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from style import brand                               # noqa: E402
from style.brand import BG, INK, TEXT, MUTED, COPPER, FRAME, FONT, SIZE, AUTHOR, rich, spaced  # noqa: E402
from style.logo import draw_badge, draw_mark, logo_axes  # noqa: E402

CMAP = LinearSegmentedColormap.from_list(
    'vort', ['#053061', '#2166ac', '#92c5de', '#ffffff', '#f4a582', '#b2182b', '#67001f'])
CMAP.set_bad('#334155')  # cylinder
VLIM = 3.0


def crop(field, res):
    D = res['D']
    xc, yc = res['cylinder_center']
    x0, x1 = int(xc - 1.5 * D), int(xc + 18.5 * D)
    y0, y1 = int(yc - 3 * D), int(yc + 3 * D)
    return field[x0:x1, y0:y1].T.astype(np.float32)


def build_figure(res):
    brand.apply()
    L = 0.06
    W = 1 - 2 * L
    fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor=BG)

    # header
    fig.text(L, 0.940, spaced('FLOW PHYSICS IN SIMULATION') + '   ·   ' + spaced('Nº 01'), color=COPPER,
             fontsize=12.5, fontfamily=FONT['sans_semi'])
    rich(fig, L, 0.872, [('When does a wake start to ', FONT['display'], SIZE['title'], INK),
                         ('oscillate?', FONT['display_it'], SIZE['title'], COPPER)])
    fig.text(L, 0.832, 'Flow past a circular cylinder, at two Reynolds numbers', color=MUTED,
             fontsize=SIZE['sub'], fontfamily=FONT['serif_it'])
    bw = 0.072                                            # logo badge, top right
    bh = bw * 1.48 / 1.96
    draw_badge(logo_axes(fig, [1 - L - bw, 0.958 - bh, bw, bh], kind='badge'), lw=0.55)

    # panels
    axes = []
    labels = [('40', 'Steady wake: a symmetric pair of attached vortices'),
              ('100', 'Periodic shedding: the von Kármán vortex street')]
    for (re, sub), top in zip(labels, [0.770, 0.455]):
        rich(fig, L, top, [('Re', FONT['display_it'], SIZE['head'], INK),
                           (f' = {re}', FONT['display'], SIZE['head'], INK)])
        fig.text(L + 0.165, top + 0.002, sub, color=TEXT, fontsize=16, fontfamily=FONT['sans'])
        ax = fig.add_axes([L, top - 0.282, W, 0.267])
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(FRAME)
            s.set_linewidth(1.2)
        axes.append(ax)

    # colour bar (bottom left)
    cax = fig.add_axes([L, 0.100, 0.44, 0.020])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-VLIM, VLIM))
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[-3, -1.5, 0, 1.5, 3])
    cb.outline.set_edgecolor(FRAME)
    cb.ax.tick_params(labelsize=12, colors=TEXT, length=3)
    for lab in cb.ax.get_xticklabels():
        lab.set_fontfamily(FONT['sans'])
    rich(fig, L, 0.133, [('Vorticity  ', FONT['sans_semi'], 14, INK), ('ωD/U', FONT['serif_it'], 16, INK)])
    fig.text(L, 0.066, '← clockwise', color=MUTED, fontsize=11.5, va='top', fontfamily=FONT['sans'])
    fig.text(L + 0.44, 0.066, 'counter-clockwise →', color=MUTED, fontsize=11.5, va='top', ha='right',
             fontfamily=FONT['sans'])

    # Strouhal number (bottom right)
    st = res['Re100_strouhal']
    rich(fig, 1 - L, 0.133, [('Shedding frequency at ', FONT['sans_semi'], 14, INK),
                             ('Re', FONT['serif_it'], 15, INK), (' = 100', FONT['sans_semi'], 14, INK)], ha='right')
    rich(fig, 1 - L, 0.087, [('St', FONT['display_it'], 30, INK), (' = ', FONT['serif'], 26, INK),
                             (f'{st:.3f}', FONT['display'], 30, COPPER)], ha='right')
    fig.text(1 - L, 0.058, 'experiment: 0.164  (Williamson, 1989)', color=MUTED, fontsize=11.5, ha='right',
             fontfamily=FONT['serif_it'])

    # footer: logo mark + name
    fig.add_artist(plt.Line2D([L, 1 - L], [0.042, 0.042], color=FRAME, lw=1))
    mh = 0.024
    mw = mh * 1.66 / 1.18
    draw_mark(logo_axes(fig, [L, 0.010, mw, mh], kind='mark'), lw=0.45)
    fig.text(L + mw + 0.010, 0.016, AUTHOR, color=INK, fontsize=13, fontfamily=FONT['sans_semi'])
    fig.text(1 - L, 0.016, 'Lattice Boltzmann (D2Q9) simulation in Python', color=MUTED, fontsize=12,
             ha='right', fontfamily=FONT['sans'])
    return fig, axes


def render_video(data, res, media, cover_only=False):
    fr = np.load(data / 'frames.npz')
    re40, re100 = fr['re40'], fr['re100']
    fig, axes = build_figure(res)
    kw = dict(cmap=CMAP, vmin=-VLIM, vmax=VLIM, origin='lower', interpolation='bicubic', aspect='auto')
    cover_idx = len(re100) // 2
    ims = [axes[0].imshow(crop(re40[cover_idx], res), **kw), axes[1].imshow(crop(re100[cover_idx], res), **kw)]
    if cover_only:
        fig.canvas.draw()
        imageio.imwrite(media / 'cover.png', np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
        return
    w = imageio.get_writer(media / 'vortex_street.mp4', fps=30, codec='libx264', quality=8,
                           pixelformat='yuv420p', macro_block_size=8)
    for k in range(len(re100)):
        ims[0].set_data(crop(re40[k], res))
        ims[1].set_data(crop(re100[k], res))
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        w.append_data(img)
        if k == cover_idx:
            imageio.imwrite(media / 'cover.png', img)
    w.close()
    plt.close(fig)
    print('frames:', len(re100))


def render_validation(data, res, media):
    brand.apply()
    probe = np.load(data / 'probe.npy')
    t = np.arange(len(probe)) / res['Re100_period_steps']
    fig, ax = plt.subplots(figsize=(10.8, 5.4), dpi=100, facecolor='white')
    ax.plot(t, probe / res['U0'], color=INK, lw=1.6)
    ax.set_xlabel('time / shedding period')
    ax.set_ylabel('transverse velocity  v / U  (3D behind cylinder)')
    ax.set_title(f"Re = 100: periodic shedding, St = {res['Re100_strouhal']:.3f}   "
                 f"(Re = 40: recirculation length L/D = {res['Re40_recirculation_length_over_D']:.2f})",
                 fontfamily=FONT['serif'])
    ax.grid(alpha=0.3)
    fig.text(0.99, 0.01, f'{AUTHOR} | LBM D2Q9, Python', ha='right', fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(media / 'validation.png')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='folder written by cylinder_lbm.py')
    ap.add_argument('--media', required=True, help='output folder for video and images')
    ap.add_argument('--cover_only', action='store_true', help='render only the cover frame')
    a = ap.parse_args()
    data, media = Path(a.data), Path(a.media)
    media.mkdir(parents=True, exist_ok=True)
    res = json.loads((data / 'results.json').read_text())
    render_video(data, res, media, cover_only=a.cover_only)
    if not a.cover_only:
        render_validation(data, res, media)


if __name__ == '__main__':
    main()
