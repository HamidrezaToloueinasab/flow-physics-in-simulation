"""
Rayleigh-Benard convection in a fluid layer heated from below (2D, Boussinesq).

Method: double-distribution lattice Boltzmann.
  flow         D2Q9, two-relaxation-time (TRT) collision, Guo forcing for buoyancy
  temperature  D2Q5, TRT collision (the antisymmetric rate sets the diffusivity)
  walls        no-slip, isothermal plates (half-way bounce-back / anti-bounce-back),
               hot plate at the bottom, cold plate at the top; periodic in x.

Non-dimensional groups: Ra = g beta dT H^3 / (nu kappa), Pr = nu / kappa.
Lattice parameters follow from the free-fall velocity U_ff = sqrt(g beta dT H) (lattice units):
  nu = U_ff H sqrt(Pr / Ra),  kappa = nu / Pr.

Nusselt number: Nu = 1 + <u_y T> H / (kappa dT)   (volume average; exact identity for the
Boussinesq equations in a periodic layer), cross-checked with the wall temperature gradient.

Subcommands
  onset     growth/decay rates of the first convective mode near Ra_c -> critical Rayleigh number
  rolls     steady Nusselt numbers of 2D rolls (wavenumber 3.117) for comparison with Clever & Busse
  movie     three layers (below onset, steady rolls, unsteady plumes) started from rest, side by side
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

# ---------------- lattices ----------------
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
WF = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
OPF = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)

DX = np.array([0, 1, 0, -1, 0], dtype=np.int64)
DY = np.array([0, 0, 1, 0, -1], dtype=np.int64)
WG = np.array([1/3, 1/6, 1/6, 1/6, 1/6])
OPG = np.array([0, 3, 4, 1, 2], dtype=np.int64)

KC = 3.117            # critical wavenumber (rigid-rigid), times H
RAC_THEORY = 1707.762
LAMBDA = 0.25          # TRT "magic" parameter


@njit(parallel=True, cache=True, fastmath=True)
def step(f, fn, g, gn, wp, wm, gwp, gwm, gb, Th, Tc, T0, rho_o, ux_o, uy_o, T_o):
    """Fused pull-stream + TRT collide for both populations. f, g hold post-collision values."""
    nx = f.shape[1]
    ny = f.shape[2]
    for x in prange(nx):
        fl = np.empty(9)
        gl = np.empty(5)
        for y in range(ny):
            # ---- stream f (bounce-back at the plates) ----
            for i in range(9):
                ys = y - CY[i]
                if ys < 0 or ys >= ny:
                    fl[i] = f[OPF[i], x, y]
                else:
                    fl[i] = f[i, (x - CX[i]) % nx, ys]
            # ---- stream g (anti-bounce-back: Dirichlet temperature at the plates) ----
            for i in range(5):
                ys = y - DY[i]
                if ys < 0:
                    gl[i] = -g[OPG[i], x, y] + 2.0 * WG[i] * Th
                elif ys >= ny:
                    gl[i] = -g[OPG[i], x, y] + 2.0 * WG[i] * Tc
                else:
                    gl[i] = g[i, (x - DX[i]) % nx, ys]
            # ---- moments ----
            T = 0.0
            for i in range(5):
                T += gl[i]
            rho = 0.0
            mx = 0.0
            my = 0.0
            for i in range(9):
                rho += fl[i]
                mx += CX[i] * fl[i]
                my += CY[i] * fl[i]
            Fy = gb * (T - T0)                 # Boussinesq buoyancy (per unit volume, rho0 = 1)
            ux = mx / rho
            uy = (my + 0.5 * Fy) / rho
            usq = ux * ux + uy * uy
            # ---- TRT collision, flow, with Guo forcing ----
            for i in range(9):
                j = OPF[i]
                if j < i:
                    continue
                cu = 3.0 * (CX[i] * ux + CY[i] * uy)
                fe_p = WF[i] * rho * (1.0 + 0.5 * cu * cu - 1.5 * usq)   # symmetric part
                fe_m = WF[i] * rho * cu                                  # antisymmetric part
                cf = CY[i] * Fy
                F_p = WF[i] * (-3.0 * uy * Fy + 9.0 * (CX[i] * ux + CY[i] * uy) * cf)
                F_m = WF[i] * 3.0 * cf
                fp = 0.5 * (fl[i] + fl[j])
                fm = 0.5 * (fl[i] - fl[j])
                if i == 0:
                    fn[0, x, y] = fl[0] - wp * (fl[0] - fe_p) + (1.0 - 0.5 * wp) * F_p
                else:
                    dp = -wp * (fp - fe_p) + (1.0 - 0.5 * wp) * F_p
                    dm = -wm * (fm - fe_m) + (1.0 - 0.5 * wm) * F_m
                    fn[i, x, y] = fl[i] + dp + dm
                    fn[j, x, y] = fl[j] + dp - dm
            # ---- TRT collision, temperature ----
            gn[0, x, y] = gl[0] - gwp * (gl[0] - WG[0] * T)
            for i in (1, 2):
                j = OPG[i]
                ge_m = WG[i] * T * 3.0 * (DX[i] * ux + DY[i] * uy)
                gp = 0.5 * (gl[i] + gl[j])
                gm = 0.5 * (gl[i] - gl[j])
                dp = -gwp * (gp - WG[i] * T)
                dm = -gwm * (gm - ge_m)
                gn[i, x, y] = gl[i] + dp + dm
                gn[j, x, y] = gl[j] + dp - dm
            rho_o[x, y] = rho
            ux_o[x, y] = ux
            uy_o[x, y] = uy
            T_o[x, y] = T


def trt_rates(tau_plus):
    """Return (omega+, omega-) for a given symmetric relaxation time and LAMBDA."""
    tau_minus = LAMBDA / (tau_plus - 0.5) + 0.5
    return 1.0 / tau_plus, 1.0 / tau_minus


class Layer:
    """One periodic fluid layer of height H (lattice nodes) and width NX, heated from below."""

    def __init__(self, H, NX, Ra, Pr, Uff, T_init):
        self.H, self.NX, self.Ra, self.Pr, self.Uff = H, NX, Ra, Pr, Uff
        self.nu = Uff * H * np.sqrt(Pr / Ra)
        self.kappa = self.nu / Pr
        self.gb = Uff ** 2 / H                         # g beta dT with dT = 1
        self.Th, self.Tc, self.T0 = 1.0, 0.0, 0.5
        tau_f = 3.0 * self.nu + 0.5
        self.wp, self.wm = trt_rates(tau_f)
        # temperature: antisymmetric rate sets kappa
        tau_gm = 3.0 * self.kappa + 0.5
        tau_gp = LAMBDA / (tau_gm - 0.5) + 0.5
        self.gwp, self.gwm = 1.0 / tau_gp, 1.0 / tau_gm
        self.tau_f, self.tau_g = tau_f, tau_gm

        T = np.asarray(T_init, dtype=np.float64)
        # hydrostatic density balancing the buoyancy of the initial state (avoids acoustic sloshing)
        Fy = self.gb * (T - self.T0)
        rho = 1.0 + 3.0 * np.cumsum(Fy, axis=1)
        rho += 1.0 - rho.mean()
        self.f = np.empty((9, NX, H))
        for i in range(9):
            self.f[i] = WF[i] * rho
        self.g = np.empty((5, NX, H))
        for i in range(5):
            self.g[i] = WG[i] * T
        self.fn = np.empty_like(self.f)
        self.gn = np.empty_like(self.g)
        self.rho = rho.copy()
        self.ux = np.zeros((NX, H))
        self.uy = np.zeros((NX, H))
        self.T = T.copy()
        self.t = 0

    @property
    def t_ff(self):
        """free-fall time H / U_ff in lattice steps"""
        return self.H / self.Uff

    @property
    def t_kappa(self):
        """thermal diffusion time H^2 / kappa in lattice steps"""
        return self.H ** 2 / self.kappa

    def advance(self, n=1):
        for _ in range(n):
            step(self.f, self.fn, self.g, self.gn, self.wp, self.wm, self.gwp, self.gwm,
                 self.gb, self.Th, self.Tc, self.T0, self.rho, self.ux, self.uy, self.T)
            self.f, self.fn = self.fn, self.f
            self.g, self.gn = self.gn, self.g
        self.t += n

    def nusselt(self):
        """(volume-average Nu, bottom-wall Nu, top-wall Nu)"""
        nu_vol = 1.0 + (self.uy * self.T).mean() * self.H / self.kappa
        T = self.T.mean(axis=0)
        # second-order one-sided gradient at the half-way wall: wall at y = -0.5, nodes at 0, 1
        dTdy_bot = (-8.0 * self.Th + 9.0 * T[0] - T[1]) / 3.0
        dTdy_top = (8.0 * self.Tc - 9.0 * T[-1] + T[-2]) / 3.0
        return float(nu_vol), float(-dTdy_bot * self.H), float(-dTdy_top * self.H)

    def conduction_error(self):
        y = (np.arange(self.H) + 0.5) / self.H
        return float(np.abs(self.T - (1.0 - y)[None, :]).max())


def conduction(H, NX):
    y = (np.arange(H) + 0.5) / H
    return np.repeat((1.0 - y)[None, :], NX, axis=0)


# ======================= onset =======================
def run_onset(a, out):
    H = a.H
    NX = int(round(2 * np.pi / KC * H))
    k_eff = 2 * np.pi * H / NX
    x = (np.arange(NX) + 0.5) / H
    y = (np.arange(H) + 0.5) / H
    mode = np.cos(k_eff * x)[:, None] * np.sin(np.pi * y)[None, :]
    rows = []
    for Ra in a.ras:
        L = Layer(H, NX, Ra, a.Pr, a.Uff, conduction(H, NX) + 1e-6 * mode)
        n_total = int(a.tk * L.t_kappa)
        every = max(1, n_total // 400)
        ts, amp = [], []
        t0 = time.time()
        while L.t < n_total:
            L.advance(every)
            # amplitude of the k_c mode in the vertical velocity at mid-height
            a_c = np.abs(np.sum(L.uy[:, H // 2] * np.exp(-1j * k_eff * x))) * 2 / NX
            ts.append(L.t / L.t_kappa)
            amp.append(a_c)
        ts, amp = np.array(ts), np.array(amp)
        # fit ln(amp) over the second half of the run (start-up transient has decayed)
        sel = ts > 0.5 * ts[-1]
        slope, icpt = np.polyfit(ts[sel], np.log(amp[sel]), 1)
        resid = np.log(amp[sel]) - (slope * ts[sel] + icpt)
        rows.append(dict(Ra=Ra, growth_rate=float(slope), fit_rms=float(resid.std()),
                         amp_end=float(amp[-1]), steps=int(L.t), t_kappa=float(L.t_kappa),
                         tau_f=L.tau_f, tau_g=L.tau_g, secs=round(time.time() - t0, 1)))
        np.save(out / f'onset_H{H}_Ra{Ra:g}.npy', np.vstack([ts, amp]))
        print(json.dumps(rows[-1]), flush=True)
    ra = np.array([r['Ra'] for r in rows])
    s = np.array([r['growth_rate'] for r in rows])
    p = np.polyfit(ra, s, 1)
    rac = -p[1] / p[0]
    res = dict(H=H, NX=NX, aspect=NX / H, k_eff=k_eff, Pr=a.Pr, Uff=a.Uff, runs=rows,
               slope_per_Ra=float(p[0]), Ra_c=float(rac), Ra_c_theory=RAC_THEORY,
               error_pct=float(100 * (rac - RAC_THEORY) / RAC_THEORY))
    (out / f'onset_H{H}.json').write_text(json.dumps(res, indent=2))
    print(f'H={H}: Ra_c = {rac:.2f}  (theory {RAC_THEORY}, {res["error_pct"]:+.3f}%)')


# ======================= rolls =======================
def run_rolls(a, out):
    H = a.H
    NX = int(round(2 * np.pi / KC * H))
    k_eff = 2 * np.pi * H / NX
    x = (np.arange(NX) + 0.5) / H
    y = (np.arange(H) + 0.5) / H
    mode = np.cos(k_eff * x)[:, None] * np.sin(np.pi * y)[None, :]
    rows = []
    for Ra in a.ras:
        L = Layer(H, NX, Ra, a.Pr, a.Uff, conduction(H, NX) + 0.05 * mode)
        t0 = time.time()
        chunk = int(0.05 * L.t_kappa)
        hist = []
        while True:
            L.advance(chunk)
            hist.append(L.nusselt())
            if len(hist) > 10 and L.t > a.tk_min * L.t_kappa:
                last = np.array([h[0] for h in hist[-6:]])
                if np.ptp(last) / last.mean() < 1e-6 or L.t > a.tk_max * L.t_kappa:
                    break
        nv, nb, nt = hist[-1]
        mach = float(np.sqrt(L.ux ** 2 + L.uy ** 2).max() * np.sqrt(3))
        rows.append(dict(Ra=Ra, Nu=nv, Nu_bottom=nb, Nu_top=nt, steps=int(L.t),
                         t_over_tkappa=L.t / L.t_kappa, drift=float(np.ptp([h[0] for h in hist[-6:]])),
                         max_mach=mach, tau_f=L.tau_f, tau_g=L.tau_g, secs=round(time.time() - t0, 1)))
        print(json.dumps(rows[-1]), flush=True)
    res = dict(H=H, NX=NX, aspect=NX / H, k_eff=k_eff, Pr=a.Pr, Uff=a.Uff, runs=rows)
    (out / f'rolls_H{H}.json').write_text(json.dumps(res, indent=2))


# ======================= movie =======================
def run_movie(a, out):
    H = a.H
    NX = int(round(a.aspect * H))
    rng = np.random.default_rng(1)
    noise = 1e-3 * rng.standard_normal((NX, H))
    noise -= noise.mean(axis=0, keepdims=True)
    base = conduction(H, NX) + noise
    layers = [Layer(H, NX, Ra, a.Pr, a.Uff, base) for Ra in a.ras]
    for L in layers:
        print(f'Ra={L.Ra:g}: nu={L.nu:.4f} tau_f={L.tau_f:.4f} tau_g={L.tau_g:.4f}', flush=True)
    t_ff = layers[0].t_ff
    every = max(1, int(round(a.frame_tff * t_ff)))
    n_frames = int(round(a.t_end / a.frame_tff))
    frames = {L.Ra: [] for L in layers}
    nu_hist = {L.Ra: [] for L in layers}
    times = []
    t0 = time.time()
    for n in range(n_frames + 1):
        for L in layers:
            frames[L.Ra].append(L.T.astype(np.float16))
            nu_hist[L.Ra].append(L.nusselt())
        times.append(layers[0].t / t_ff)
        if n == n_frames:
            break
        for L in layers:
            L.advance(every)
        if n % 50 == 0:
            print(f'frame {n}/{n_frames}  t={times[-1]:.1f} t_ff  ({time.time() - t0:.0f}s)  '
                  + '  '.join(f'Nu={nu_hist[L.Ra][-1][0]:.3f}' for L in layers), flush=True)
    # statistics after the movie: continue the runs to average Nu
    stats = {}
    for L in layers:
        stats[str(L.Ra)] = dict(Ra=L.Ra, tau_f=L.tau_f, tau_g=L.tau_g,
                                max_mach=float(np.sqrt(L.ux ** 2 + L.uy ** 2).max() * np.sqrt(3)))
    if a.t_avg > 0:
        n_avg = int(a.t_avg * t_ff)
        samp = max(1, int(0.25 * t_ff))
        acc = {L.Ra: [] for L in layers}
        done = 0
        while done < n_avg:
            for L in layers:
                L.advance(samp)
                acc[L.Ra].append(L.nusselt())
            done += samp
        for L in layers:
            arr = np.array(acc[L.Ra])
            stats[str(L.Ra)].update(Nu_mean=float(arr[:, 0].mean()), Nu_std=float(arr[:, 0].std()),
                                    Nu_bottom_mean=float(arr[:, 1].mean()),
                                    Nu_top_mean=float(arr[:, 2].mean()),
                                    avg_window_tff=a.t_avg, avg_start_tff=a.t_end)
    res = dict(H=H, NX=NX, aspect=NX / H, Pr=a.Pr, Uff=a.Uff, frame_tff=a.frame_tff,
               t_end_tff=a.t_end, cases=stats, secs=round(time.time() - t0, 1))
    (out / 'movie.json').write_text(json.dumps(res, indent=2))
    np.savez_compressed(out / 'movie.npz', times=np.array(times),
                        **{f'T_{L.Ra:g}': np.stack(frames[L.Ra]) for L in layers},
                        **{f'Nu_{L.Ra:g}': np.array(nu_hist[L.Ra]) for L in layers})
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['onset', 'rolls', 'movie'])
    ap.add_argument('--out', default='run')
    ap.add_argument('--H', type=int, default=64)
    ap.add_argument('--Pr', type=float, default=0.71)
    ap.add_argument('--Uff', type=float, default=0.05)
    ap.add_argument('--ras', type=float, nargs='+')
    ap.add_argument('--tk', type=float, default=3.0, help='onset: run length in thermal times')
    ap.add_argument('--tk_min', type=float, default=1.0)
    ap.add_argument('--tk_max', type=float, default=8.0)
    ap.add_argument('--aspect', type=float, default=4.0)
    ap.add_argument('--frame_tff', type=float, default=0.25)
    ap.add_argument('--t_end', type=float, default=90.0)
    ap.add_argument('--t_avg', type=float, default=0.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == 'onset':
        a.ras = a.ras or [1650, 1700, 1750, 1800]
        run_onset(a, out)
    elif a.mode == 'rolls':
        a.ras = a.ras or [2000, 3000, 5000, 10000]
        run_rolls(a, out)
    else:
        a.ras = a.ras or [1500, 1e4, 1e6]
        run_movie(a, out)


if __name__ == '__main__':
    main()
