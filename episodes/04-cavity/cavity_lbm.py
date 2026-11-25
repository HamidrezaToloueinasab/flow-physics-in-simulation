"""
Natural convection of air in a square cavity heated from the side (2D, Boussinesq).

The left wall is hot, the right wall is cold, the top and bottom walls are adiabatic; all walls are no-slip.
This is the classic benchmark of de Vahl Davis (1983) for Pr = 0.71 and Ra = 10^3 ... 10^6.

Method: double-distribution lattice Boltzmann (same core as episode 03).
  flow         D2Q9, two-relaxation-time (TRT) collision, Guo forcing for buoyancy (gravity along -y)
  temperature  D2Q5, TRT collision (the antisymmetric rate sets the diffusivity)
  walls        half-way bounce-back (no slip) for the flow;
               anti-bounce-back (fixed temperature) on the hot and cold walls,
               bounce-back (zero heat flux) on the adiabatic top and bottom walls.

Non-dimensional groups: Ra = g beta dT L^3 / (nu kappa), Pr = nu / kappa, with L the cavity side.
Lattice parameters follow from the free-fall velocity U_ff = sqrt(g beta dT L) (lattice units):
  nu = U_ff L sqrt(Pr / Ra),  kappa = nu / Pr.

Benchmark quantities (de Vahl Davis scaling: lengths by L, velocities by kappa / L):
  Nu_avg     mean Nusselt number on the hot wall
  Nu_mid     mean Nusselt number on the vertical mid-plane (convection + conduction)
  Nu_max/min local Nusselt number extremes on the hot wall, with their heights y
  u_max      largest horizontal velocity on the vertical mid-line x = 1/2, and its height y
  v_max      largest vertical velocity on the horizontal mid-line y = 1/2, and its position x

Subcommands
  bench   steady states for a list of Ra on one grid -> bench_L{L}.json (+ fields_L{L}_Ra{Ra}.npz)
  movie   all Ra started together from rest (uniform mean temperature), frames for the video
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
            # ---- stream f: bounce-back at all four walls ----
            for i in range(9):
                xs = x - CX[i]
                ys = y - CY[i]
                if xs < 0 or xs >= nx or ys < 0 or ys >= ny:
                    fl[i] = f[OPF[i], x, y]
                else:
                    fl[i] = f[i, xs, ys]
            # ---- stream g: fixed T on the side walls, zero flux on top and bottom ----
            for i in range(5):
                xs = x - DX[i]
                ys = y - DY[i]
                if xs < 0:
                    gl[i] = -g[OPG[i], x, y] + 2.0 * WG[i] * Th
                elif xs >= nx:
                    gl[i] = -g[OPG[i], x, y] + 2.0 * WG[i] * Tc
                elif ys < 0 or ys >= ny:
                    gl[i] = g[OPG[i], x, y]
                else:
                    gl[i] = g[i, xs, ys]
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
            Fy = gb * (T - T0)                 # Boussinesq buoyancy, gravity along -y
            ux = mx / rho
            uy = (my + 0.5 * Fy) / rho
            usq = ux * ux + uy * uy
            # ---- TRT collision, flow, with Guo forcing ----
            for i in range(9):
                j = OPF[i]
                if j < i:
                    continue
                cu = 3.0 * (CX[i] * ux + CY[i] * uy)
                fe_p = WF[i] * rho * (1.0 + 0.5 * cu * cu - 1.5 * usq)
                fe_m = WF[i] * rho * cu
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


class Cavity:
    """Square cavity of side L lattice nodes; hot wall at x = -1/2, cold wall at x = L - 1/2."""

    def __init__(self, L, Ra, Pr, Uff):
        self.L, self.Ra, self.Pr, self.Uff = L, Ra, Pr, Uff
        self.nu = Uff * L * np.sqrt(Pr / Ra)
        self.kappa = self.nu / Pr
        self.gb = Uff ** 2 / L                         # g beta dT with dT = 1
        self.Th, self.Tc, self.T0 = 1.0, 0.0, 0.5
        tau_f = 3.0 * self.nu + 0.5
        self.wp = 1.0 / tau_f
        self.wm = 1.0 / (LAMBDA / (tau_f - 0.5) + 0.5)
        tau_gm = 3.0 * self.kappa + 0.5
        self.gwm = 1.0 / tau_gm
        self.gwp = 1.0 / (LAMBDA / (tau_gm - 0.5) + 0.5)
        self.tau_f, self.tau_g = tau_f, tau_gm
        # start: fluid at rest at the mean temperature (no buoyancy, uniform density)
        self.f = np.empty((9, L, L))
        for i in range(9):
            self.f[i] = WF[i]
        self.g = np.empty((5, L, L))
        for i in range(5):
            self.g[i] = WG[i] * self.T0
        self.fn = np.empty_like(self.f)
        self.gn = np.empty_like(self.g)
        self.rho = np.ones((L, L))
        self.ux = np.zeros((L, L))
        self.uy = np.zeros((L, L))
        self.T = np.full((L, L), self.T0)
        self.t = 0

    @property
    def t_ff(self):
        """free-fall time L / U_ff in lattice steps"""
        return self.L / self.Uff

    def advance(self, n=1):
        for _ in range(n):
            step(self.f, self.fn, self.g, self.gn, self.wp, self.wm, self.gwp, self.gwm,
                 self.gb, self.Th, self.Tc, self.T0, self.rho, self.ux, self.uy, self.T)
            self.f, self.fn = self.fn, self.f
            self.g, self.gn = self.gn, self.g
        self.t += n

    # ---------------- benchmark quantities ----------------
    def local_nu_hot(self):
        """Local Nusselt number along the hot wall (second-order one-sided gradient at the half-way wall)."""
        dTdx = (-8.0 * self.Th + 9.0 * self.T[0, :] - self.T[1, :]) / 3.0
        return -dTdx * self.L

    def local_nu_cold(self):
        dTdx = (8.0 * self.Tc - 9.0 * self.T[-1, :] + self.T[-2, :]) / 3.0
        return -dTdx * self.L

    def nu_midplane(self):
        """Mean Nu across the vertical mid-plane x = L/2 (between the two central columns)."""
        L = self.L
        a, b = L // 2 - 1, L // 2
        Tm = 0.5 * (self.T[a] + self.T[b])
        um = 0.5 * (self.ux[a] + self.ux[b])
        dTdx = self.T[b] - self.T[a]
        return float(np.mean(um * Tm * L / self.kappa - dTdx * L))

    def scaled(self, v):
        """velocity in de Vahl Davis units (kappa / L)"""
        return v * self.L / self.kappa

    def benchmark(self):
        L = self.L
        y = (np.arange(L) + 0.5) / L
        nh = self.local_nu_hot()
        a, b = L // 2 - 1, L // 2
        u_mid = self.scaled(0.5 * (self.ux[a, :] + self.ux[b, :]))      # u on x = 1/2, vs y
        v_mid = self.scaled(0.5 * (self.uy[:, a] + self.uy[:, b]))      # v on y = 1/2, vs x
        u_max, y_u = _peak(y, u_mid)
        v_max, x_v = _peak(y, v_mid)
        nmax, y_nmax = _peak(y, nh)
        nmin, y_nmin = _peak(y, -nh)
        return dict(
            Nu_avg=float(nh.mean()), Nu_cold=float(self.local_nu_cold().mean()), Nu_mid=self.nu_midplane(),
            Nu_max=nmax, y_Nu_max=y_nmax, Nu_min=-nmin, y_Nu_min=y_nmin,
            u_max=u_max, y_u_max=y_u, v_max=v_max, x_v_max=x_v,
            max_mach=float(np.sqrt(self.ux ** 2 + self.uy ** 2).max() * np.sqrt(3)),
        )


def _peak(s, q):
    """Maximum of a sampled profile, refined with a parabola through the three highest samples."""
    k = int(np.argmax(q))
    if k == 0 or k == len(q) - 1:          # extreme at the end of the line (e.g. Nu_min in a corner)
        return float(q[k]), float(s[k])
    y0, y1, y2 = q[k - 1], q[k], q[k + 1]
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    h = s[1] - s[0]
    return float(y1 - 0.25 * (y0 - y2) * d), float(s[k] + d * h)


# ======================= bench =======================
def run_bench(a, out):
    rows = []
    for Ra in a.ras:
        c = Cavity(a.L, Ra, a.Pr, a.Uff)
        chunk = max(100, int(round(0.5 * c.t_ff)))
        t0 = time.time()
        prev = None
        converged = False
        while c.t < a.max_tff * c.t_ff:
            c.advance(chunk)
            nu = c.local_nu_hot().mean()
            if prev is not None:
                dnu = abs(nu - prev[0]) / nu
                du = np.abs(c.uy - prev[1]).max() / max(np.abs(c.uy).max(), 1e-30)
                if dnu < a.tol and du < a.tol * 10:
                    converged = True
                    break
            prev = (nu, c.uy.copy())
        r = dict(Ra=Ra, L=a.L, **c.benchmark(), steps=int(c.t), t_over_tff=c.t / c.t_ff, converged=converged,
                 tau_f=c.tau_f, tau_g=c.tau_g, secs=round(time.time() - t0, 1))
        rows.append(r)
        np.savez_compressed(out / f'fields_L{a.L}_Ra{Ra:g}.npz', T=c.T.astype(np.float32),
                            u=c.scaled(c.ux).astype(np.float32), v=c.scaled(c.uy).astype(np.float32))
        print(json.dumps(r), flush=True)
    res = dict(L=a.L, Pr=a.Pr, Uff=a.Uff, tol=a.tol, runs=rows)
    (out / f'bench_L{a.L}.json').write_text(json.dumps(res, indent=2))


# ======================= movie =======================
def run_movie(a, out):
    cavs = [Cavity(a.L, Ra, a.Pr, a.Uff) for Ra in a.ras]
    for c in cavs:
        print(f'Ra={c.Ra:g}: nu={c.nu:.4f} tau_f={c.tau_f:.4f} tau_g={c.tau_g:.4f}', flush=True)
    t_ff = cavs[0].t_ff
    # frame schedule: every frame_fine up to t_fine (early transient), then every frame_tff up to t_end
    sched = np.concatenate([np.arange(0.0, a.t_fine, a.frame_fine),
                            np.arange(a.t_fine, a.t_end + 1e-9, a.frame_tff)])
    steps = np.unique(np.round(sched * t_ff).astype(np.int64))
    n_frames = len(steps) - 1
    frames = {c.Ra: dict(T=[], u=[], v=[]) for c in cavs}
    nu_hist = {c.Ra: [] for c in cavs}
    times = []
    t0 = time.time()
    for n in range(n_frames + 1):
        for c in cavs:
            k = a.stride
            frames[c.Ra]['T'].append(c.T[k // 2::k, k // 2::k].astype(np.float16))
            frames[c.Ra]['u'].append(c.scaled(c.ux)[k // 2::k, k // 2::k].astype(np.float16))
            frames[c.Ra]['v'].append(c.scaled(c.uy)[k // 2::k, k // 2::k].astype(np.float16))
            nu_hist[c.Ra].append(c.local_nu_hot().mean())
        times.append(cavs[0].t / t_ff)
        if n == n_frames:
            break
        for c in cavs:
            c.advance(int(steps[n + 1] - steps[n]))
        if n % 50 == 0:
            print(f'frame {n}/{n_frames}  t={times[-1]:.1f} t_ff  ({time.time() - t0:.0f}s)  '
                  + '  '.join(f'Nu={nu_hist[c.Ra][-1]:.3f}' for c in cavs), flush=True)
    res = dict(L=a.L, Pr=a.Pr, Uff=a.Uff, frame_tff=a.frame_tff, frame_fine=a.frame_fine, t_fine=a.t_fine,
               t_end_tff=a.t_end, stride=a.stride,
               final={str(c.Ra): c.benchmark() for c in cavs}, secs=round(time.time() - t0, 1))
    (out / 'movie.json').write_text(json.dumps(res, indent=2))
    arrays = {'times': np.array(times)}
    for c in cavs:
        k = f'{c.Ra:g}'
        for q in ('T', 'u', 'v'):
            arrays[f'{q}_{k}'] = np.stack(frames[c.Ra][q])
        arrays[f'Nu_{k}'] = np.array(nu_hist[c.Ra])
    np.savez_compressed(out / 'movie.npz', **arrays)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['bench', 'movie'])
    ap.add_argument('--out', default='run')
    ap.add_argument('--L', type=int, default=128, help='cavity side in lattice nodes')
    ap.add_argument('--Pr', type=float, default=0.71)
    ap.add_argument('--Uff', type=float, default=0.1, help='free-fall velocity in lattice units')
    ap.add_argument('--ras', type=float, nargs='+', default=[1e3, 1e4, 1e5, 1e6])
    ap.add_argument('--tol', type=float, default=1e-7, help='bench: steady-state tolerance per half free-fall time')
    ap.add_argument('--max_tff', type=float, default=3000.0, help='bench: maximum run length in free-fall times')
    ap.add_argument('--frame_tff', type=float, default=0.5)
    ap.add_argument('--t_end', type=float, default=150.0)
    ap.add_argument('--stride', type=int, default=1, help='movie: store every n-th node')
    ap.add_argument('--t_fine', type=float, default=0.0, help='movie: use frame_fine spacing up to this time')
    ap.add_argument('--frame_fine', type=float, default=0.1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == 'bench':
        run_bench(a, out)
    else:
        run_movie(a, out)


if __name__ == '__main__':
    main()
