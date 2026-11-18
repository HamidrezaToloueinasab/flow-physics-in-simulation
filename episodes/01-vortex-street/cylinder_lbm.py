"""
Flow past a circular cylinder: steady wake (Re = 40) vs. von Karman vortex street (Re = 100).

Method: lattice Boltzmann, D2Q9 lattice, BGK collision, half-way bounce-back on the cylinder,
equilibrium velocity inlet, zero-gradient outlet, periodic top/bottom boundaries.
Both cases run side by side on identical grids; only the viscosity differs.

The defaults reproduce the published results (D = 30 cells, 20D periodic spacing,
inlet 15D upstream, outlet 25D downstream).

Outputs (in --out):
  frames.npz    vorticity snapshots (float16) for the video, both cases
  results.json  Strouhal number (Re = 100), recirculation length (Re = 40), run parameters
  probe.npy     transverse-velocity probe signal 3D behind the cylinder (Re = 100)

Usage:
  python cylinder_lbm.py --out run
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

# ---------------- lattice ----------------
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
WT = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)

# ---------------- case setup (overwritten from the command line in main) ----------------
D = 30                 # cylinder diameter [lattice units]
NX, NY = 40 * D, 20 * D
XC, YC = 15 * D, NY // 2
U0 = 0.06              # inlet velocity [lattice units], Mach ~ 0.1


@njit(cache=True, inline='always')
def feq_i(i, rho, ux, uy):
    cu = 3.0 * (CX[i] * ux + CY[i] * uy)
    return WT[i] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy))


@njit(parallel=True, cache=True)
def step(f, fnew, solid, omega, u0, rho_out, ux_out, uy_out):
    """One fused pull-stream + BGK collide step. f holds post-collision populations."""
    nx, ny = solid.shape
    for x in prange(nx):
        fl = np.empty(9)
        for y in range(ny):
            if solid[x, y]:
                for i in range(9):
                    fnew[i, x, y] = 0.0
                rho_out[x, y] = 1.0
                ux_out[x, y] = 0.0
                uy_out[x, y] = 0.0
                continue
            if x == 0:
                # equilibrium velocity inlet
                for i in range(9):
                    fnew[i, x, y] = feq_i(i, 1.0, u0, 0.0)
                rho_out[x, y] = 1.0
                ux_out[x, y] = u0
                uy_out[x, y] = 0.0
                continue
            # pull streaming
            for i in range(9):
                xs = x - CX[i]
                ys = (y - CY[i]) % ny
                if xs >= nx:          # zero-gradient outlet
                    xs = nx - 1
                if solid[xs, ys]:     # half-way bounce-back
                    fl[i] = f[OPP[i], x, y]
                else:
                    fl[i] = f[i, xs, ys]
            rho = 0.0
            mx = 0.0
            my = 0.0
            for i in range(9):
                rho += fl[i]
                mx += CX[i] * fl[i]
                my += CY[i] * fl[i]
            ux = mx / rho
            uy = my / rho
            for i in range(9):
                fnew[i, x, y] = fl[i] - omega * (fl[i] - feq_i(i, rho, ux, uy))
            rho_out[x, y] = rho
            ux_out[x, y] = ux
            uy_out[x, y] = uy


class Case:
    def __init__(self, re):
        self.re = re
        self.nu = U0 * D / re
        self.tau = 3.0 * self.nu + 0.5
        self.omega = 1.0 / self.tau
        xx, yy = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
        self.solid = (xx - XC) ** 2 + (yy - YC) ** 2 <= (D / 2) ** 2
        ux = np.full((NX, NY), U0)
        uy = 0.01 * U0 * np.sin(2 * np.pi * xx / NX)   # small asymmetric kick to break symmetry
        ux[self.solid] = 0.0
        uy[self.solid] = 0.0
        self.f = np.empty((9, NX, NY))
        for i in range(9):
            self.f[i] = feq_i_np(i, np.ones((NX, NY)), ux, uy)
        self.fnew = np.empty_like(self.f)
        self.rho = np.ones((NX, NY))
        self.ux = ux.copy()
        self.uy = uy.copy()

    def advance(self):
        step(self.f, self.fnew, self.solid, self.omega, U0, self.rho, self.ux, self.uy)
        self.f, self.fnew = self.fnew, self.f

    def vorticity(self):
        """Normalised vorticity omega_z * D / U0."""
        dvdx = np.gradient(self.uy, axis=0)
        dudy = np.gradient(self.ux, axis=1)
        w = (dvdx - dudy) * D / U0
        w[self.solid] = np.nan
        return w


def feq_i_np(i, rho, ux, uy):
    cu = 3.0 * (CX[i] * ux + CY[i] * uy)
    return WT[i] * rho * (1.0 + cu + 0.5 * cu ** 2 - 1.5 * (ux ** 2 + uy ** 2))


def strouhal(signal, dt_steps=1):
    """Mean period from upward zero crossings (linear interpolation) of a zero-mean signal."""
    s = signal - signal.mean()
    idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    t = idx + (-s[idx]) / (s[idx + 1] - s[idx])
    periods = np.diff(t) * dt_steps
    T = periods.mean()
    return D / (T * U0), T, periods


def recirculation_length(case):
    """Distance from the cylinder's rear surface to where centreline u_x changes sign, in D."""
    u = case.ux[:, YC]
    x_rear = XC + D / 2
    xs = np.arange(NX)
    behind = (xs > x_rear) & ~case.solid[:, YC]
    xb = xs[behind]
    ub = u[behind]
    k = np.where((ub[:-1] < 0) & (ub[1:] >= 0))[0][0]
    x0 = xb[k] + (-ub[k]) / (ub[k + 1] - ub[k])
    return (x0 - x_rear) / D


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='output folder')
    ap.add_argument('--warmup', type=int, default=120000, help='steps before recording')
    ap.add_argument('--record', type=int, default=30000, help='steps recorded for St and the video')
    ap.add_argument('--frame_every', type=int, default=75, help='steps between saved video frames')
    ap.add_argument('--D', type=int, default=30, help='cylinder diameter in lattice cells')
    ap.add_argument('--spacing', type=int, default=20, help='domain height (periodic spacing) in D')
    ap.add_argument('--upstream', type=int, default=15, help='inlet-to-cylinder-centre distance in D')
    ap.add_argument('--downstream', type=int, default=25, help='cylinder-centre-to-outlet distance in D')
    a = ap.parse_args()
    global D, NX, NY, XC, YC
    D = a.D
    NX, NY = (a.upstream + a.downstream) * D, a.spacing * D
    XC, YC = a.upstream * D, NY // 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = [Case(40), Case(100)]
    for c in cases:
        print(f"Re={c.re}: nu={c.nu:.4f}, tau={c.tau:.4f}")

    t0 = time.time()
    for n in range(a.warmup):
        for c in cases:
            c.advance()
        if n % 10000 == 0:
            print(f"warm-up {n}/{a.warmup}  ({time.time() - t0:.0f}s)", flush=True)

    # Re = 40 steadiness check: change of the centreline velocity over 1000 steps
    ref = cases[0].ux[:, YC].copy()
    for _ in range(1000):
        for c in cases:
            c.advance()
    steady_change = float(np.abs(cases[0].ux[:, YC] - ref).max() / U0)

    probe_x = XC + 3 * D
    probe = np.empty(a.record)
    frames = {40: [], 100: []}
    for n in range(a.record):
        for c in cases:
            c.advance()
        probe[n] = cases[1].uy[probe_x, YC]
        if n % a.frame_every == 0:
            for c in cases:
                frames[c.re].append(c.vorticity().astype(np.float16))
    print(f"done in {time.time() - t0:.0f}s", flush=True)

    st, T, periods = strouhal(probe)
    lr = recirculation_length(cases[0])
    max_mach = float(max(np.sqrt(c.ux ** 2 + c.uy ** 2).max() for c in cases) * np.sqrt(3))
    rho_dev = float(max(np.abs(c.rho[~c.solid] - 1).max() for c in cases))
    res = {
        'grid': [NX, NY], 'D': D, 'U0': U0, 'cylinder_center': [XC, YC],
        'blockage_D_over_spacing': D / NY,
        'tau': {str(c.re): c.tau for c in cases},
        'warmup_steps': a.warmup, 'record_steps': a.record, 'frame_every': a.frame_every,
        'Re100_strouhal': st, 'Re100_period_steps': T,
        'Re100_period_spread_steps': float(periods.std()), 'Re100_n_periods': int(len(periods)),
        'Re40_recirculation_length_over_D': lr,
        'Re40_steadiness_max_du_over_U0_per_1000_steps': steady_change,
        'max_mach': max_mach, 'max_density_deviation': rho_dev,
    }
    (out / 'results.json').write_text(json.dumps(res, indent=2))
    np.save(out / 'probe.npy', probe)
    np.savez_compressed(out / 'frames.npz',
                        re40=np.stack(frames[40]), re100=np.stack(frames[100]),
                        solid=cases[0].solid)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
