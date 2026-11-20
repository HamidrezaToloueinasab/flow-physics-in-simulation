"""
Pulsatile (Womersley) flow in a rigid circular tube.

Fully developed, axisymmetric, incompressible flow driven by an oscillating pressure gradient
dp/dx = -G cos(wt). The axial momentum (Navier-Stokes) equation reduces exactly to

    rho du/dt = G cos(wt) + mu (1/r) d/dr (r du/dr),   u(R) = 0,   du/dr(0) = 0.

Non-dimensional form (r* = r/R, t* = wt, u* = u / (G R^2 / 4 mu)):

    alpha^2 du/dt = 4 cos t + (1/r) d/dr (r du/dr),     alpha = R sqrt(w / nu)  (Womersley number)

Numerics: second-order finite-volume discretisation in r, Crank-Nicolson in time, started from rest
and run until the transient has decayed. The periodic state is compared with Womersley's exact
solution (Womersley 1955):

    u = Re{ 4/(i alpha^2) [1 - J0(i^{3/2} alpha r) / J0(i^{3/2} alpha)] e^{it} }

Outputs (in --out): results.json (errors, flow-rate amplitude and phase lag per alpha) and
profiles.npz (velocity profiles over one cycle for the video).

Usage:  python womersley.py --out run
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import jv

I32 = 1j ** 1.5


# ----------------------------------------------------------------------------- exact solution
def exact_u(alpha, r, t):
    k = I32 * alpha
    return np.real(4 / (1j * alpha ** 2) * (1 - jv(0, k * r) / jv(0, k)) * np.exp(1j * t))


def exact_Q_complex(alpha):
    """Complex amplitude of Q* = 2 int_0^1 u r dr (steady Poiseuille with the same G gives Q* = 1/2)."""
    k = I32 * alpha
    return 4 / (1j * alpha ** 2) * (1 - 2 * jv(1, k) / (k * jv(0, k)))


def exact_tau_complex(alpha):
    """Complex amplitude of the wall shear stress -du*/dr at r = 1 (steady Poiseuille gives 2)."""
    k = I32 * alpha
    return 4 / (1j * alpha ** 2) * (-k * jv(1, k) / jv(0, k))


# ----------------------------------------------------------------------------- numerical solution
def operator_bands(n):
    """Banded matrix of the finite-volume operator (1/r) d/dr(r du/dr) on r_j = j h, j = 0..n-1
    (u_n = 0 at the wall is eliminated)."""
    h = 1.0 / n
    r = np.arange(n) * h
    lo, di, up = np.zeros(n), np.zeros(n), np.zeros(n)
    # centre cell j = 0 (control volume [0, h/2]): 4 (u1 - u0) / h^2
    di[0], up[0] = -4 / h ** 2, 4 / h ** 2
    j = np.arange(1, n)
    rp, rm = r[j] + h / 2, r[j] - h / 2
    lo[j] = rm / (r[j] * h ** 2)
    up[j] = rp / (r[j] * h ** 2)
    di[j] = -(rp + rm) / (r[j] * h ** 2)
    up[-1] = 0.0                                       # wall neighbour u_n = 0
    return r, lo, di, up


def apply(lo, di, up, u):
    out = di * u
    out[1:] += lo[1:] * u[:-1]
    out[:-1] += up[:-1] * u[1:]
    return out


def solve(alpha, n=400, steps_per_cycle=4000, cycles=60, samples=120):
    r, lo, di, up = operator_bands(n)
    dt = 2 * np.pi / steps_per_cycle
    a2 = alpha ** 2
    # (a2 I - dt/2 L) u^{n+1} = (a2 I + dt/2 L) u^n + dt * 4 cos(t^{n+1/2})
    ab = np.zeros((3, n))
    ab[0, 1:] = -dt / 2 * up[:-1]
    ab[1] = a2 - dt / 2 * di
    ab[2, :-1] = -dt / 2 * lo[1:]
    u = np.zeros(n)
    every = steps_per_cycle // samples
    prof, t_s = [], []
    for s in range(cycles * steps_per_cycle):
        t_half = (s + 0.5) * dt
        rhs = a2 * u + dt / 2 * apply(lo, di, up, u) + dt * 4 * np.cos(t_half)
        u = solve_banded((1, 1), ab, rhs)
        if s >= (cycles - 1) * steps_per_cycle and (s + 1) % every == 0:
            prof.append(np.append(u, 0.0))             # add the wall point
            t_s.append((s + 1) * dt)
    return np.append(r, 1.0), np.array(t_s), np.array(prof)


def fundamental(signal, t):
    """Complex amplitude c of the e^{it} harmonic: signal ~ Re{c e^{it}} over one full cycle."""
    return 2 * np.mean(signal * np.exp(-1j * t))


def analyse(alpha, r, t, prof):
    w = r.copy(); w[0] /= 2; w[-1] /= 2                # trapezoid weights for int u r dr
    h = r[1] - r[0]
    Q = 2 * h * (prof * w).sum(axis=1)
    tau = -(3 * prof[:, -1] - 4 * prof[:, -2] + prof[:, -3]) / (2 * h)   # -du/dr at the wall (2nd order)
    ue = np.array([exact_u(alpha, r, ti) for ti in t])
    err = np.abs(prof - ue).max() / np.abs(ue).max()
    Qc, Qe = fundamental(Q, t), exact_Q_complex(alpha)
    Tc, Te = fundamental(tau, t), exact_tau_complex(alpha)
    deg = lambda z: float(np.degrees(np.angle(z)))
    return {
        'alpha': alpha,
        'max_profile_error_rel': float(err),
        'flow_amplitude_vs_steady': {'numerical': float(abs(Qc) / 0.5), 'exact': float(abs(Qe) / 0.5)},
        'flow_phase_lag_deg': {'numerical': -deg(Qc), 'exact': -deg(Qe)},
        'wall_shear_lead_over_flow_deg': {'numerical': deg(Tc) - deg(Qc), 'exact': deg(Te) - deg(Qe)},
        'reverse_flow_near_wall_while_core_forward': bool(np.any((prof[:, 0] > 0) & (prof[:, -5] < 0))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--alphas', type=float, nargs='+', default=[1.0, 5.0, 15.0])
    ap.add_argument('--n', type=int, default=400, help='radial cells')
    ap.add_argument('--steps', type=int, default=4000, help='time steps per cycle')
    ap.add_argument('--cycles', type=int, default=60, help='cycles simulated from rest')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results, store = [], {}
    for alpha in a.alphas:
        r, t, prof = solve(alpha, a.n, a.steps, a.cycles)
        res = analyse(alpha, r, t, prof)
        results.append(res)
        store[f'u_{alpha:g}'] = prof.astype(np.float32)
        print(json.dumps(res))
    np.savez_compressed(out / 'profiles.npz', r=r, t=t, **store)
    meta = dict(n=a.n, steps_per_cycle=a.steps, cycles=a.cycles, results=results)
    (out / 'results.json').write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
