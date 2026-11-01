"""Compute the exact profile of the optimal pattern and the costs it implies.

Inputs: none; rebuilds Theta_mu from verify.py and the S-box tables.
Outputs: sum tau^2, tau^4, tau^6 and the two cost rules for selected m.
Paper object: the profile sums behind Section 4.2 and the cost of Table 4.
"""

import sys, io, contextlib, math
import numpy as np
import os, sys
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_CODE, 'mask_search', 'correlation') not in sys.path:
    sys.path.insert(0, os.path.join(_CODE, 'mask_search', 'correlation'))
with contextlib.redirect_stdout(io.StringIO()):
    from verify import LAT, ILAT, L, Linv, sigma, pullback
MU = (4, 5, 6, 12, 13, 14)
from math import log, sqrt, pi, log1p

def phi_inv_tail(eps):
    """x with Phi(x) = 1 - eps."""
    if eps > 1e-13:
        lo, hi = -40.0, 40.0
        for _ in range(300):
            mid = (lo + hi) / 2
            if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < 1.0 - eps: lo = mid
            else: hi = mid
        return (lo + hi) / 2
    u = -log(eps)
    x = sqrt(2 * u - log(2 * pi * u))
    for _ in range(60):
        lq = -x * x / 2 - log(x) - 0.5 * log(2 * pi) + log1p(-1 / x**2 + 3 / x**4 - 15 / x**6)
        fv = lq - log(eps)
        df = -x + 1.0 / x
        step = fv / df
        x -= step
        if abs(step) < 1e-12: break
    return x

def nullspace(rows, n):
    rows = list(rows); piv = {}; r = 0
    for col in range(n):
        idx = None
        for i in range(r, len(rows)):
            if (rows[i] >> col) & 1: idx = i; break
        if idx is None: continue
        rows[r], rows[idx] = rows[idx], rows[r]
        for i in range(len(rows)):
            if i != r and ((rows[i] >> col) & 1): rows[i] ^= rows[r]
        piv[col] = r; r += 1
        if r == len(rows): break
    out = []
    for f in [c for c in range(n) if c not in piv]:
        v = 1 << f
        for c, ri in piv.items():
            if (rows[ri] >> f) & 1: v |= 1 << c
        out.append(v)
    return out
def admissible_basis(mu, f):
    free = [(b, t) for b in mu for t in range(8)]
    outc = [b for b in range(16) if b not in mu]
    img = [f(1 << (8 * b + t)) for (b, t) in free]
    rows = []
    for ob in outc:
        for ot in range(8):
            row = 0
            for j, w in enumerate(img):
                if (w >> (8 * ob + ot)) & 1: row |= 1 << j
            rows.append(row)
    out = []
    for v in nullspace(rows, len(free)):
        m = 0
        for j, (b, t) in enumerate(free):
            if (v >> j) & 1: m |= 1 << (8 * b + t)
        out.append(m)
    return out
tA = admissible_basis(MU, lambda w: pullback(w, Linv))
tC = admissible_basis(MU, lambda w: pullback(w, lambda s: Linv(sigma(L(s)))))
Pb_ = [pullback(b, Linv) for b in tA]
Qb_ = [pullback(c, lambda s: Linv(sigma(L(s)))) for c in tC]
def arr(v): return np.array([[x >> (8 * i) & 0xff for i in range(16)] for x in v], dtype=np.uint8)
def expand(basis, k, sel): return np.bitwise_xor.reduce(sel[:, :, None] * basis[None, :, :], axis=1)
Bm = expand(arr(tA), 16, ((np.arange(1 << 16)[:, None] >> np.arange(16)) & 1).astype(np.uint8))
Pm = expand(arr(Pb_), 16, ((np.arange(1 << 16)[:, None] >> np.arange(16)) & 1).astype(np.uint8))
Cm = expand(arr(tC), 8, ((np.arange(1 << 8)[:, None] >> np.arange(8)) & 1).astype(np.uint8))
Qm = expand(arr(Qb_), 8, ((np.arange(1 << 8)[:, None] >> np.arange(8)) & 1).astype(np.uint8))
absL = np.abs(LAT.astype(np.int16)); absI = np.abs(ILAT.astype(np.int16))
ILa = np.abs(ILAT.astype(np.int32))
T12a = np.zeros((256, 256), dtype=np.uint8)
for m1 in range(256):
    pr = ILa[m1][None, :] * ILa; pr[:, 0] = 0
    T12a[m1] = pr.argmax(axis=1).astype(np.uint8)
N = np.ones((len(Bm), len(Cm)), dtype=np.int64)
for i in MU:
    b = Bm[:, i][:, None].astype(np.int32); c = Cm[:, i][None, :].astype(np.int32)
    p = Pm[:, i][:, None].astype(np.int32); q = Qm[:, i][None, :].astype(np.int32)
    cb = (b ^ c).astype(np.int32); a = T12a[cb, c].astype(np.int32)
    N *= (absL[p, q] // 4).astype(np.int64) * (absI[cb, a] // 4).astype(np.int64) * (absI[c, a] // 4).astype(np.int64)
N[0, 0] = 0
tau = N[N > 0].astype(np.float64) * 2.0 ** -108
tau.sort(); tau = tau[::-1]
def lg(x): return math.log2(x)
z1 = phi_inv_tail(1 - 0.999992)
print('Phi^-1(0.999992) = %.4f' % z1)
print('sum tau^2 = 2^%.3f ;  sum tau^4 = 2^%.3f ;  (sum tau^2)^2 = 2^%.3f ;  sum tau^6 = 2^%.3f'
      % (lg(np.sum(tau**2)), lg(np.sum(tau**4)), 2*lg(np.sum(tau**2)), lg(np.sum(tau**6))))

l = 512
def attack(s2, s4, rule):
    best = None
    for Bg in range(1, l + 1):
        rho2 = s2 * s2 if rule == 'sq' else s4
        M = (z1 + phi_inv_tail(2.0 ** -(Bg + 1))) ** 2 / rho2
        Nk = math.sqrt(M * 2.0 ** (l + 1 - Bg))
        T1 = math.sqrt(2.0) * Nk if M <= 2.0 ** (l - Bg) else Nk * (1 + lg(Nk))
        T2 = 2.0 ** Bg * (1 + math.ceil(lg(M)))
        T = T1 + T2
        if best is None or T < best[0]: best = (T, Bg, Nk, M)
    return best

print('\ncalibration against the paper (rule rho^2 = tau^4 for one mask):')
for lbl, tv in [("designers' tau=2^-55.741", 2.0**-55.741), ("optimum tau*=2^-54.771", 2.0**-54.771)]:
    a = attack(tv*tv, tv**4, 'quartic')
    print('  %-28s Bg=%3d  log2N=%.3f  log2T=%.3f' % (lbl, a[1], lg(a[2]), lg(a[0])))

print('\n m          | paper rule  rho=sum tau^2      | check-level  rho^2=sum tau^4')
print('            |  Bg   log2N   log2T            |  Bg   log2N   log2T')
for m in [1, 4, 16, 24, 64, 512, 4096, tau.size]:
    sub = tau[:m]; s2 = float(np.sum(sub**2)); s4 = float(np.sum(sub**4))
    a = attack(s2, s4, 'sq'); b = attack(s2, s4, 'quartic')
    print(' %-10d | %3d  %7.3f %7.3f            | %3d  %7.3f %7.3f' %
          (m, a[1], lg(a[2]), lg(a[0]), b[1], lg(b[2]), lg(b[0])))

