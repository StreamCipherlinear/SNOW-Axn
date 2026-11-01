"""Search nested spans of strong three-word approximations for the cheapest
linear space, under the pricing rule of Theorem 4 of [ZLM+25].

Self-checks first: the top-ranked mask must be the optimum of Equation (19)
and its correlation must be 2^-54.7706.  Prints one row per dimension k.
"""
import sys, os, io, math, contextlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os, sys
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_CODE, 'mask_search', 'correlation') not in sys.path:
    sys.path.insert(0, os.path.join(_CODE, 'mask_search', 'correlation'))
with contextlib.redirect_stdout(io.StringIO()):
    from verify import LAT, ILAT, L, Linv, sigma, pullback

MU = (4, 5, 6, 12, 13, 14)


def nullspace(rows, n):
    rows = list(rows); piv = {}; r = 0
    for col in range(n):
        idx = None
        for i in range(r, len(rows)):
            if (rows[i] >> col) & 1:
                idx = i
                break
        if idx is None:
            continue
        rows[r], rows[idx] = rows[idx], rows[r]
        for i in range(len(rows)):
            if i != r and ((rows[i] >> col) & 1):
                rows[i] ^= rows[r]
        piv[col] = r
        r += 1
        if r == len(rows):
            break
    out = []
    for f in [c for c in range(n) if c not in piv]:
        v = 1 << f
        for c, ri in piv.items():
            if (rows[ri] >> f) & 1:
                v |= 1 << c
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
                if (w >> (8 * ob + ot)) & 1:
                    row |= 1 << j
            rows.append(row)
    out = []
    for v in nullspace(rows, len(free)):
        m = 0
        for j, (b, t) in enumerate(free):
            if (v >> j) & 1:
                m |= 1 << (8 * b + t)
        out.append(m)
    return out


tA = admissible_basis(MU, lambda w: pullback(w, Linv))
tC = admissible_basis(MU, lambda w: pullback(w, lambda s: Linv(sigma(L(s)))))
assert len(tA) == 16 and len(tC) == 8, (len(tA), len(tC))

to_bytes = lambda x: np.array([(int(x) >> (8 * i)) & 0xFF for i in range(16)], dtype=np.uint8)
A_basis = np.array([to_bytes(x) for x in tA], dtype=np.uint8)
C_basis = np.array([to_bytes(x) for x in tC], dtype=np.uint8)
P_basis = np.array([to_bytes(pullback(int(x), Linv)) for x in tA], dtype=np.uint8)
Q_basis = np.array([to_bytes(pullback(int(x), lambda s: Linv(sigma(L(s))))) for x in tC], dtype=np.uint8)

def expand(basis, sel):
    return np.bitwise_xor.reduce(sel[:, :, None] * basis[None, :, :], axis=1)

selB = ((np.arange(1 << 16)[:, None] >> np.arange(16)) & 1).astype(np.uint8)
selC = ((np.arange(1 << 8)[:, None] >> np.arange(8)) & 1).astype(np.uint8)
B = expand(A_basis, selB)
C = expand(C_basis, selC)
P = expand(P_basis, selB)
Q = expand(Q_basis, selC)

absL = np.abs(LAT.astype(np.int64))
absI = np.abs(ILAT.astype(np.int64))
ILa = np.abs(ILAT.astype(np.int32))
T12a = np.zeros((256, 256), dtype=np.uint8)
for m1 in range(256):
    pr = ILa[m1][None, :] * ILa
    pr[:, 0] = 0
    T12a[m1] = pr.argmax(axis=1).astype(np.uint8)

N = np.ones((len(B), len(C)), dtype=np.int64)
for i in MU:
    b = B[:, i][:, None]
    c = C[:, i][None, :]
    p = P[:, i][:, None]
    q = Q[:, i][None, :]
    cb = (b ^ c).astype(np.int64)
    a = T12a[cb, c].astype(np.int64)
    N *= (absL[p, q] // 4) * (absI[cb, a] // 4) * (absI[c, a] // 4)
N[0, 0] = 0
flat = N.astype(np.float64).ravel()
order = np.argsort(np.where((N != 0).ravel(), -flat, np.inf))[:64]
bi, cj = np.unravel_index(order, N.shape)

def triple(idx):
    Bb = B[bi[idx]].copy()
    Cb = C[cj[idx]].copy()
    Ab = np.zeros(16, dtype=np.uint8)
    for i in MU:
        Ab[i] = T12a[Cb[i] ^ Bb[i]][Cb[i]]
    return Ab, Bb, Cb

def log2_tau(Ab, Bb, Cb):
    Pf = pullback(int.from_bytes(bytes(Bb), 'little'), Linv)
    Qf = pullback(int.from_bytes(bytes(Cb), 'little'), lambda s: Linv(sigma(L(s))))
    p = np.array([(Pf >> (8 * i)) & 0xFF for i in range(16)], dtype=np.int64)
    q = np.array([(Qf >> (8 * i)) & 0xFF for i in range(16)], dtype=np.int64)
    cb = (Cb ^ Bb).astype(np.int64)
    a = Ab.astype(np.int64)
    v1 = absL[p, q]
    v2 = absI[cb, a]
    v3 = absI[Cb.astype(np.int64), a]
    if (v1 == 0).any() or (v2 == 0).any() or (v3 == 0).any():
        return float('-inf')
    return float(np.log2(v1 / 256).sum() + np.log2(v2 / 256).sum() + np.log2(v3 / 256).sum())

base = [triple(k) for k in range(len(order))]
lg0 = log2_tau(*base[0])
print('top mask  A', bytes(base[0][0]).hex())
print('          B', bytes(base[0][1]).hex())
print('          C', bytes(base[0][2]).hex())
print('log2 tau  %.6f   (paper: -54.770580)' % lg0)
assert abs(lg0 + 54.770580) < 1e-3, 'top-ranked mask is not the optimum'

print()
print('%3s %12s %9s %9s %9s' % ('k', 'log2 SEI', 'log2 N', 'log2 T', 'log2 mem'))
best = None
for k in range(1, 17):
    A_ = np.array([base[i][0] for i in range(k)])
    B_ = np.array([base[i][1] for i in range(k)])
    C_ = np.array([base[i][2] for i in range(k)])
    idx = np.arange(1, 1 << k, dtype=np.int64)
    sel = ((idx[:, None] >> np.arange(k)) & 1).astype(np.uint8)
    Ab = np.bitwise_xor.reduce(sel[:, :, None] * A_[None, :, :], axis=1)
    Bb = np.bitwise_xor.reduce(sel[:, :, None] * B_[None, :, :], axis=1)
    Cb = np.bitwise_xor.reduce(sel[:, :, None] * C_[None, :, :], axis=1)
    Pb_k = np.array([to_bytes(pullback(int.from_bytes(bytes(base[i][1]), 'little'), Linv)) for i in range(k)], dtype=np.uint8)
    Qb_k = np.array([to_bytes(pullback(int.from_bytes(bytes(base[i][2]), 'little'), lambda s: Linv(sigma(L(s))))) for i in range(k)], dtype=np.uint8)
    p = np.bitwise_xor.reduce(sel[:, :, None] * Pb_k[None, :, :], axis=1).astype(np.int64)
    q = np.bitwise_xor.reduce(sel[:, :, None] * Qb_k[None, :, :], axis=1).astype(np.int64)
    cb = (Cb ^ Bb).astype(np.int64)
    a = Ab.astype(np.int64)
    v1 = absL[p, q]
    v2 = absI[cb, a]
    v3 = absI[Cb.astype(np.int64), a]
    bad = (v1 == 0).any(1) | (v2 == 0).any(1) | (v3 == 0).any(1)
    lg = np.log2(np.maximum(v1, 1) / 256).sum(1) + np.log2(np.maximum(v2, 1) / 256).sum(1) + np.log2(np.maximum(v3, 1) / 256).sum(1)
    lg[bad] = -np.inf
    sei = float(np.exp2(4.0 * lg).sum())
    Nreq = 4 * k * math.log(2) / sei
    T = 2 * Nreq + k * 2 ** k
    mem = k * 2 ** k
    print('%3d %12.4f %9.4f %9.4f %9.4f' % (k, math.log2(sei), math.log2(Nreq), math.log2(T), math.log2(mem)))
    if best is None or T < best[0]:
        best = (T, k, sei)
print()
print('best k=%d   log2 T=%.4f   log2 SEI=%.4f' % (best[1], math.log2(best[0]), math.log2(best[2])))
