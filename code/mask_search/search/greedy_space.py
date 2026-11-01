"""Greedy basis selection from the correlation profile, in the spirit of
[ZLM+25]: take the profile of the family, choose a basis of masks, generate
every linear combination of the basis, compute the correlation of each, and
sum the fourth powers to obtain the capacity SEI(V).  At every step the mask
that adds the largest SEI is chosen.  The costing follows Theorem 4 of
[ZLM+25] with v = k and d1 = k.
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
                idx = i; break
        if idx is None:
            continue
        rows[r], rows[idx] = rows[idx], rows[r]
        for i in range(len(rows)):
            if i != r and ((rows[i] >> col) & 1):
                rows[i] ^= rows[r]
        piv[col] = r; r += 1
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


to_bytes = lambda x: np.array([(int(x) >> (8 * i)) & 0xFF for i in range(16)], dtype=np.uint8)
tA = admissible_basis(MU, lambda w: pullback(w, Linv))
tC = admissible_basis(MU, lambda w: pullback(w, lambda s: Linv(sigma(L(s)))))
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

Nrank = np.ones((len(B), len(C)), dtype=np.int64)
for i in MU:
    b = B[:, i][:, None]; c = C[:, i][None, :]
    p = P[:, i][:, None]; q = Q[:, i][None, :]
    cb = (b ^ c).astype(np.int64)
    a = T12a[cb, c].astype(np.int64)
    Nrank *= (absL[p, q] // 4) * (absI[cb, a] // 4) * (absI[c, a] // 4)
Nrank[0, 0] = 0
flat = Nrank.astype(np.float64).ravel()
order = np.argsort(np.where((Nrank != 0).ravel(), -flat, np.inf))[:96]
bi, cj = np.unravel_index(order, Nrank.shape)

cands = []
for k in range(len(order)):
    Bb = B[bi[k]].copy(); Cb = C[cj[k]].copy()
    Ab = np.zeros(16, dtype=np.uint8)
    for i in MU:
        Ab[i] = T12a[Cb[i] ^ Bb[i]][Cb[i]]
    cands.append((Ab, Bb, Cb, P[bi[k]].copy(), Q[cj[k]].copy()))


def lg_of(Ab, Bb, Cb, p, q):
    cb = (Cb ^ Bb).astype(np.int64)
    a = Ab.astype(np.int64)
    v1 = absL[p, q]; v2 = absI[cb, a]; v3 = absI[Cb.astype(np.int64), a]
    bad = (v1 == 0).any(1) | (v2 == 0).any(1) | (v3 == 0).any(1)
    lg = (np.log2(np.maximum(v1, 1) / 256).sum(1) + np.log2(np.maximum(v2, 1) / 256).sum(1)
          + np.log2(np.maximum(v3, 1) / 256).sum(1))
    lg[bad] = -np.inf
    return lg


def log2_sei(lgs):
    finite = lgs[np.isfinite(lgs)]
    if finite.size == 0:
        return float('-inf')
    m = 4.0 * finite.max()
    return float(m + math.log2(np.exp2(4.0 * finite - m).sum()))


V_A = np.zeros((1, 16), dtype=np.uint8)
V_B = np.zeros((1, 16), dtype=np.uint8)
V_C = np.zeros((1, 16), dtype=np.uint8)
V_p = np.zeros((1, 16), dtype=np.uint8)
V_q = np.zeros((1, 16), dtype=np.uint8)
chosen = []
print('%2s %13s %10s %10s %10s' % ('k', 'log2 SEI', 'log2 N', 'log2 T', 'log2 mem'))
best = None
for step in range(1, 17):
    bestc, bestsei, bestvec = None, float('-inf'), None
    for c, (Ab, Bb, Cb, p, q) in enumerate(cands):
        if c in chosen:
            continue
        nA = V_A ^ Ab; nB = V_B ^ Bb; nC = V_C ^ Cb
        np_ = V_p ^ p; nq = V_q ^ q
        lg = lg_of(nA, nB, nC, np_, nq)
        s = log2_sei(lg)
        if s > bestsei:
            bestsei, bestc, bestvec = s, c, (nA, nB, nC, np_, nq)
    if bestc is None:
        break
    V_A, V_B, V_C, V_p, V_q = bestvec
    chosen.append(bestc)
    Ab, Bb, Cb, _, _ = cands[bestc]
    lg = lg_of(V_A, V_B, V_C, V_p, V_q)
    sei = float(np.exp2(lg[np.isfinite(lg)] * 4 - 0).sum()) if np.isfinite(lg).any() else 0.0
    L = log2_sei(lg)
    k = step
    sei = float(np.exp2(L - 0)) if False else float(np.exp2(0))
    Nreq = 4 * k * math.log(2) / float(np.exp2(L))
    T = 2 * Nreq + k * 2 ** k
    mem = k * 2 ** k
    print('%2d %13.4f %10.4f %10.4f %10.4f' % (k, L, math.log2(Nreq), math.log2(T), math.log2(mem)))
    if best is None or T < best[0]:
        best = (T, k, L, list(chosen))
    print('      added A=%s B=%s C=%s' % (bytes(Ab).hex(), bytes(Bb).hex(), bytes(Cb).hex()))
print()
print('best k=%d  log2 T=%.4f  log2 SEI=%.4f' % (best[1], math.log2(best[0]), best[2]))
