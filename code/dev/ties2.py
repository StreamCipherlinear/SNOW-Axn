"""Count tied bytewise optima and locate them among the 24 strongest masks.

Inputs: none; the profile is rebuilt from verify.py.
Outputs: the number of tied (m1,m2) pairs and the tied bytes of three selections.
Paper object: Section 4.2; development aid for the tie-breaking of the basis.
"""

import sys, io, contextlib
import numpy as np
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
with contextlib.redirect_stdout(io.StringIO()):
    from verify import LAT, ILAT, L, Linv, sigma, pullback
MU = (4, 5, 6, 12, 13, 14)
ILa = np.abs(ILAT.astype(np.int32))
T12v = np.zeros((256, 256), dtype=np.int32); T12n = np.zeros((256, 256), dtype=np.int32)
T12a = np.zeros((256, 256), dtype=np.uint8)
for m1 in range(256):
    pr = ILa[m1][None, :] * ILa
    pr[:, 0] = 0
    T12v[m1] = pr.max(axis=1)
    T12n[m1] = (pr == T12v[m1][:, None]).sum(axis=1)
    T12a[m1] = pr.argmax(axis=1).astype(np.uint8)
print('(m1,m2) with a tied maximum: %d / 65536' % int(((T12n > 1) & (T12v > 0)).sum()))

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
B = expand(arr(tA), 16, ((np.arange(1 << 16)[:, None] >> np.arange(16)) & 1).astype(np.uint8))
P = expand(arr(Pb_), 16, ((np.arange(1 << 16)[:, None] >> np.arange(16)) & 1).astype(np.uint8))
C = expand(arr(tC), 8, ((np.arange(1 << 8)[:, None] >> np.arange(8)) & 1).astype(np.uint8))
Q = expand(arr(Qb_), 8, ((np.arange(1 << 8)[:, None] >> np.arange(8)) & 1).astype(np.uint8))
absL = np.abs(LAT.astype(np.int16)); absI = np.abs(ILAT.astype(np.int16))
N = np.ones((len(B), len(C)), dtype=np.int64)
for i in MU:
    b = B[:, i][:, None].astype(np.int32); c = C[:, i][None, :].astype(np.int32)
    p = P[:, i][:, None].astype(np.int32); q = Q[:, i][None, :].astype(np.int32)
    cb = (b ^ c).astype(np.int32); a = T12a[cb, c].astype(np.int32)
    N *= (absL[p, q] // 4).astype(np.int64) * (absI[cb, a] // 4).astype(np.int64) * (absI[c, a] // 4).astype(np.int64)
N[0, 0] = 0
flat = N.astype(np.float64).ravel()
order = np.argsort(np.where((N != 0).ravel(), -flat, np.inf))
bi, cj = np.unravel_index(order, N.shape)

for label, selk in (('22 + (22,23)', list(range(24))), ('22 + (22,24)', list(range(22)) + [22, 24]),
                    ('22 + (23,25)', list(range(22)) + [23, 25])):
    ties = []
    for k in selk:
        Bb = [int(x) for x in B[bi[k]]]; Cb = [int(x) for x in C[cj[k]]]
        for i in MU:
            n = int(T12n[Cb[i] ^ Bb[i], Cb[i]])
            if n > 1: ties.append((k, i, n))
    print('%-14s tied active bytes: %d %s' % (label, len(ties), ties[:6]))
