"""Look for a better linear space than the single optimum: enumerate pairs and
triples drawn from a wide pool of the profile and report the capacity SEI of
each span.  Priced by Theorem 4 with v = k, d1 = k.
"""
import sys, os, io, math, contextlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    from verify import LAT, ILAT, L, Linv, sigma, pullback

MU = (4, 5, 6, 12, 13, 14)
POOL = 400

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
B = expand(A_basis, selB); C = expand(C_basis, selC)
P = expand(P_basis, selB); Q = expand(Q_basis, selC)

absL = np.abs(LAT.astype(np.int64)); absI = np.abs(ILAT.astype(np.int64))
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
    cb = (b ^ c).astype(np.int64); a = T12a[cb, c].astype(np.int64)
    Nrank *= (absL[p, q] // 4) * (absI[cb, a] // 4) * (absI[c, a] // 4)
Nrank[0, 0] = 0
flat = Nrank.astype(np.float64).ravel()
order = np.argsort(np.where((Nrank != 0).ravel(), -flat, np.inf))[:POOL]
bi, cj = np.unravel_index(order, Nrank.shape)

MA = np.zeros((POOL, 16), np.uint8); MB = np.zeros((POOL, 16), np.uint8); MC = np.zeros((POOL, 16), np.uint8)
MP = np.zeros((POOL, 16), np.uint8); MQ = np.zeros((POOL, 16), np.uint8)
for k in range(POOL):
    Bb = B[bi[k]].copy(); Cb = C[cj[k]].copy()
    Ab = np.zeros(16, dtype=np.uint8)
    for i in MU:
        Ab[i] = T12a[Cb[i] ^ Bb[i]][Cb[i]]
    MA[k], MB[k], MC[k], MP[k], MQ[k] = Ab, Bb, Cb, P[bi[k]], Q[cj[k]]

def lg_of(Ab, Bb, Cb, p, q):
    cb = (Cb ^ Bb).astype(np.int64); a = Ab.astype(np.int64)
    w1 = absL[p, q]; w2 = absI[cb, a]; w3 = absI[Cb.astype(np.int64), a]
    bad = (w1 == 0).any(-1) | (w2 == 0).any(-1) | (w3 == 0).any(-1)
    lg = (np.log2(np.maximum(w1, 1) / 256).sum(-1) + np.log2(np.maximum(w2, 1) / 256).sum(-1)
          + np.log2(np.maximum(w3, 1) / 256).sum(-1))
    lg[bad] = -np.inf
    ze = ~((Ab.astype(np.int64) | Bb.astype(np.int64) | Cb.astype(np.int64)).any(-1))
    lg[ze] = -np.inf
    return lg

def log2_sei(lg):
    fin = lg[np.isfinite(lg)]
    if fin.size == 0:
        return float('-inf')
    m = 4.0 * fin.max()
    return float(m + math.log2(np.exp2(4.0 * fin - m).sum()))

lg_single = lg_of(MA, MB, MC, MP, MQ)
print('single optimum  log2 tau = %.6f' % lg_single[0])
print('SEI(k=1)        %.4f' % log2_sei(lg_single[:1]))

# pairs
best = None
for i in range(POOL):
    for j in range(i + 1, POOL):
        Ab = np.stack([MA[i], MA[j], MA[i] ^ MA[j]])
        Bb = np.stack([MB[i], MB[j], MB[i] ^ MB[j]])
        Cb = np.stack([MC[i], MC[j], MC[i] ^ MC[j]])
        p = np.stack([MP[i], MP[j], MP[i] ^ MP[j]])
        q = np.stack([MQ[i], MQ[j], MQ[i] ^ MQ[j]])
        L = log2_sei(lg_of(Ab, Bb, Cb, p, q))
        if best is None or L > best[0]:
            best = (L, i, j)
print('best pair: SEI %.4f (i=%d j=%d)  vs single %.4f' % (best[0], best[1], best[2], log2_sei(lg_single[:1])))

# triples, extending the best pairs
best3 = None
for (i, j) in [(best[1], best[2])]:
    for kk in range(POOL):
        if kk in (i, j):
            continue
        idx = [i, j, kk]
        Z = np.zeros(16, dtype=np.uint8)
        combos = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
        sel = lambda Mx, a, b, c: (Mx[idx[0]] if a else Z) ^ (Mx[idx[1]] if b else Z) ^ (Mx[idx[2]] if c else Z)
        Ab = np.stack([sel(MA, *t) for t in combos])
        Bb = np.stack([sel(MB, *t) for t in combos])
        Cb = np.stack([sel(MC, *t) for t in combos])
        p = np.stack([sel(MP, *t) for t in combos])
        q = np.stack([sel(MQ, *t) for t in combos])
        L = log2_sei(lg_of(Ab, Bb, Cb, p, q))
        if best3 is None or L > best3[0]:
            best3 = (L, i, j, kk)
print('best triple: SEI %.4f (i=%d j=%d k=%d)' % (best3[0], best3[1], best3[2], best3[3]))
for L, k in ((log2_sei(lg_single[:1]), 1), (best[0], 2), (best3[0], 3)):
    Nreq = 4 * k * math.log(2) / float(np.exp2(L))
    T = 2 * Nreq + k * 2 ** k
    print('k=%d  log2 SEI=%8.4f  log2 N=%9.4f  log2 T=%9.4f' % (k, L, math.log2(Nreq), math.log2(T)))
