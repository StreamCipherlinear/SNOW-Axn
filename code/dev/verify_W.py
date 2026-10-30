# W = span of the 24 strongest triples: rank, (B,C) projection rank, the 30
# admissible characters, the 6,254,429 non-zero characters, and SEI(W).
import sys, io, contextlib, math, time
import numpy as np
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
with contextlib.redirect_stdout(io.StringIO()):
    from verify import LAT, ILAT, L, Linv, sigma, pullback
MU = (4, 5, 6, 12, 13, 14)
t0 = time.time()

# ---- rebuild the profile just enough to get the 24 strongest ----
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
def expand(basis, k, sel):
    return np.bitwise_xor.reduce(sel[:, :, None] * basis[None, :, :], axis=1)
B = expand(arr(tA), 16, ((np.arange(1 << 16)[:, None] >> np.arange(16)) & 1).astype(np.uint8))
P = expand(arr(Pb_), 16, ((np.arange(1 << 16)[:, None] >> np.arange(16)) & 1).astype(np.uint8))
C = expand(arr(tC), 8, ((np.arange(1 << 8)[:, None] >> np.arange(8)) & 1).astype(np.uint8))
Q = expand(arr(Qb_), 8, ((np.arange(1 << 8)[:, None] >> np.arange(8)) & 1).astype(np.uint8))
absL = np.abs(LAT.astype(np.int16)); absI = np.abs(ILAT.astype(np.int16))
ILa = np.abs(ILAT.astype(np.int32))
T12a = np.zeros((256, 256), dtype=np.uint8)
for m1 in range(256):
    pr = ILa[m1][None, :] * ILa; pr[:, 0] = 0
    T12a[m1] = pr.argmax(axis=1).astype(np.uint8)
N = np.ones((len(B), len(C)), dtype=np.int64)
for i in MU:
    b = B[:, i][:, None].astype(np.int32); c = C[:, i][None, :].astype(np.int32)
    p = P[:, i][:, None].astype(np.int32); q = Q[:, i][None, :].astype(np.int32)
    cb = (b ^ c).astype(np.int32); a = T12a[cb, c].astype(np.int32)
    N *= (absL[p, q] // 4).astype(np.int64) * (absI[cb, a] // 4).astype(np.int64) * (absI[c, a] // 4).astype(np.int64)
N[0, 0] = 0
flat = N.astype(np.float64).ravel()
order = np.argsort(np.where((N != 0).ravel(), -flat, np.inf))[:24]
bi, cj = np.unravel_index(order, N.shape)
print('24 strongest located (%.1fs)' % (time.time() - t0))

triples = []
for k in range(24):
    Bb = [int(x) for x in B[bi[k]]]; Cb = [int(x) for x in C[cj[k]]]
    Ab = [0] * 16
    for i in MU: Ab[i] = int(T12a[Cb[i] ^ Bb[i]][Cb[i]])
    triples.append((Ab, Bb, Cb))
Ab_ = np.array([t[0] for t in triples], dtype=np.uint8)
Bb_ = np.array([t[1] for t in triples], dtype=np.uint8)
Cb_ = np.array([t[2] for t in triples], dtype=np.uint8)
Pb__ = np.array([[pullback(int.from_bytes(bytes(row), 'little'), Linv) >> (8 * i) & 0xff for i in range(16)] for row in Bb_], dtype=np.uint8)
Qb__ = np.array([[pullback(int.from_bytes(bytes(row), 'little'), lambda s: Linv(sigma(L(s)))) >> (8 * i) & 0xff for i in range(16)] for row in Cb_], dtype=np.uint8)

def gf2rank(vecs):
    piv = {}
    for v in vecs:
        while v:
            p = v.bit_length() - 1
            if p in piv: v ^= piv[p]
            else: piv[p] = v; break
    return len(piv)
V = [int.from_bytes(bytes(Ab_[k]) + bytes(Bb_[k]) + bytes(Cb_[k]), 'little') for k in range(24)]
BC = [int.from_bytes(bytes(Bb_[k]) + bytes(Cb_[k]), 'little') for k in range(24)]
print('rank of the 24 triples      : %d   (paper: 24, linearly independent)' % gf2rank(V))
print('rank of the (B,C) projection: %d   (paper: 20)' % gf2rank(BC))

# ---- enumerate W ----
CH = 1 << 20
outS = 0.0            # sum tau^6 over admissible
nz_count = 0
adm = []
for start in range(0, 1 << 24, CH):
    idx = np.arange(start, start + CH, dtype=np.int64)
    sel = ((idx[:, None] >> np.arange(24)) & 1).astype(np.uint8)
    A = expand(Ab_, 24, sel); Bc = expand(Bb_, 24, sel); Cc = expand(Cb_, 24, sel)
    Pp = expand(Pb__, 24, sel); Qq = expand(Qb__, 24, sel)
    lg = np.zeros(CH); zr = np.zeros(CH, dtype=bool)
    for l in range(16):
        v1 = absL[Pp[:, l], Qq[:, l]].astype(np.float64)
        v2 = absI[Cc[:, l] ^ Bc[:, l], A[:, l]].astype(np.float64)
        v3 = absI[Cc[:, l], A[:, l]].astype(np.float64)
        zr |= (v1 == 0) | (v2 == 0) | (v3 == 0)
        lg += np.log2(np.maximum(v1, 1) / 256) + np.log2(np.maximum(v2, 1) / 256) + np.log2(np.maximum(v3, 1) / 256)
    lg[start == 0] = -np.inf          # drop the zero vector
    zr[0] = True
    nz_count += int((~zr).sum())
    # admissibility: B,C supported on MU, P,Q vanish off MU, A = A* on MU and 0 off
    off = [i for i in range(16) if i not in MU]
    ok = (Bc[:, off].sum(1) == 0) & (Cc[:, off].sum(1) == 0) & (Pp[:, off].sum(1) == 0) & (Qq[:, off].sum(1) == 0)
    ok &= (A[:, off].sum(1) == 0)
    star = np.ones(CH, dtype=bool)
    for i in MU:
        star &= (A[:, i] == T12a[Cc[:, i] ^ Bc[:, i], Cc[:, i]])
    ok &= star & ~zr
    if ok.any():
        for k in np.nonzero(ok)[0]:
            adm.append(float(lg[k]))
print('\nnonzero characters of W : %d    (paper 6,254,429)' % nz_count)
print('admissible in W         : %d    (paper 30)' % len(adm))
s6 = sum(2.0 ** (6.0 * x) for x in adm)
print('log2 sum tau^6 over them: %.3f  (paper -327.203)' % math.log2(s6))
print('price at factor 2^6.218 : 2^%.3f  (paper 2^333.421)' % (6.218 - math.log2(s6)))
print('%.1fs' % (time.time() - t0))
