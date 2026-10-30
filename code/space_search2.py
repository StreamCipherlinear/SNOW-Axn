# Search nested spans of the strongest three-word approximations for the
# linear space with the lowest state-recovery cost.  p = B.L^-1 and q = C.L^-1
# sigma^-1 L are linear in B and C, so they are taken from the basis instead of
# being recomputed for every combination.
import sys, os, math, io, contextlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    from verify import LAT, ILAT, L, Linv, sigma, pullback
MU = (4, 5, 6, 12, 13, 14)

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
def arr(v): return np.array([[x >> (8 * i) & 0xff for i in range(16)] for x in v], dtype=np.uint8)
def expand(basis, sel): return np.bitwise_xor.reduce(sel[:, :, None] * basis[None, :, :], axis=1)
selB = ((np.arange(1 << 16)[:, None] >> np.arange(16)) & 1).astype(np.uint8)
selC = ((np.arange(1 << 8)[:, None] >> np.arange(8)) & 1).astype(np.uint8)
B = expand(arr(tA), selB); C = expand(arr(tC), selC)
absL = np.abs(LAT.astype(np.int16)); absI = np.abs(ILAT.astype(np.int16))
ILa = np.abs(ILAT.astype(np.int32))
T12a = np.zeros((256, 256), dtype=np.uint8)
for m1 in range(256):
    pr = ILa[m1][None, :] * ILa
    pr[:, 0] = 0
    T12a[m1] = pr.argmax(axis=1).astype(np.uint8)
Pb_ = arr([pullback(int(x), Linv) for x in tA])
Qb_ = arr([pullback(int(x), lambda s: Linv(sigma(L(s)))) for x in tC])
P = expand(Pb_, selB)
Q = expand(Qb_, selC)
N = np.ones((len(B), len(C)), dtype=np.int64)
for i in MU:
    b = B[:, i][:, None].astype(np.int32); c = C[:, i][None, :].astype(np.int32)
    p = P[:, i][:, None].astype(np.int32); q = Q[:, i][None, :].astype(np.int32)
    cb = (b ^ c).astype(np.int32); a = T12a[cb, c].astype(np.int32)
    N *= (absL[p, q] // 4).astype(np.int64) * (absI[cb, a] // 4).astype(np.int64) * (absI[c, a] // 4).astype(np.int64)
N[0, 0] = 0
flat = N.astype(np.float64).ravel()
order = np.argsort(np.where((N != 0).ravel(), -flat, np.inf))[:200]
bi, cj = np.unravel_index(order, N.shape)

base = []
for k in range(len(order)):
    Bb = B[bi[k]].copy(); Cb = C[cj[k]].copy()
    Ab = np.zeros(16, dtype=np.uint8)
    for i in MU: Ab[i] = T12a[Cb[i] ^ Bb[i]][Cb[i]]
    base.append((Ab, Bb, Cb))

def to_arr128(v):
    return np.array([[int(x) >> (8 * i) & 0xff for i in range(16)] for x in v], dtype=np.uint8)
Pb = np.array([to_arr128([pullback(int.from_bytes(bytes(b), 'little'), Linv) for b in base[k][1]])[0] for k in range(len(base))], dtype=np.uint8)
Qb = np.array([to_arr128([pullback(int.from_bytes(bytes(c), 'little'), lambda s: Linv(sigma(L(s)))) for c in base[k][2]])[0] for k in range(len(base))], dtype=np.uint8)

print('%3s %11s %9s %9s %9s' % ('k', 'log2 SEI', 'log2 N', 'log2 T', 'log2 mem'))
best = None
for k in range(1, 17):
    A_ = np.array([base[i][0] for i in range(k)])
    B_ = np.array([base[i][1] for i in range(k)])
    C_ = np.array([base[i][2] for i in range(k)])
    P_ = Pb[:k].copy(); Q_ = Qb[:k].copy()
    idx = np.arange(1, 1 << k, dtype=np.int64)
    sel = ((idx[:, None] >> np.arange(k)) & 1).astype(np.uint8)
    Ab = np.bitwise_xor.reduce(sel[:, :, None] * A_[None, :, :], axis=1)
    Bb = np.bitwise_xor.reduce(sel[:, :, None] * B_[None, :, :], axis=1)
    Cb = np.bitwise_xor.reduce(sel[:, :, None] * C_[None, :, :], axis=1)
    Pp = np.bitwise_xor.reduce(sel[:, :, None] * P_[None, :, :], axis=1)
    Qq = np.bitwise_xor.reduce(sel[:, :, None] * Q_[None, :, :], axis=1)
    v1 = absL[Pp, Qq].astype(np.float64)
    v2 = absI[Cb ^ Bb, Ab].astype(np.float64)
    v3 = absI[Cb, Ab].astype(np.float64)
    lg = (np.log2(np.maximum(v1, 1) / 256) + np.log2(np.maximum(v2, 1) / 256) + np.log2(np.maximum(v3, 1) / 256)).sum(axis=1)
    sei = float(np.exp2(4.0 * lg).sum())
    Nreq = 4 * k * math.log(2) / sei
    T = 2 * Nreq + k * 2 ** k
    mem = k * 2 ** k
    print('%3d %11.4f %9.4f %9.4f %9.4f' % (k, math.log2(sei), math.log2(Nreq), math.log2(T), math.log2(mem)))
    if best is None or T < best[0]:
        best = (T, k, sei)
print('\nbest: k=%d, log2 T=%.4f, log2 SEI=%.4f' % (best[1], math.log2(best[0]), math.log2(best[2])))
