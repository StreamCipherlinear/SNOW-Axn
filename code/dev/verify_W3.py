# Hypothesis: W is spanned by the 26 masks of levels 0..8 (rank 24, two dependencies).
import sys, io, contextlib, math, time
import numpy as np
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
with contextlib.redirect_stdout(io.StringIO()):
    from verify import LAT, ILAT, L, Linv, sigma, pullback
MU = (4, 5, 6, 12, 13, 14)
t0 = time.time()

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
ILa = np.abs(ILAT.astype(np.int32))
T12a = np.zeros((256, 256), dtype=np.uint8); T12v = np.zeros((256, 256), dtype=np.int32)
for m1 in range(256):
    pr = ILa[m1][None, :] * ILa; pr[:, 0] = 0
    T12a[m1] = pr.argmax(axis=1).astype(np.uint8); T12v[m1] = pr.max(axis=1)
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

def triple(k):
    Bb = [int(x) for x in B[bi[k]]]; Cb = [int(x) for x in C[cj[k]]]
    Ab = [0] * 16
    for i in MU: Ab[i] = int(T12a[Cb[i] ^ Bb[i]][Cb[i]])
    return Ab, Bb, Cb
def gf2rank(vecs):
    piv = {}
    for v in vecs:
        while v:
            p = v.bit_length() - 1
            if p in piv: v ^= piv[p]
            else: piv[p] = v; break
    return len(piv)

for nset in (24, 26):
    TX = [triple(k) for k in range(nset)]
    V = [int.from_bytes(bytes(a) + bytes(b) + bytes(c), 'little') for (a, b, c) in TX]
    BC = [int.from_bytes(bytes(b) + bytes(c), 'little') for (a, b, c) in TX]
    print('top %d masks: rank384 = %d, rank(B,C) = %d' % (nset, gf2rank(V), gf2rank(BC)))

# build a basis of the span of the top 26 and enumerate it
TX26 = [triple(k) for k in range(26)]
V26 = [int.from_bytes(bytes(a) + bytes(b) + bytes(c), 'little') for (a, b, c) in TX26]
piv = {}
sel_idx = []
for k, v in enumerate(V26):
    w = v
    while w:
        p = w.bit_length() - 1
        if p in piv: w ^= piv[p]
        else:
            piv[p] = w; sel_idx.append(k); break
print('independent subset of the top 26: %d masks %s' % (len(sel_idx), sel_idx))
if len(sel_idx) != 24:
    print('unexpected rank'); sys.exit()

TXb = [TX26[k] for k in sel_idx]
Ab_ = np.array([t[0] for t in TXb], dtype=np.uint8)
Bb_ = np.array([t[1] for t in TXb], dtype=np.uint8)
Cb_ = np.array([t[2] for t in TXb], dtype=np.uint8)
Pb__ = np.array([[pullback(int.from_bytes(bytes(r), 'little'), Linv) >> (8 * i) & 0xff for i in range(16)] for r in Bb_], dtype=np.uint8)
Qb__ = np.array([[pullback(int.from_bytes(bytes(r), 'little'), lambda s: Linv(sigma(L(s)))) >> (8 * i) & 0xff for i in range(16)] for r in Cb_], dtype=np.uint8)

CH = 1 << 20
nz_count = 0; adm = []; adm_strict = 0
off = [i for i in range(16) if i not in MU]
for start in range(0, 1 << 24, CH):
    idx = np.arange(start, start + CH, dtype=np.int64)
    s = ((idx[:, None] >> np.arange(24)) & 1).astype(np.uint8)
    A = expand(Ab_, 24, s); Bc = expand(Bb_, 24, s); Cc = expand(Cb_, 24, s)
    Pp = expand(Pb__, 24, s); Qq = expand(Qb__, 24, s)
    lg = np.zeros(CH); zr = np.zeros(CH, dtype=bool)
    for l in range(16):
        v1 = absL[Pp[:, l], Qq[:, l]].astype(np.float64)
        v2 = absI[Cc[:, l] ^ Bc[:, l], A[:, l]].astype(np.float64)
        v3 = absI[Cc[:, l], A[:, l]].astype(np.float64)
        zr |= (v1 == 0) | (v2 == 0) | (v3 == 0)
        lg += np.log2(np.maximum(v1, 1) / 256) + np.log2(np.maximum(v2, 1) / 256) + np.log2(np.maximum(v3, 1) / 256)
    zr[0] = True
    nz_count += int((~zr).sum())
    ok = (Bc[:, off].sum(1) == 0) & (Cc[:, off].sum(1) == 0) & (Pp[:, off].sum(1) == 0) & (Qq[:, off].sum(1) == 0)
    ok &= (A[:, off].sum(1) == 0) & ~zr
    weak = np.ones(CH, dtype=bool); strict = np.ones(CH, dtype=bool)
    for i in MU:
        cbv = Cc[:, i] ^ Bc[:, i]
        weak &= (absI[cbv, A[:, i]].astype(np.int32) * absI[Cc[:, i], A[:, i]].astype(np.int32) == T12v[cbv, Cc[:, i]])
        strict &= (A[:, i] == T12a[cbv, Cc[:, i]])
    hit = ok & weak
    if hit.any(): adm.extend(float(x) for x in lg[hit])
    adm_strict += int((ok & strict).sum())
print('nonzero characters : %d   (paper 6,254,429)' % nz_count)
print('admissible (weak)  : %d   (paper 30)' % len(adm))
print('admissible (strict): %d' % adm_strict)
print('sum tau^6          : 2^%.3f   (paper 2^-327.203)' % math.log2(sum(2.0 ** (6 * x) for x in adm)))
print('price              : 2^%.3f   (paper 2^333.421)' % (6.218 - math.log2(sum(2.0 ** (6 * x) for x in adm))))
print('%.1fs' % (time.time() - t0))
