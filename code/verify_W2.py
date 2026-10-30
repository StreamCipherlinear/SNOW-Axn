"""Enumerate the span of a 22 + 2 selection of the strongest masks.

Inputs: two indices on the command line, e.g. 22,24.
Outputs: rank, non-zero characters, admissible counts and tau^6 sums.
Paper object: Section 4.2; development aid, superseded by verify_list.py.
"""

import sys, io, contextlib, math, time
import numpy as np
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
with contextlib.redirect_stdout(io.StringIO()):
    from verify import LAT, ILAT, L, Linv, sigma, pullback
MU = (4, 5, 6, 12, 13, 14)
PAIR = tuple(int(x) for x in sys.argv[1].split(','))
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
sel24 = list(range(22)) + list(PAIR)

TX = []
for k in sel24:
    Bb = [int(x) for x in B[bi[k]]]; Cb = [int(x) for x in C[cj[k]]]
    Ab = [0] * 16
    for i in MU: Ab[i] = int(T12a[Cb[i] ^ Bb[i]][Cb[i]])
    TX.append((Ab, Bb, Cb))
Ab_ = np.array([t[0] for t in TX], dtype=np.uint8)
Bb_ = np.array([t[1] for t in TX], dtype=np.uint8)
Cb_ = np.array([t[2] for t in TX], dtype=np.uint8)
Pb__ = np.array([[pullback(int.from_bytes(bytes(r), 'little'), Linv) >> (8 * i) & 0xff for i in range(16)] for r in Bb_], dtype=np.uint8)
Qb__ = np.array([[pullback(int.from_bytes(bytes(r), 'little'), lambda s: Linv(sigma(L(s)))) >> (8 * i) & 0xff for i in range(16)] for r in Cb_], dtype=np.uint8)

# ---- fixed list of 24 complete triples and its rank -------------------------
pivots = {}
for k, (A_, B_, C_) in enumerate(TX):
    print("T%02d A=%s B=%s C=%s" % (k, bytes(A_).hex(), bytes(B_).hex(), bytes(C_).hex()))
    v = int.from_bytes(bytes(A_) + bytes(B_) + bytes(C_), "little")
    while v:
        q = v.bit_length() - 1
        if q in pivots:
            v ^= pivots[q]
        else:
            pivots[q] = v
            break
print("rank of complete triples =", len(pivots))

CH = 1 << 20
nz_count = 0
six_total = 0.0
adm_strict = []
adm_weak = []
adm_base = []
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
    if start == 0:
        zr[0] = True
    nz_count += int((~zr).sum())
    six_total += float(np.exp2(6.0 * lg[~zr]).sum())
    ok = (Bc[:, off].sum(1) == 0) & (Cc[:, off].sum(1) == 0) & (Pp[:, off].sum(1) == 0) & (Qq[:, off].sum(1) == 0)
    ok &= (A[:, off].sum(1) == 0) & ~zr
    star_strict = np.ones(CH, dtype=bool); star_weak = np.ones(CH, dtype=bool)
    for i in MU:
        cbv = Cc[:, i] ^ Bc[:, i]
        star_strict &= (A[:, i] == T12a[cbv, Cc[:, i]])
        star_weak &= (absI[cbv, A[:, i]].astype(np.int32) * absI[Cc[:, i], A[:, i]].astype(np.int32)
                      == T12v[cbv, Cc[:, i]])
    for arr_, mask, store in ((A, star_strict, adm_strict), (A, star_weak, adm_weak)):
        hit = ok & mask
        if hit.any():
            store.extend(float(x) for x in lg[hit])
    base = ((Bc[:, off].sum(1) == 0) & (Cc[:, off].sum(1) == 0) &
            (Pp[:, off].sum(1) == 0) & (Qq[:, off].sum(1) == 0) & (A[:, off].sum(1) == 0) & ~zr)
    if base.any():
        adm_base.extend(float(x) for x in lg[base])
print('pair %s' % (PAIR,))
print('  masks with nonzero correlation : %d   (expected for fixed T00-T23: 6,138,951)' % nz_count)
print('  admissible (A == A*): %d  -> sum tau^6 = 2^%.3f' %
      (len(adm_strict), math.log2(sum(2.0 ** (6 * x) for x in adm_strict)) if adm_strict else float('nan')))
print('  admissible (tie-free): %d  -> sum tau^6 = 2^%.3f   (expected 29)' %
      (len(adm_weak), math.log2(sum(2.0 ** (6 * x) for x in adm_weak)) if adm_weak else float('nan')))
print('  price = 2^%.3f   (paper 2^333.421)' %
      (6.218 - math.log2(sum(2.0 ** (6 * x) for x in adm_weak))))
print('  admissible (class only): %d  -> sum tau^6 = 2^%.3f' % (len(adm_base), math.log2(sum(2.0 ** (6 * x) for x in adm_base))))
print('  %.1fs' % (time.time() - t0))
print('FULL six-power sum over W: log2 =', __import__('math').log2(six_total))
