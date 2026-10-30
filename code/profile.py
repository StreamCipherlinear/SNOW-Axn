# Independent recomputation of Theorem 1: the full correlation profile of the
# optimal activity pattern, by exhaustive enumeration of Theta_mu = B_mu x C_mu.
import sys, io, contextlib, time
import numpy as np
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
with contextlib.redirect_stdout(io.StringIO()):
    from verify import LAT, ILAT, L, Linv, sigma, pullback, to_bytes, to_word

MU = (4, 5, 6, 12, 13, 14)
t0 = time.time()

def nullspace(rows, n):
    rows = list(rows); piv = {}
    r = 0
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
        piv[col] = r
        r += 1
        if r == len(rows):
            break
    free = [c for c in range(n) if c not in piv]
    out = []
    for f in free:
        v = 1 << f
        for c, ri in piv.items():
            if (rows[ri] >> f) & 1:
                v |= 1 << c
        out.append(v)
    return out, piv

def admissible_basis(mu, f):
    """basis (as 128-bit masks) of { m : supp(m) <= mu, f(m) vanishes off mu }"""
    free = [(b, t) for b in mu for t in range(8)]
    outc = [b for b in range(16) if b not in mu]
    n = len(free)
    img = [f(1 << (8 * b + t)) for (b, t) in free]
    rows = []
    for ob in outc:
        for ot in range(8):
            row = 0
            for j, w in enumerate(img):
                if (w >> (8 * ob + ot)) & 1:
                    row |= 1 << j
            rows.append(row)
    ker, piv = nullspace(rows, n)
    basis = []
    for v in ker:
        mask = 0
        for j, (b, t) in enumerate(free):
            if (v >> j) & 1:
                mask |= 1 << (8 * b + t)
        basis.append(mask)
    return basis, piv

tA, _ = admissible_basis(MU, lambda w: pullback(w, Linv))
tC, _ = admissible_basis(MU, lambda w: pullback(w, lambda s: Linv(sigma(L(s)))))
print('dim B_mu = %d, dim C_mu = %d  ->  |Theta| = 2^%d' % (len(tA), len(tC), len(tA) + len(tC)))

P_basis = [pullback(b, Linv) for b in tA]
Q_basis = [pullback(c, lambda s: Linv(sigma(L(s)))) for c in tC]
basisB = np.array([[v >> (8 * i) & 0xff for i in range(16)] for v in tA], dtype=np.uint8)
basisP = np.array([[v >> (8 * i) & 0xff for i in range(16)] for v in P_basis], dtype=np.uint8)
basisC = np.array([[v >> (8 * i) & 0xff for i in range(16)] for v in tC], dtype=np.uint8)
basisQ = np.array([[v >> (8 * i) & 0xff for i in range(16)] for v in Q_basis], dtype=np.uint8)

def expand(basis, k):
    sel = ((np.arange(1 << k)[:, None] >> np.arange(k)) & 1).astype(np.uint8)
    contrib = sel[:, :, None] * basis[None, :, :]
    return np.bitwise_xor.reduce(contrib, axis=1)      # (2^k, 16) bytes

B = expand(basisB, len(tA)); P = expand(basisP, len(tA))
C = expand(basisC, len(tC)); Q = expand(basisQ, len(tC))
print('expanded in %.1fs: B%s P%s C%s Q%s' % (time.time() - t0, B.shape, P.shape, C.shape, Q.shape))

LATi = LAT.astype(np.int16)
ILATi = ILAT.astype(np.int16)
# T12[m1][m2] = max_{u!=0} |ILAT[m1][u]| * |ILAT[m2][u]|, with its argmax
T12v = np.zeros((256, 256), dtype=np.int32)
T12a = np.zeros((256, 256), dtype=np.uint8)
IL = np.abs(ILATi.astype(np.int32))
for m1 in range(256):
    prod = IL[m1][None, :] * IL              # (256 m2, 256 u)
    prod[:, 0] = 0                            # u = 0 excluded
    T12a[m1] = prod.argmax(axis=1).astype(np.uint8)
    T12v[m1] = prod.max(axis=1)

acc = np.zeros((len(B), len(C)), dtype=np.float64)
zero = np.zeros((len(B), len(C)), dtype=bool)
for i in MU:
    b = B[:, i][:, None].astype(np.int32)
    c = C[:, i][None, :].astype(np.int32)
    p = P[:, i][:, None].astype(np.int32)
    q = Q[:, i][None, :].astype(np.int32)
    v1 = np.abs(LATi[p, q]).astype(np.int32)
    cb = (b ^ c).astype(np.int32)
    a = T12a[cb, c].astype(np.int32)
    v2 = np.abs(ILATi[cb, a]).astype(np.int32)
    v3 = np.abs(ILATi[c, a]).astype(np.int32)
    zero |= (v1 == 0) | (v2 == 0) | (v3 == 0)
    with np.errstate(divide='ignore'):
        acc += np.log2(np.maximum(v1, 1)) - 8 + np.log2(np.maximum(v2, 1)) - 8 + np.log2(np.maximum(v3, 1)) - 8
    del v1, v2, v3, cb, a
print('enumerated in %.1fs' % (time.time() - t0))

zero[0, 0] = True   # exclude the trivial triple (A,B,C) = (0,0,0), which has tau = 1
nz = ~zero
vals = acc[nz]
print('\ntotal points        : %d' % acc.size)
print('nonzero points      : %d   (paper: 11,093,111)' % vals.size)
print('max  log2|tau|      : %.4f  multiplicity %d  (paper: -54.771, 1)'
      % (vals.max(), int((np.abs(vals - vals.max()) < 5e-6).sum())))
print('min  log2|tau|      : %.4f  (paper: -73.660)' % vals.min())
lv = np.unique(np.round(vals, 5))
print('distinct levels     : %d   (paper: 3,295)' % lv.size)
print('log2 E|tau|         : %.4f  (paper: -61.888)' % (np.log2(np.mean(2.0 ** vals))))
s = np.sum(2.0 ** (2.0 * vals))
print('log2 sum tau^2      : %.4f  (paper: -97.938)' % np.log2(s))
print('log2(sum/max^2)     : %.4f  (paper: 11.603)' % np.log2(s / (2.0 ** (2.0 * vals.max()))))
print('tau_eff = sqrt(SEI) : 2^%.3f  (paper: 2^-48.969)' % (0.5 * np.log2(s)))
print('\ntotal %.1fs' % (time.time() - t0))

