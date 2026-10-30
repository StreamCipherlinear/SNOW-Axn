# Exact-integer profile (tau = N / 2^108 with N = prod of the 18 m-values, N <= 2^54),
# plus the top-24 correlation matrix and the 2000-level cumulative fraction.
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
Pb = [pullback(b, Linv) for b in tA]
Qb = [pullback(c, lambda s: Linv(sigma(L(s)))) for c in tC]
def arr(vecs): return np.array([[v >> (8 * i) & 0xff for i in range(16)] for v in vecs], dtype=np.uint8)
def expand(basis, k):
    sel = ((np.arange(1 << k)[:, None] >> np.arange(k)) & 1).astype(np.uint8)
    return np.bitwise_xor.reduce(sel[:, :, None] * basis[None, :, :], axis=1)
B = expand(arr(tA), 16); P = expand(arr(Pb), 16)
C = expand(arr(tC), 8);  Q = expand(arr(Qb), 8)

LATi = LAT.astype(np.int16); ILATi = ILAT.astype(np.int16)
absL = np.abs(LATi); absI = np.abs(ILATi)
ILa = np.abs(ILATi.astype(np.int32))
T12a = np.zeros((256, 256), dtype=np.uint8)
for m1 in range(256):
    prod = ILa[m1][None, :] * ILa
    prod[:, 0] = 0
    T12a[m1] = prod.argmax(axis=1).astype(np.uint8)

N = np.ones((len(B), len(C)), dtype=np.int64)     # prod of the 18 m-values
for i in MU:
    b = B[:, i][:, None].astype(np.int32); c = C[:, i][None, :].astype(np.int32)
    p = P[:, i][:, None].astype(np.int32); q = Q[:, i][None, :].astype(np.int32)
    cb = (b ^ c).astype(np.int32)
    a = T12a[cb, c].astype(np.int32)
    m1 = (absL[p, q] // 4).astype(np.int64)
    m2 = (absI[cb, a] // 4).astype(np.int64)
    m3 = (absI[c, a] // 4).astype(np.int64)
    N *= m1 * m2 * m3
N[0, 0] = 0                                        # trivial triple
print('exact enumeration %.1fs   nonzero = %d' % (time.time() - t0, int((N > 0).sum())))

nz = N[N > 0]
print('\n=== exact profile (tau = N/2^108) ===')
print('  #nonzero        : %d     (paper 11,093,111)' % nz.size)
print('  max  N          : %d  -> log2|tau| = %.4f, mult %d   (paper -54.771, 1)'
      % (nz.max(), math.log2(nz.max()) - 108, int((N == nz.max()).sum())))
print('  min  N          : %d  -> log2|tau| = %.4f           (paper -73.660)'
      % (nz.min(), math.log2(nz.min()) - 108))
u, cnt = np.unique(nz, return_counts=True)
print('  #levels (exact) : %d     (paper 3,295)' % u.size)
cum = np.cumsum(cnt[::-1])          # descending N
uu = u[::-1]
print('  E|tau|          : log2 = %.4f                      (paper -61.888)'
      % (math.log2(nz.mean()) - 108))
print('  sum tau^2       : log2 = %.4f                      (paper -97.938)'
      % (math.log2(float(np.sum((nz.astype(np.float64)) ** 2))) - 216))
print('  sum tau^6       : log2 = %.4f                      (paper -326.187)'
      % (math.log2(float(np.sum((nz.astype(np.float64)) ** 6))) - 648))
for L_ in (999, 1000, 1999, 2000):
    print('  cumulative at level %4d: %.3f%%   (paper: 2,000 -> 43.317%%)'
          % (L_, 100.0 * cum[min(L_, cum.size - 1)] / nz.size))

# ---------- top 24 ----------
flat = N.astype(np.float64).ravel()
mask = ~((N == 0).ravel())
order = np.argsort(np.where(mask, -flat, np.inf))[:24]
bi, cj = np.unravel_index(order, N.shape)
tau6 = float(np.sum((N[bi, cj].astype(np.float64)) ** 6)) * 2.0 ** -648
print('\n=== the 24 strongest ===')
print('  log2 values     : %s ...' % np.round(np.log2(N[bi, cj]) - 108, 3)[:8])
print('  sum tau^6       : 2^%.3f   (paper 2^-327.214)' % math.log2(tau6))
lv24 = np.log2(N[bi, cj].astype(np.float64)) - 108
pr = [lv24[a] + lv24[b] for a in range(24) for b in range(24) if a != b]
print('  pairwise products: 2^%.3f .. 2^%.3f   (paper 2^-111.527 .. 2^-109.926)'
      % (min(pr), max(pr)))

def triple(k):
    Bb = [int(x) for x in B[bi[k]]]; Cb = [int(x) for x in C[cj[k]]]
    Ab = [0] * 16
    for i in MU:
        Ab[i] = int(T12a[Cb[i] ^ Bb[i]][Cb[i]])
    return Ab, Bb, Cb, [int(x) for x in P[bi[k]]], [int(x) for x in Q[cj[k]]]

TX = [triple(k) for k in range(24)]
tau = 2.0 ** lv24
Cmat = np.eye(24)
for a in range(24):
    Aa, Ba, Ca, Pa, Qa = TX[a]
    for b in range(a + 1, 24):
        Ax, Bb, Cb, Pb, Qb = TX[b]
        prod = 1.0
        for l in range(16):
            prod *= (LAT[Aa[l] ^ Ax[l]][(Ba[l] ^ Ca[l]) ^ (Bb[l] ^ Cb[l])] / 256.0)
            prod *= (LAT[Aa[l] ^ Ax[l]][Ca[l] ^ Cb[l]] / 256.0)
            prod *= (LAT[Pa[l] ^ Pb[l]][Qa[l] ^ Qb[l]] / 256.0)
        Cmat[a, b] = Cmat[b, a] = prod
off = np.abs(Cmat - np.eye(24))
pos = np.unravel_index(np.argmax(off), off.shape)
print('  max |C_ij|      : %.4e = 2^%.3f at (%d,%d)  (paper 6.766e-21 = 2^-67.002 at (3,6))'
      % (off.max(), math.log2(off.max()), pos[0], pos[1]))
ev = np.linalg.eigvalsh(Cmat)
ev = np.linalg.eigvalsh(Cmat - np.eye(24))
print('  rho(C-I) via eig(C-I)  : 2^%.3f   (paper 2^-66.991)' % math.log2(max(abs(ev))))
rows = np.abs(Cmat - np.eye(24)).sum(axis=1)
print('  max row sum ||C-I||_inf: %.4e = 2^%.3f   (paper 6.820e-21 = 2^-66.991)' % (rows.max(), math.log2(rows.max())))
SEI = float(tau @ np.linalg.solve(Cmat, tau)); s2 = float(np.sum(tau ** 2))
print('  relative |SEI - sum tau^2| / sum tau^2 : 2^%.3f   (paper 2^-52.000)'
      % math.log2(abs(SEI - s2) / s2))
print('  log2 SEI        : %.3f  (paper ~ -106.272 for m=24)' % math.log2(SEI))
print('\n%.1fs' % (time.time() - t0))



