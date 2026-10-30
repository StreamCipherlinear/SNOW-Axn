# Verify Section 5 / Table 5 / Table 7 claims against the exhaustive profile.
import sys, io, contextlib, math, time
import numpy as np
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
with contextlib.redirect_stdout(io.StringIO()):
    from verify import LAT, ILAT, L, Linv, sigma, pullback

MU = (4, 5, 6, 12, 13, 14)
t0 = time.time()

# ---------- rebuild Theta_mu and the profile (same code as profile.py) ----------
def nullspace(rows, n):
    rows = list(rows); piv = {}; r = 0
    for col in range(n):
        idx = None
        for i in range(r, len(rows)):
            if (rows[i] >> col) & 1:
                idx = i; break
        if idx is None: continue
        rows[r], rows[idx] = rows[idx], rows[r]
        for i in range(len(rows)):
            if i != r and ((rows[i] >> col) & 1):
                rows[i] ^= rows[r]
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
ILa = np.abs(ILATi.astype(np.int32))
T12a = np.zeros((256, 256), dtype=np.uint8)
for m1 in range(256):
    prod = ILa[m1][None, :] * ILa
    prod[:, 0] = 0
    T12a[m1] = prod.argmax(axis=1).astype(np.uint8)

acc = np.zeros((len(B), len(C))); zero = np.zeros((len(B), len(C)), dtype=bool)
for i in MU:
    b = B[:, i][:, None].astype(np.int32); c = C[:, i][None, :].astype(np.int32)
    p = P[:, i][:, None].astype(np.int32); q = Q[:, i][None, :].astype(np.int32)
    v1 = np.abs(LATi[p, q]).astype(np.int32)
    cb = (b ^ c).astype(np.int32)
    a = T12a[cb, c].astype(np.int32)
    v2 = np.abs(ILATi[cb, a]).astype(np.int32)
    v3 = np.abs(ILATi[c, a]).astype(np.int32)
    zero |= (v1 == 0) | (v2 == 0) | (v3 == 0)
    acc += np.log2(np.maximum(v1, 1)) - 24 + np.log2(np.maximum(v2, 1)) + np.log2(np.maximum(v3, 1))
zero[0, 0] = True
print('enumeration ready in %.1fs, nonzero = %d' % (time.time() - t0, int((~zero).sum())))

vals = acc[~zero]
print('\n=== Table 5 head (levels of log2|tau|) ===')
order = np.argsort(-vals)
lv, cnt = np.unique(np.round(vals, 4), return_counts=True)
lv = lv[::-1]; cnt = cnt[::-1]
cum = np.cumsum(cnt)
print(' level  log2|tau|  mult  cumulative   paper')
paper = [(0, -54.771, 1, 1), (1, -55.156, 5, 6), (2, -55.215, 1, 7), (5, -55.601, 8, 19),
         (7, -55.741, 1, 22), (14, -55.986, 40, 85), (29, -56.279, 14, 239),
         (52, -56.572, 15, 876), (437, -58.424, 354, 91069), (3294, -73.660, 10, 11093111)]
for (k, v, c, cu) in paper:
    print('  #%-5d %-9.3f %-5d %-9d   #%d %.3f %d %d   %s'
          % (k, lv[k], cnt[k], cum[k], k, v, c, cu,
             'OK' if abs(lv[k] - v) < 2e-3 and cnt[k] == c and cum[k] == cu else 'MISMATCH'))
print('  cumulative fraction at level 2000: %.3f%%   (paper 43.317%%)' % (100.0 * cum[2000] / len(vals)))
print('  cumulative fraction at level 1000: %.3f%%' % (100.0 * cum[1000] / len(vals)))

# ---------- capacities ----------
def lg(x): return math.log2(x)
print('\n=== distinguishing capacities ===')
print('  single best mask   : tau^6 = 2^%.3f   (paper 2^-328.623)' % (6 * vals.max()))
K = int((~zero).sum())
s6 = np.sum(2.0 ** (6.0 * vals))
print('  whole admissible   : sum tau^6 = 2^%.3f   (paper 2^-326.187)' % lg(s6))
print('  single-instance sum: sum tau^2 = 2^%.3f   (paper 2^-97.938)' % lg(np.sum(2.0 ** (2.0 * vals))))

# pricing factor
def Phi(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
lo, hi = 0.0, 10.0
for _ in range(200):
    mid = (lo + hi) / 2
    if Phi(mid) < 0.999992: lo = mid
    else: hi = mid
x = (lo + hi) / 2
fac = 4 * x * x
print('  Phi^-1(0.999992) = %.6f  ->  4*x^2 = 2^%.3f   (paper 2^6.218)' % (x, lg(fac)))
print('  price(single)  = 2^%.3f  (paper 2^334.841)' % (lg(fac) - 6 * vals.max()))
print('  price(whole)   = 2^%.3f  (paper 2^332.405)' % (lg(fac) - lg(s6)))

# ---------- the 24 strongest ----------
idx = np.argsort(-vals)[:24]
sel = np.unravel_index(np.argsort(-np.where(zero.ravel(), -1e9, acc.ravel()))[:24], acc.shape)
sel = np.array(sel)
bi, cj = sel
print('\n=== the 24 strongest ===')
print('  their log2|tau|:', np.round(acc[bi, cj], 3)[:10], '...')
print('  sum tau^6 = 2^%.3f   (paper 2^-327.214)' % lg(np.sum(2.0 ** (6.0 * acc[bi, cj]))))
print('  min/max pairwise product: 2^%.3f .. 2^%.3f   (paper 2^-111.527 .. 2^-109.926)'
      % (min(acc[bi[a], cj[a]] + acc[bi[b], cj[b]] for a in range(24) for b in range(24) if a != b),
         max(acc[bi[a], cj[a]] + acc[bi[b], cj[b]] for a in range(24) for b in range(24) if a != b)))

def triple(bi_, cj_):
    Bb = [int(x) for x in B[bi_]]; Cb = [int(x) for x in C[cj_]]
    Ab = [0] * 16
    for i in MU:
        Ab[i] = int(T12a[Cb[i] ^ Bb[i]][Cb[i]])
    P_ = [int(x) for x in P[bi_]]; Q_ = [int(x) for x in Q[cj_]]
    return Ab, Bb, Cb, P_, Q_

TX = [triple(bi[k], cj[k]) for k in range(24)]
tau = np.array([2.0 ** acc[bi[k], cj[k]] for k in range(24)])
Cm = np.eye(24)
for a in range(24):
    for b in range(24):
        if a == b: continue
        Aa, Ba, Ca, Pa, Qa = TX[a]; Ab_, Bb, Cb, Pb, Qb = TX[b]
        prod = 1.0
        for l in range(16):
            v1 = LAT[(Aa[l] ^ Ab_[l])][((Ba[l] ^ Ca[l]) ^ (Bb[l] ^ Cb[l]))]
            v2 = LAT[(Aa[l] ^ Ab_[l])][(Ca[l] ^ Cb[l])]
            v3 = LAT[(Pa[l] ^ Pb[l])][(Qa[l] ^ Qb[l])]
            prod *= (v1 / 256.0) * (v2 / 256.0) * (v3 / 256.0)
        Cm[a, b] = prod
off = Cm - np.eye(24)
mx = np.max(np.abs(off)); pos = np.unravel_index(np.argmax(np.abs(off)), off.shape)
print('  max |C_ij| = %.4e = 2^%.3f  at (%d,%d)   (paper 6.766e-21 = 2^-67.002 at (3,6))'
      % (mx, lg(mx), pos[0], pos[1]))
ev = np.linalg.eigvalsh(Cm)
print('  spectral radius of C-I = 2^%.3f   (paper 2^-66.991)' % lg(max(abs(ev - 1.0))))
SEI = tau @ np.linalg.solve(Cm, tau)
s2 = np.sum(tau ** 2)
print('  SEI = %.6e , sum tau^2 = %.6e , relative diff = 2^%.3f   (paper 2^-52.000)'
      % (SEI, s2, lg(abs(SEI - s2) / s2)))
print('\ntotal %.1fs' % (time.time() - t0))
