# Vectorial FCA with the SNOW substitution tau -> rho = tau^2 (the decoder sees the
# folded parity check, not the sample):  SEI = sum rho^2 = sum tau^4,
# N = 4*l*ln2/SEI,  cost = sqrt(N*2^(l-B)) + 2N + B*2^B.
import sys, io, contextlib, math
import numpy as np
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
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
Pb_ = [pullback(b, Linv) for b in tA]
Qb_ = [pullback(c, lambda s: Linv(sigma(L(s)))) for c in tC]
def arr(v): return np.array([[x >> (8 * i) & 0xff for i in range(16)] for x in v], dtype=np.uint8)
def expand(basis, k, sel): return np.bitwise_xor.reduce(sel[:, :, None] * basis[None, :, :], axis=1)
Bm = expand(arr(tA), 16, ((np.arange(1 << 16)[:, None] >> np.arange(16)) & 1).astype(np.uint8))
Pm = expand(arr(Pb_), 16, ((np.arange(1 << 16)[:, None] >> np.arange(16)) & 1).astype(np.uint8))
Cm = expand(arr(tC), 8, ((np.arange(1 << 8)[:, None] >> np.arange(8)) & 1).astype(np.uint8))
Qm = expand(arr(Qb_), 8, ((np.arange(1 << 8)[:, None] >> np.arange(8)) & 1).astype(np.uint8))
absL = np.abs(LAT.astype(np.int16)); absI = np.abs(ILAT.astype(np.int16))
ILa = np.abs(ILAT.astype(np.int32))
T12a = np.zeros((256, 256), dtype=np.uint8)
for m1 in range(256):
    pr = ILa[m1][None, :] * ILa; pr[:, 0] = 0
    T12a[m1] = pr.argmax(axis=1).astype(np.uint8)
N = np.ones((len(Bm), len(Cm)), dtype=np.int64)
for i in MU:
    b = Bm[:, i][:, None].astype(np.int32); c = Cm[:, i][None, :].astype(np.int32)
    p = Pm[:, i][:, None].astype(np.int32); q = Qm[:, i][None, :].astype(np.int32)
    cb = (b ^ c).astype(np.int32); a = T12a[cb, c].astype(np.int32)
    N *= (absL[p, q] // 4).astype(np.int64) * (absI[cb, a] // 4).astype(np.int64) * (absI[c, a] // 4).astype(np.int64)
N[0, 0] = 0
tau = N[N > 0].astype(np.float64) * 2.0 ** -108
tau.sort(); tau = tau[::-1]
def lg(x): return math.log2(x)
l = 512

def cost(m):
    sub = tau[:m]
    s4 = float(np.sum(sub ** 4))          # SEI after tau -> rho = tau^2
    best = None
    for B in range(8, 400):
        Nc = 4.0 * B * math.log(2.0) / s4          # checks needed (l = B)
        D = math.sqrt(Nc * 2.0 ** (l - B))         # collision: data
        walsh = B * 2.0 ** B
        tot = D + 2 * Nc + walsh
        if best is None or tot < best[0]:
            best = (tot, B, Nc, D, walsh)
    return s4, best

print(' m            log2 sum tau^4 |  B    log2N(checks)  log2D(data)  log2Walsh  log2TOT')
for m in [1, 4, 8, 16, 24, 64, 512, 4096, tau.size]:
    s4, b = cost(m)
    print(' %-12d %9.3f     | %3d   %9.3f     %9.3f   %9.3f  %9.3f' %
          (m, lg(s4), b[1], lg(b[2]), lg(b[3]), lg(b[4]), lg(b[0])))
print('\nfor comparison, the paper (Selcuk with rho = sum tau^2): all -> 2^243.091')
