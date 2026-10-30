# Sum tau^k over the exact profile, and the attack table under the check-level rule
#   rho_eff^2 = sum tau^4   (instead of the paper's rho = sum tau^2).
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
tau = N[N > 0].astype(np.float64) * 2.0 ** -108
tau.sort(); tau = tau[::-1]
print('nonzero masks:', tau.size)

def lg(x): return math.log2(x)
print('\nfunctional            log2        (paper)')
print('  sum tau^2      : %9.3f   -97.938' % lg(np.sum(tau ** 2)))
print('  sum tau^4      : %9.3f   (not reported)' % lg(np.sum(tau ** 4)))
print('  sum tau^6      : %9.3f   -326.187' % lg(np.sum(tau ** 6)))
print('  (sum tau^2)^2  : %9.3f    <- what the paper uses' % (2 * lg(np.sum(tau ** 2))))
print('  sum tau^4, top 24 : %9.3f' % lg(np.sum(tau[:24] ** 4)))
print('  sum tau^4, top 512: %9.3f' % lg(np.sum(tau[:512] ** 4)))

# ---------------- attack table ----------------
def Phi(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def PhiInv(p):
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if Phi(mid) < p: lo = mid
        else: hi = mid
    return (lo + hi) / 2
ps = 0.999992
z1 = PhiInv(ps)
l = 512

def attack(tau_ms, rule):
    """rule 'sum2': rho = sum tau^2 (paper);  rule 'sum4': rho_eff^2 = sum tau^4 (correct)"""
    s2 = float(np.sum(tau_ms ** 2)); s4 = float(np.sum(tau_ms ** 4))
    best = None
    for Bg in range(1, l + 1):
        if rule == 'sum2':
            rho2 = s2 ** 2                       # (rho)^2 with rho = sum tau^2
        else:
            rho2 = s4                            # rho_eff^2 = sum tau^4
        M = (z1 + PhiInv(1 - 2.0 ** -(Bg + 1))) ** 2 / rho2
        if M <= 0: continue
        Nk = math.sqrt(M * 2.0 ** (l + 1 - Bg))
        T1 = math.sqrt(2.0) * Nk if M <= 2.0 ** (l - Bg) else Nk * (1 + lg(Nk))
        T2 = 2.0 ** Bg * (1 + math.ceil(lg(M)))
        T = T1 + T2
        if best is None or T < best[0]:
            best = (T, Bg, Nk, M, T1, T2)
    return best

print('\n m          | paper: rho=sum tau^2            | check-level: rho^2=sum tau^4')
print('            |  Bg   log2N  log2T      |  Bg   log2N  log2T')
for m in [1, 4, 8, 16, 24, 64, 512, 4096, tau.size]:
    sub = tau[:m]
    a = attack(sub, 'sum2'); b = attack(sub, 'sum4')
    print(' %-10d | %3d  %7.3f %7.3f      | %3d  %7.3f %7.3f' %
          (m, a[1], lg(a[2]), lg(a[0]), b[1], lg(b[2]), lg(b[0])))
