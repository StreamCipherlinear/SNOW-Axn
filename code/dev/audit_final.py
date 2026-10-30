# End-to-end audit: recompute every cell of the sweep table and the headline numbers,
# then compare with what is written in appendix.tex / document.tex.
import re, sys, io, contextlib, math
import numpy as np
from math import log, sqrt, pi, log1p, erf
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
with contextlib.redirect_stdout(io.StringIO()):
    from verify import LAT, ILAT, L, Linv, sigma, pullback
MU = (4, 5, 6, 12, 13, 14)

def phi_inv_tail(eps):
    if eps > 1e-13:
        lo, hi = -40.0, 40.0
        for _ in range(300):
            mid = (lo + hi) / 2
            if 0.5 * (1.0 + erf(mid / math.sqrt(2.0))) < 1.0 - eps: lo = mid
            else: hi = mid
        return (lo + hi) / 2
    u = -log(eps); x = sqrt(2 * u - log(2 * pi * u))
    for _ in range(80):
        lq = -x * x / 2 - log(x) - 0.5 * log(2 * pi) + log1p(-1 / x**2 + 3 / x**4 - 15 / x**6)
        x -= (lq - log(eps)) / (-x + 1.0 / x)
    return x

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
l = 512; ps = 0.999992; z1 = phi_inv_tail(1 - ps)

def attack(s4):
    best = None
    for Bg in range(1, l + 1):
        M = (z1 + phi_inv_tail(2.0 ** -(Bg + 1))) ** 2 / s4
        Nk = math.sqrt(M * 2.0 ** (l + 1 - Bg))
        T1 = math.sqrt(2.0) * Nk if M <= 2.0 ** (l - Bg) else Nk * (1 + lg(Nk))
        T2 = 2.0 ** Bg * (1 + math.ceil(lg(M)))
        T = T1 + T2
        if best is None or T < best[0]: best = (T, Bg, Nk, M, T1, T2)
    return best

# --- the table as written ---
tex = open(r'C:\Users\Lenovo\Desktop\Processing\snowa\SNOW-A\appendix.tex', encoding='utf-8-sig').read()
body = tex.split(r'\label{tab:complexity-full}')[1]
body = body.split(r'\midrule')[1].split(r'\bottomrule')[0]
rows = []
for line in body.split('\\\\'):
    cells = line.split('&')
    if len(cells) >= 8:
        rows.append('&'.join(cells[1:]).strip() + '\\\\')
def nums(s):
    return [x.replace('$','').replace('{','').replace('}','').replace(',','').replace('\\mathbf','').strip()
            for x in re.findall(r'\$[^$]*\$', s)]
print('rows found in the table:', len(rows))
ms = [1, 1, 4, 8, 16, 24, 64, 512, 4096, tau.size]
labels = ['designers', 'optimum'] + ['m=%d' % m for m in ms[2:]]
ok = True
for lbl, m, row in zip(labels, ms, rows):
    got = nums(row)
    if m == 1 and lbl == 'designers':
        s4 = 2.0**-222.964
    else:
        s4 = float(np.sum(tau[:m] ** 4))
    b = attack(s4)
    exp = ['%.3f' % lg(s4), '%.3f' % (lg(s4) / 4), '%d' % b[1], '%.3f' % lg(b[2]), '%.3f' % lg(b[4]), '%.3f' % lg(b[5]), '%.3f' % lg(b[0])]
    diffs = []
    for e, g in zip(exp, got):
        try:
            diffs.append(abs(float(g) - float(e)))
        except ValueError:
            diffs.append(9)
    bad = max(diffs)
    flag = 'OK ' if bad <= 0.002 else 'BAD'
    if bad > 0.002: ok = False
    print('%s %-12s log2S4=%9s  tau4=%8s B=%3s N=%9s T1=%9s T2=%9s T=%9s   maxdiff=%.4f'
          % (flag, lbl, exp[0], exp[1], exp[2], exp[3], exp[4], exp[5], exp[6], bad))
print('\nALL TABLE CELLS MATCH:', ok)

# --- headline numbers quoted in the text ---
s4_24 = float(np.sum(tau[:24] ** 4)); s4_all = float(np.sum(tau ** 4))
b24, ball = attack(s4_24), attack(s4_all)
print('\nquoted vs computed:')
print('  m=24  T=2^250.158? %.3f   M=2^248.928? %.3f   B=241? %d' % (lg(b24[0]), lg(b24[2]), b24[1]))
print('  all   T=2^249.003? %.3f   M=2^247.675? %.3f   B=240? %d' % (lg(ball[0]), lg(ball[2]), ball[1]))
print('  gains vs designers 252.201/250.974: %.3f / %.3f  (text 2.043 / 2.046)' % (252.201 - lg(b24[0]), 250.974 - lg(b24[2])))
print('  gains vs single    250.920/250.029: %.3f / %.3f  (text 0.762 / 1.101)' % (250.920 - lg(b24[0]), 250.029 - lg(b24[2])))
print('  whole family vs designers 252.201/250.974: %.3f / %.3f  (text 3.198 / 3.299)'
      % (252.201 - lg(ball[0]), 250.974 - lg(ball[2])))
print('  whole family vs single    250.920/250.029: %.3f / %.3f  (text 1.917 / 2.354)'
      % (250.920 - lg(ball[0]), 250.029 - lg(ball[2])))
print('  family adds %.3f bits in data (text 2.85)' % ((lg(float(np.sum(tau[:1] ** 4))) - lg(s4_all)) / 2))
print('  Eq.(23) at B=240: kappa=24: M=2^%.3f (text 222.760);  kappa=1: M=2^%.3f (text 228.460);  gain %.3f (text 5.700)'
      % (lg(4 * 240 * math.log(2) / s4_all), lg(4 * 240 * math.log(2) / float(np.sum(tau[:1] ** 4))),
         lg(4 * 240 * math.log(2) / float(np.sum(tau[:1] ** 4))) - lg(4 * 240 * math.log(2) / s4_all)))

