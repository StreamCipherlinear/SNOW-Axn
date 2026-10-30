"""Two basis-selection strategies over the correlation profile, both priced by
Theorem 4 of [ZLM+25] with v = k and d1 = k.

(a) nested: the k strongest masks of the profile;
(b) greedy: at every step the mask that adds the largest SEI.

SEI is summed over the non-zero elements of the span only, and the sum is
accumulated in log space.  The script asserts that SEI is non-decreasing in k
and that the single-mask value equals (2^-54.770580)^4.
"""
import sys, os, io, math, json, contextlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    from verify import LAT, ILAT, L, Linv, sigma, pullback

MU = (4, 5, 6, 12, 13, 14)
LOG2_TAU_STAR = -54.770580


def nullspace(rows, n):
    rows = list(rows); piv = {}; r = 0
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
        piv[col] = r; r += 1
        if r == len(rows):
            break
    out = []
    for f in [c for c in range(n) if c not in piv]:
        v = 1 << f
        for c, ri in piv.items():
            if (rows[ri] >> f) & 1:
                v |= 1 << c
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
                if (w >> (8 * ob + ot)) & 1:
                    row |= 1 << j
            rows.append(row)
    out = []
    for v in nullspace(rows, len(free)):
        m = 0
        for j, (b, t) in enumerate(free):
            if (v >> j) & 1:
                m |= 1 << (8 * b + t)
        out.append(m)
    return out


to_bytes = lambda x: np.array([(int(x) >> (8 * i)) & 0xFF for i in range(16)], dtype=np.uint8)
tA = admissible_basis(MU, lambda w: pullback(w, Linv))
tC = admissible_basis(MU, lambda w: pullback(w, lambda s: Linv(sigma(L(s)))))
A_basis = np.array([to_bytes(x) for x in tA], dtype=np.uint8)
C_basis = np.array([to_bytes(x) for x in tC], dtype=np.uint8)
P_basis = np.array([to_bytes(pullback(int(x), Linv)) for x in tA], dtype=np.uint8)
Q_basis = np.array([to_bytes(pullback(int(x), lambda s: Linv(sigma(L(s))))) for x in tC], dtype=np.uint8)

def expand(basis, sel):
    return np.bitwise_xor.reduce(sel[:, :, None] * basis[None, :, :], axis=1)

selB = ((np.arange(1 << 16)[:, None] >> np.arange(16)) & 1).astype(np.uint8)
selC = ((np.arange(1 << 8)[:, None] >> np.arange(8)) & 1).astype(np.uint8)
B = expand(A_basis, selB); C = expand(C_basis, selC)
P = expand(P_basis, selB); Q = expand(Q_basis, selC)

absL = np.abs(LAT.astype(np.int64))
absI = np.abs(ILAT.astype(np.int64))
ILa = np.abs(ILAT.astype(np.int32))
T12a = np.zeros((256, 256), dtype=np.uint8)
for m1 in range(256):
    pr = ILa[m1][None, :] * ILa
    pr[:, 0] = 0
    T12a[m1] = pr.argmax(axis=1).astype(np.uint8)

Nrank = np.ones((len(B), len(C)), dtype=np.int64)
for i in MU:
    b = B[:, i][:, None]; c = C[:, i][None, :]
    p = P[:, i][:, None]; q = Q[:, i][None, :]
    cb = (b ^ c).astype(np.int64)
    a = T12a[cb, c].astype(np.int64)
    Nrank *= (absL[p, q] // 4) * (absI[cb, a] // 4) * (absI[c, a] // 4)
Nrank[0, 0] = 0
flat = Nrank.astype(np.float64).ravel()
order = np.argsort(np.where((Nrank != 0).ravel(), -flat, np.inf))[:64]
bi, cj = np.unravel_index(order, Nrank.shape)

def triple(k):
    Bb = B[bi[k]].copy(); Cb = C[cj[k]].copy()
    Ab = np.zeros(16, dtype=np.uint8)
    for i in MU:
        Ab[i] = T12a[Cb[i] ^ Bb[i]][Cb[i]]
    return Ab, Bb, Cb, P[bi[k]].copy(), Q[cj[k]].copy()

cands = [triple(k) for k in range(len(order))]
lg0 = 0.0
A0, B0, C0, p0, q0 = cands[0]
v1 = absL[p0, q0]; v2 = absI[C0 ^ B0, A0]; v3 = absI[C0, A0]
lg0 = float(np.log2(v1 / 256).sum() + np.log2(v2 / 256).sum() + np.log2(v3 / 256).sum())
print('top mask log2 tau = %.6f   (paper %.6f)' % (lg0, LOG2_TAU_STAR))
assert abs(lg0 - LOG2_TAU_STAR) < 1e-4


def lg_array(V):
    Ab, Bb, Cb, p, q = V
    cb = (Cb ^ Bb).astype(np.int64); a = Ab.astype(np.int64)
    w1 = absL[p, q]; w2 = absI[cb, a]; w3 = absI[Cb.astype(np.int64), a]
    bad = (w1 == 0).any(1) | (w2 == 0).any(1) | (w3 == 0).any(1)
    lg = (np.log2(np.maximum(w1, 1) / 256).sum(1) + np.log2(np.maximum(w2, 1) / 256).sum(1)
          + np.log2(np.maximum(w3, 1) / 256).sum(1))
    lg[bad] = -np.inf
    zero_elem = ~((Ab.astype(np.int64) | Bb.astype(np.int64) | Cb.astype(np.int64)).any(1)) & True
    lg[zero_elem] = -np.inf   # any all-zero mask has correlation 1 and is excluded
    return lg


def log2_sei(lg):
    fin = lg[np.isfinite(lg)]
    if fin.size == 0:
        return float('-inf')
    m = 4.0 * fin.max()
    return float(m + math.log2(np.exp2(4.0 * fin - m).sum()))

ZERO = (np.zeros((1, 16), np.uint8),) * 3 + (np.zeros((1, 16), np.uint8), np.zeros((1, 16), np.uint8))

def price(L, k):
    Nreq = 4 * k * math.log(2) / float(np.exp2(L))
    T = 2 * Nreq + k * 2 ** k
    return Nreq, T, k * 2 ** k

print()
print('(a) nested, k strongest masks')
print('%2s %13s %10s %10s' % ('k', 'log2 SEI', 'log2 N', 'log2 T'))
V = ZERO; prev = float('-inf'); bestn = None
sei_list = []
for k in range(1, 21):
    Ab, Bb, Cb, p, q = cands[k - 1]
    V = tuple(np.concatenate([V[i], V[i] ^ x]) for i, x in enumerate((Ab, Bb, Cb, p, q)))
    L = log2_sei(lg_array(V))
    assert L >= prev - 1e-9, 'SEI decreased at k=%d' % k
    prev = L
    Nreq, T, mem = price(L, k)
    print('%2d %13.4f %10.4f %10.4f' % (k, L, math.log2(Nreq), math.log2(T)))
    sei_list.append(L)
    if bestn is None or T < bestn[0]:
        bestn = (T, k, L)

with open('sei_nested.json','w') as f:
    json.dump(sei_list, f)
print()
print('(b) greedy over the first 64 masks')
print('%2s %13s %10s %10s  added' % ('k', 'log2 SEI', 'log2 N', 'log2 T'))
V = ZERO; used = []; prev = float('-inf'); bestg = None
for k in range(1, 13):
    pick = None; pickL = float('-inf'); pickV = None
    for idx, (Ab, Bb, Cb, p, q) in enumerate(cands):
        if idx in used:
            continue
        W = tuple(np.concatenate([V[i], V[i] ^ x]) for i, x in enumerate((Ab, Bb, Cb, p, q)))
        nz = ((W[0].astype(np.int64) | W[1].astype(np.int64) | W[2].astype(np.int64)).any(1))
        if int((~nz).sum()) > 1:
            continue
        L = log2_sei(lg_array(W))
        if L > pickL:
            pickL, pick, pickV = L, idx, W
    assert pickL >= prev - 1e-9, 'SEI decreased at k=%d' % k
    prev = pickL; V = pickV; used.append(pick)
    Nreq, T, mem = price(pickL, k)
    print('%2d %13.4f %10.4f %10.4f  %s' % (k, pickL, math.log2(Nreq), math.log2(T), bytes(cands[pick][0]).hex()))
    if bestg is None or T < bestg[0]:
        bestg = (T, k, pickL)

print()
print('nested best: k=%d log2 T=%.4f' % (bestn[1], math.log2(bestn[0])))
print('greedy best: k=%d log2 T=%.4f' % (bestg[1], math.log2(bestg[0])))
