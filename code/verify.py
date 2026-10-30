# Independent verification of the central numbers of
# "Improved Correlation and Distinguishing Attacks on the SNOW-Axn Stream Cipher"
# (and of Appendix A of the SNOW-Axn design document).
# Built from scratch: GF(2^8), AES S-box, L = MC o SR, sigma, LAT/ILAT, then the
# per-byte factorisation of the three-block correlation.
import numpy as np
from math import log2

# ---------- GF(2^8) ----------
def xtime(a): return ((a << 1) ^ 0x1b) & 0xff if a & 0x80 else (a << 1)
def gmul(a, b):
    r = 0
    while b:
        if b & 1: r ^= a
        a = xtime(a); b >>= 1
    return r
def gpow(a, n):
    r = 1
    for _ in range(n): r = gmul(r, a)
    return r
def ginv(a): return 0 if a == 0 else gpow(a, 254)

AFF = [(1 << i) | (1 << ((i + 4) % 8)) | (1 << ((i + 5) % 8)) | (1 << ((i + 6) % 8)) | (1 << ((i + 7) % 8)) for i in range(8)]
def sbox(x):
    inv = ginv(x); r = 0x63
    for i, m in enumerate(AFF):
        if bin(inv & m).count('1') & 1: r ^= (1 << i)
    return r
SBOX = [sbox(x) for x in range(256)]
assert sorted(SBOX) == list(range(256))
INV_SBOX = [0] * 256
for x, y in enumerate(SBOX): INV_SBOX[y] = x
assert SBOX[0] == 0x63 and SBOX[1] == 0x7c and SBOX[0x53] == 0xed and SBOX[0xff] == 0x16

# ---------- byte-level linear layers ----------
SR = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]
SG = [0, 7, 14, 11, 4, 13, 10, 1, 8, 15, 6, 3, 12, 5, 2, 9]
MC = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]
INV_MC = [[14, 11, 13, 9], [9, 14, 11, 13], [13, 9, 14, 11], [11, 13, 9, 14]]
def shiftrows(st): return [st[SR[i]] for i in range(16)]
def inv_shiftrows(st):
    out = [0] * 16
    for i in range(16): out[SR[i]] = st[i]
    return out
def mixcolumns(st, M=MC):
    out = [0] * 16
    for c in range(4):
        for r in range(4):
            v = 0
            for s in range(4): v ^= gmul(M[r][s], st[4 * c + s])
            out[4 * c + r] = v
    return out
def L(st): return mixcolumns(shiftrows(st))
def Linv(st): return inv_shiftrows(mixcolumns(st, INV_MC))
def sigma(st): return [st[SG[i]] for i in range(16)]

# sanity: L and Linv are inverse, sigma is an involution
import random
random.seed(1)
for _ in range(50):
    st = [random.randrange(256) for _ in range(16)]
    assert Linv(L(st)) == st and sigma(sigma(st)) == st

# ---------- 128-bit words ----------
def to_bytes(w): return [(w >> (8 * i)) & 0xff for i in range(16)]
def to_word(b):
    w = 0
    for i in range(16): w |= b[i] << (8 * i)
    return w
def pullback(mask, f):
    """m' = m o f  (mask applied after the linear map f)"""
    out = 0
    for j in range(128):
        if bin(to_word(f(to_bytes(1 << j))) & mask).count('1') & 1:
            out |= 1 << j
    return out
def act(w): return tuple(i for i in range(16) if (w >> (8 * i)) & 0xff)

# ---------- LAT / ILAT ----------
PAR = np.zeros((256, 256), dtype=np.int8)
for a in range(256):
    PAR[a] = [(bin(a & x).count('1') & 1) for x in range(256)]
SA = np.array([PAR[m] for m in range(256)], dtype=np.int8)          # parity(m.x)
SB = np.array([PAR[u][SBOX] for u in range(256)], dtype=np.int8)    # parity(u.S(x))
SI = np.array([PAR[u][INV_SBOX] for u in range(256)], dtype=np.int8)
mA = 1 - 2 * SA                                                     # (-1)^{m.x}
mB = 1 - 2 * SB
mI = 1 - 2 * SI
LAT = (mA.astype(np.int16) @ mB.T.astype(np.int16)).astype(int)      # LAT[m][u] = sum_x (-1)^{m.x + u.S(x)}
ILAT = (mA.astype(np.int16) @ mI.T.astype(np.int16)).astype(int)
dist = {}
for v in np.abs(LAT).ravel(): dist[int(v)] = dist.get(int(v), 0) + 1
print('LAT magnitude distribution (m,u over 256x256):')
for k in sorted(dist): print('   %3d : %6d' % (k, dist[k]))
print("paper's Table 1 : 0:4590  4:12240  8:9180  12:10200  16:8670  20:6120  24:9180  28:4080  32:1275 (+256:1)")

# ---------- the two mask triples ----------
def parse(h):
    out = []
    for g in h.split():
        v = int(g, 16)
        out += [(v >> s) & 0xff for s in (24, 16, 8, 0)]   # byte 0 leftmost
    return out
opt = dict(A=parse('00000000 106aea00 00000000 106aea00'),
           B=parse('00000000 3f001300 00000000 3f001300'),
           C=parse('00000000 56061700 00000000 56061700'))
des = dict(A=parse('280036b9 00000000 280036e0 00000000'),
           B=parse('4d00d900 00000000 4d00d900 00000000'),
           C=parse('b800b820 00000000 b800b820 00000000'))

def analyse(name, tri):
    A, B, C = to_word(tri['A']), to_word(tri['B']), to_word(tri['C'])
    P = pullback(B, Linv)                              # B . L^-1
    Q = pullback(C, lambda s: Linv(sigma(L(s))))       # C . L^-1 sigma^-1 L
    CB = to_word([x ^ y for x, y in zip(tri['C'], tri['B'])])
    muA, muC, muCB, muP, muQ = act(A), act(C), act(CB), act(P), act(Q)
    print('\n=== %s ===' % name)
    print('  mu(A)=%s' % (muA,))
    print('  mu(C)=%s  mu(C^B)=%s' % (muC, muCB))
    print('  mu(B.L^-1)=%s' % (muP,))
    print('  mu(C.L^-1 s^-1 L)=%s' % (muQ,))
    print('  designers criterion holds: %s' % (muA == muC == muCB and muP == muQ))
    boxes = []
    Ab, Bb, Cb, Pb, Qb = tri['A'], tri['B'], tri['C'], to_bytes(P), to_bytes(Q)
    for i in range(16):
        if i in muP: boxes.append(('R1', i, Pb[i], Qb[i], abs(LAT[Pb[i]][Qb[i]])))
        if i in muA:
            boxes.append(('R2', i, Cb[i] ^ Bb[i], Ab[i], abs(ILAT[Cb[i] ^ Bb[i]][Ab[i]])))
            boxes.append(('R3', i, Cb[i], Ab[i], abs(ILAT[Cb[i]][Ab[i]])))
    prod = 1
    for blk, i, mi, ui, v in boxes:
        prod *= int(v)
        print('    %-2s byte %2d  in=%02x out=%02x  |tab|=%2d  corr=2^%.3f' % (blk, i, mi, ui, v, log2(v / 256)))
    print('  active boxes = %d ;  HW(mu_P)+2HW(mu_A) = %d' % (len(boxes), len(muP) + 2 * len(muA)))
    print('  total correlation = 2^%.4f' % (log2(prod) - 8 * len(boxes)))
    print('  box magnitudes   : %s' % {v: sum(1 for b in boxes if b[4] == v) for v in sorted({b[4] for b in boxes})})
    return prod

p1 = analyse('optimal triple (eq. optmasks)', opt)
p2 = analyse("designers' triple (Appendix A of the design document)", des)
print('\nlog2 tau*  = %.4f  (paper: -54.771)' % (log2(p1) - 18 * 8))
print('log2 tau_d = %.4f  (paper: -55.741)' % (log2(p2) - 18 * 8))
print('gap = %.4f bits ; ratio = %.3f' % (log2(p1) - log2(p2), p1 / p2))

