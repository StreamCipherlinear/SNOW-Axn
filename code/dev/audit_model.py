# End-to-end: is the paper's optimal triple an assignment of the STP model?
# Reconstruct every model variable from (alpha, beta) and check each constraint.
import sys, io, contextlib
import numpy as np
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
with contextlib.redirect_stdout(io.StringIO()):
    from verify import LAT, L, Linv, sigma, pullback, to_bytes, to_word, parse, opt, des

M = (1 << 128) - 1
def xor(a, b): return a ^ b

def model_state(tri):
    A = to_word(tri['A']); B = to_word(tri['B']); C = to_word(tri['C'])
    alpha = A
    beta = pullback(B, Linv)                 # B . L^-1
    Q = pullback(C, lambda s: Linv(sigma(L(s))))   # C . L^-1 sigma^-1 L
    m3 = Q
    gamma = pullback(m3, Linv)               # (L^T)^-1 m3
    sg = to_word(sigma(to_bytes(gamma)))     # sg = sigma(gamma)
    sgb = xor(sg, beta)
    m1 = pullback(sg, L)                     # AES_P(m1, sg):  m1 = L^T sg
    m2 = pullback(sgb, L)
    return dict(alpha=alpha, beta=beta, m1=m1, m2=m2, m3=m3, gamma=gamma, sg=sg, sgb=sgb,
                A=A, B=B, C=C, Q=Q)

def boxes(m, inmask, outmask):
    """(class cost, count) of the S-box layer  inmask -> outmask"""
    cost = 0; n = 0; vals = []
    for i in range(16):
        mi = (inmask >> (8 * i)) & 0xff
        ui = (outmask >> (8 * i)) & 0xff
        if mi == 0 and ui == 0:
            continue
        v = abs(LAT[mi][ui])
        if v == 0:
            return None, None, None
        cost += int(10000 * np.log2(256.0 / v))
        n += 1; vals.append(v)
    return cost, n, vals

for name, tri in [('optimal', opt), ("designers'", des)]:
    s = model_state(tri)
    print('=== %s triple ===' % name)
    print('  m1 == C            :', s['m1'] == s['C'])
    print('  m2 == C^B          :', s['m2'] == s['C'] ^ s['B'])
    print('  m3 == C.L^-1 s^-1 L:', s['m3'] == s['Q'])
    print('  alpha == A         :', s['alpha'] == s['A'])
    print('  beta  == B.L^-1    :', s['beta'] == pullback(s['B'], Linv))
    tot = 0; nb = 0; allv = []
    for nm, a, b in [('S1 alpha->m1', s['alpha'], s['m1']),
                     ('S2 alpha->m2', s['alpha'], s['m2']),
                     ('S3 beta ->m3', s['beta'], s['m3'])]:
        c, n, v = boxes(None, a, b)
        if c is None:
            print('   %s: INVALID (a zero-LAT pair)' % nm); continue
        tot += c; nb += n; allv += v
        print('   %-14s boxes=%2d  cost=%6d  (sum -log2|LAT|/256 = %.4f)' % (nm, n, c, c / 1e4))
    print('  Total = %d units = %.4f bits ; boxes = %d' % (tot, tot / 1e4, nb))
    print('  accepted by  Total < 540001 :', tot < 540001)
    print('  |tau| = 2^-%.4f' % (tot / 1e4))
    print()
