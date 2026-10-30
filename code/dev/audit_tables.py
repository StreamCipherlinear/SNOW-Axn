# Independent check: parse the two tables of appendix.tex and compare with a fresh computation.
import re, sys, io, contextlib, math
import numpy as np
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study\fourth2.py', encoding='utf-8-sig').read().split("def lg(x)")[0])
from math import log, sqrt, pi, log1p, erf
def phi_inv_tail(eps):
    if eps > 1e-13:
        lo, hi = -40.0, 40.0
        for _ in range(300):
            mid = (lo + hi) / 2
            if 0.5 * (1.0 + erf(mid / math.sqrt(2.0))) < 1.0 - eps: lo = mid
            else: hi = mid
        return (lo + hi) / 2
    u = -log(eps); x = sqrt(2 * u - log(2 * pi * u))
    for _ in range(60):
        lq = -x * x / 2 - log(x) - 0.5 * log(2 * pi) + log1p(-1 / x**2 + 3 / x**4 - 15 / x**6)
        x -= (lq - log(eps)) / (-x + 1.0 / x)
    return x
def lg(x): return math.log2(x)
tau = tau[::-1] if tau[0] < tau[-1] else tau
z1 = phi_inv_tail(1 - 0.999992); l = 512
def row(m):
    s4 = float(np.sum(tau[:m] ** 4)); best = None
    for Bg in range(1, l + 1):
        M = (z1 + phi_inv_tail(2.0 ** -(Bg + 1))) ** 2 / s4
        N = math.sqrt(M * 2.0 ** (l + 1 - Bg))
        T1 = math.sqrt(2.0) * N if M <= 2.0 ** (l - Bg) else N * (1 + lg(N))
        T2 = 2.0 ** Bg * (1 + math.ceil(lg(M)))
        if best is None or T1 + T2 < best[0]: best = (T1 + T2, Bg, N, M, T1, T2)
    return s4, best

ms = [1, 4, 8, 16, 24, 64, 512, 4096, tau.size]
tex = open(r'C:\Users\Lenovo\Desktop\Processing\snowa\SNOW-A\appendix.tex', encoding='utf-8').read()
def num(s): return float(s.replace('$', '').replace('{', '').replace('}', '').replace(',', ''))
ok = True
for m in ms:
    s4, b = row(m)
    # expected strings
    exp = ['%.3f' % lg(s4), '%.3f' % (lg(s4) / 4), '%d' % b[1], '%.3f' % lg(b[2]), '%.3f' % lg(b[0])]
    found = all(('-$' + e + '$' in tex) or ('$' + e + '$' in tex) or (e in tex) for e in exp)
    if not found:
        ok = False
        print('MISSING m=%-10d  expected: sum4=-%s  tau4=-%s  B=%s  N=%s  T=%s' % (m, exp[0], exp[1], exp[2], exp[3], exp[4]))
print('all table values present in appendix.tex:', ok)

# cross-check the derived quantities quoted in the text
s4all = float(np.sum(tau ** 4)); ball = row(tau.size)[1]
print('\nquoted values:')
print('  sum tau^4      = 2^%.3f   (text -213.382)' % lg(s4all))
print('  rho_eff        = 2^%.3f   (text -106.691)' % (lg(s4all) / 2))
print('  tau_eff        = 2^%.3f   (text -53.345)' % (lg(s4all) / 4))
print('  family  B=%d  log2T=%.3f  log2N=%.3f (text 240 / 249.003 / 247.675)' % (ball[1], lg(ball[0]), lg(ball[2])))
print('  M(kappa=24)    = 2^%.3f   (text 219.438)' % lg(4 * 24 * math.log(2) / s4all))
print('  M(kappa=1)     = 2^%.3f   (text 220.553)' % lg(4 * 1 * math.log(2) / float(np.sum(tau[:1] ** 4))))
print('  gain           = %.3f bit (text 1.115)' % (lg(4 * math.log(2) / float(np.sum(tau[:1] ** 4))) - lg(4 * 24 * math.log(2) / s4all)))
print('  vs designers   = %.3f / %.3f (text 3.198 / 3.299)' % (252.201 - lg(ball[0]), 250.974 - lg(ball[2])))
print('  vs single      = %.3f / %.3f (text 1.917 / 2.354)' % (250.920 - lg(ball[0]), 250.029 - lg(ball[2])))
print('  m=24 row       = 2^%.3f / 2^%.3f / B=%d (text 250.158 / 248.928 / 241)' % (lg(row(24)[1][0]), lg(row(24)[1][2]), row(24)[1][1]))
print('  m=24 gains     = %.3f %.3f vs designers, %.3f %.3f vs single (text 2.043 2.046 0.762 1.101)'
      % (252.201 - lg(row(24)[1][0]), 250.974 - lg(row(24)[1][2]),
         250.920 - lg(row(24)[1][0]), 250.029 - lg(row(24)[1][2])))

