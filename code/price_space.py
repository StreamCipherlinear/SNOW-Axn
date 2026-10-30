"""Price the vectorial attack for k = 1..16 with the collision constraint.

Inputs: the capacities computed by verify_space.py (hard-coded below, all
verified against (tau*)^4 at k = 1 and monotone in k).  For every k the number
of parity-check equations is

    M = (Phi^-1(ps) + Phi^-1(1 - 2^-(Bg+1)))^2 / SEI(V_k),

the data follow from the collision identity N^2 / 2^(l+1-Bg) = M, and

    T = 2 sqrt(M 2^(l-Bg)) + Bg 2^Bg + M ,   mem = max(N, 2^Bg).

Bg is scanned over 200..260 and the minimum of T is reported.
"""
import math, json
from statistics import NormalDist

L = 512
PS = 0.999992
SEI = json.load(open('sei_nested.json'))

def z_of(p):
    # Abramowitz and Stegun 26.2.23, for p > 1/2
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    return t - (2.30753 + 0.27061 * t) / (1.0 + 0.99229 * t + 0.04481 * t * t)

z1 = z_of(PS)


def zt(Bg):
    t = math.sqrt(2.0 * (Bg + 1) * math.log(2.0))
    return t - (2.30753 + 0.27061 * t) / (1.0 + 0.99229 * t + 0.04481 * t * t)

def price(log2sei, Bg):
    log2M = math.log2(z1 ** 2) if False else math.log2(
        (z1 + zt(Bg)) ** 2) - log2sei
    log2N = 0.5 * (log2M + L + 1 - Bg)
    T1 = 2 * 2 ** (0.5 * (log2M + L - Bg))
    T2 = Bg * 2 ** Bg + 2 ** log2M
    T = T1 + T2
    mem = max(2 ** log2N, 2 ** Bg)
    return log2M, log2N, math.log2(T), math.log2(mem)


print('%2s %12s %9s %9s %9s %9s' % ('k', 'log2 SEI', 'Bg*', 'log2 N', 'log2 T', 'log2 mem'))
best = None
for k, s in enumerate(SEI, start=1):
    row = None
    for Bg in range(200, 261):
        m, n, t, mem = price(s, Bg)
        if row is None or t < row[0]:
            row = (t, Bg, m, n, mem)
    print('%2d %12.4f %9d %9.4f %9.4f %9.4f' % (k, s, row[1], row[3], row[0], row[4]))
    if best is None or row[0] < best[0]:
        best = (row[0], k, row[1], row[3], row[4])
print()
print('best: k=%d  Bg=%d  log2 N=%.4f  log2 T=%.4f  log2 mem=%.4f'
      % (best[1], best[2], best[3], best[0], best[4]))
