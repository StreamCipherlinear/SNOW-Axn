"""Price the vectorial attack from the cached capacities of the nested spaces.

Inputs: sei_nested.json and sei_nested_24.json in this directory.
Outputs: per-k Bg, log2 N, T and memory for Bg in 235..250, and the best k.
Paper object: the dimension scan of Table 6 and the space SEI of Section 4.2.
"""

import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__))
a = json.load(open(os.path.join(HERE, 'sei_nested.json')))
b = json.load(open(os.path.join(HERE, 'sei_nested_24.json')))
SEI = a + b
L, PS = 512, 0.999992


def zt(Bg):
    t = math.sqrt(2.0 * (Bg + 1) * math.log(2.0))
    return t - (2.30753 + 0.27061 * t) / (1.0 + 0.99229 * t + 0.04481 * t * t)


def z_of(p):
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    return t - (2.30753 + 0.27061 * t) / (1.0 + 0.99229 * t + 0.04481 * t * t)


z1 = z_of(PS)
print('%2s %12s %6s %10s %10s %10s' % ('k', 'log2 SEI', 'Bg*', 'log2 N', 'log2 T', 'log2 mem'))
best = None
for k, s in enumerate(SEI, start=1):
    row = None
    for Bg in range(235, 251):
        log2M = math.log2((z1 + zt(Bg)) ** 2) - s
        log2N = 0.5 * (log2M + L + 1 - Bg)
        T1 = 2 * 2 ** (0.5 * (log2M + L - Bg))
        T2 = Bg * 2 ** Bg + 2 ** log2M
        T = T1 + T2
        mem = max(2 ** log2N, 2 ** Bg)
        if row is None or T < row[0]:
            row = (T, Bg, log2N, mem)
    print('%2d %12.4f %6d %10.4f %10.4f %10.4f' % (k, s, row[1], row[2], math.log2(row[0]), math.log2(row[3])))
    if best is None or row[0] < best[0]:
        best = (row[0], k, row[1], row[2], row[3])
print()
print('best k=%d Bg=%d log2 N=%.4f log2 T=%.4f log2 mem=%.4f' % (best[1], best[2], best[3], math.log2(best[0]), math.log2(best[4])))
b1 = SEI[0]
for k, s in ((1, SEI[0]), (best[1], SEI[best[1] - 1])):
    pass
