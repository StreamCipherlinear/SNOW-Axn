"""Price the vectorial attack of Section 4.2.

Convention of the paper. The statistic is a vector over the space V_v, so one
sample carries the whole vector and no pair of clocks has to collide. The rule
of Zhang, Liu, Meier, Sun, Feng and Wu (ASIACRYPT 2025) then assigns

    N = 4 kappa ln 2 / SEI(V_v),   T = N + kappa 2^kappa,   memory = 2^kappa,

where kappa is the number of state bits decoded in one step and not the
dimension v of the space, which enters only through the capacity.

Inputs: sei_nested.json (v = 1..20) and sei_nested_24.json (v = 21..24).
Outputs: the dimension scan of Table 6 and the figures of the Result paragraph.
Paper object: Table 6, Eq. (22), Eq. (23), and the numbers of Section 4.2.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
KAPPA = 240                      # state bits decoded in one step
SINGLE = 250.919                 # log2 time of the single approximation, Table 4


def log2_sum(a, b):
    m = max(a, b)
    return m + math.log2(2.0 ** (a - m) + 2.0 ** (b - m))


def log2_data(log2sei):
    """4 kappa ln 2 / SEI, in base-two logarithm."""
    return math.log2(4.0 * KAPPA * math.log(2.0)) - log2sei


def main():
    sei = (json.load(open(os.path.join(HERE, 'sei_nested.json'))) +
           json.load(open(os.path.join(HERE, 'sei_nested_24.json'))))
    assert len(sei) == 24
    print('%2s %12s %10s' % ('v', 'log2 SEI', 'log2 N'))
    best_v, best_n = 0, None
    for v, s in enumerate(sei, start=1):
        n = log2_data(s)
        print('%2d %12.3f %10.3f' % (v, s, n))
        if best_n is None or n < best_n:
            best_v, best_n = v, n
    log2_mem = float(KAPPA)
    log2_t = log2_sum(best_n, KAPPA + math.log2(KAPPA))
    print()
    print('kappa = %d  ->  log2 mem = %.3f' % (KAPPA, log2_mem))
    print('best space v = %d' % best_v)
    print('v = 24: log2 N = %.3f  log2 T = %.3f  log2 mem = %.3f' % (best_n, log2_t, log2_mem))
    print('gain in time over the single approximation: %.3f bits' % (SINGLE - log2_t))
    print()
    print('paper: 226.222 247.907 240 ; gain 3.012')
    assert best_v == 24
    assert abs(best_n - 226.2216) < 5e-4
    assert abs(log2_t - 247.9069) < 5e-4
    print('checks passed')


if __name__ == '__main__':
    main()
