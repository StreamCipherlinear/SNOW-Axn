"""Recompute the state-recovery cost table from the cached SEI values.

Inputs: sei_nested.json and sei_nested_24.json in this directory.
Outputs: log2 M, N, T and memory for the designers' trail, v = 1 and v = 24.
Paper object: Table 4, as a second implementation next to audit_attack_costs.py.
"""

import math, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
L = 512
Z1 = 4.2653          # Phi^-1(0.999992), exact to four decimals
DES = {'tau': -55.741, 'Bg': 243, 'N': 250.975, 'T': 252.201, 'M': 231.949}


def zt(Bg):
    t = math.sqrt(2.0 * (Bg + 1) * math.log(2.0))
    return t - (2.30753 + 0.27061 * t) / (1.0 + 0.99229 * t + 0.04481 * t * t)


def attack(log2sei, v, Bg):
    log2M = math.log2((Z1 + zt(Bg)) ** 2) - log2sei
    log2N = 0.5 * (log2M + L + 1 - Bg)
    T1 = 2 * 2 ** (0.5 * (log2M + L - Bg))
    T2 = Bg * 2 ** Bg + 2 ** log2M
    return log2M, log2N, math.log2(T1 + T2), math.log2(max(2 ** log2N, v * 2 ** Bg))


sei1 = json.load(open(os.path.join(HERE, 'sei_nested.json')))[0]
sei24 = json.load(open(os.path.join(HERE, 'sei_nested_24.json')))[-1]
rows = {}
for name, s, v, Bg in (('des', 4 * DES['tau'], 1, 243), ('one', sei1, 1, 241), ('full', sei24, 24, 241)):
    rows[name] = attack(s, v, Bg)
    print('%-5s log2M=%9.4f log2N=%9.4f log2T=%9.4f log2mem=%9.4f' % ((name,) + rows[name]))
print()
print('gain one vs des : T %.3f  N %.3f  M %.3f' % (rows['des'][2] - rows['one'][2], rows['des'][1] - rows['one'][1], rows['des'][0] - rows['one'][0]))
print('gain full vs des: T %.3f  N %.3f  M %.3f' % (rows['des'][2] - rows['full'][2], rows['des'][1] - rows['full'][1], rows['des'][0] - rows['full'][0]))
print('gain full vs one: T %.3f  N %.3f' % (rows['one'][2] - rows['full'][2], rows['one'][1] - rows['full'][1]))
