"""Recompute the correlation of the optimal triple of Eq. (19) from its bytes.

Inputs: the mask bytes printed in the paper, hard-coded below; verify.py tables.
Outputs: one line with the three masks and log2 tau, expected -54.771.
Paper object: Eq. (19) and Table 3; a one-off check next to verify.py.
"""

import sys, os, io, math, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from verify import LAT, ILAT, L, Linv, sigma, pullback

# the three strongest masks of the profile, hard-coded from the search
M = [
 ('0', '0x106aea00', '0', '0x106aea00', '0', '0x3f001300', '0', '0x3f001300', '0', '0x56061700', '0', '0x56061700'),
]

def parse12(*parts):
    out = []
    for w in parts:
        w = w.replace('0x', '').strip()
        if len(w) == 1:
            w = '0' * 8
        out += [int(w[2 * i:2 * i + 2], 16) for i in range(4)]
    return np.array(out, dtype=np.uint8)

def log2_tau(Ab, Bb, Cb):
    Pf = pullback(int.from_bytes(bytes(Bb), 'little'), Linv)
    Qf = pullback(int.from_bytes(bytes(Cb), 'little'), lambda s: Linv(sigma(L(s))))
    lg = 0.0
    for i in range(16):
        p = (Pf >> (8 * i)) & 0xFF
        q = (Qf >> (8 * i)) & 0xFF
        v1 = abs(int(LAT[p, q])); v2 = abs(int(ILAT[int(Cb[i]) ^ int(Bb[i]), int(Ab[i])])); v3 = abs(int(ILAT[int(Cb[i]), int(Ab[i])]))
        if v1 == 0 or v2 == 0 or v3 == 0:
            return float('-inf')
        lg += math.log2(v1 / 256) + math.log2(v2 / 256) + math.log2(v3 / 256)
    return lg

A0 = parse12('0', '0x106aea00', '0', '0x106aea00')
B0 = parse12('0', '0x3f001300', '0', '0x3f001300')
C0 = parse12('0', '0x56061700', '0', '0x56061700')
print('mask0 A=%s B=%s C=%s  log2 tau=%.6f' % (bytes(A0).hex(), bytes(B0).hex(), bytes(C0).hex(), log2_tau(A0, B0, C0)))
