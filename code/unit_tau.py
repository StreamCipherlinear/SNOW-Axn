"""Recompute the correlation of the optimal triple byte by byte.

Inputs: the mask bytes of Eq. (19), hard-coded below; verify.py tables.
Outputs: one line with log2 tau, expected -54.771.
Paper object: Eq. (19); a one-off unit check next to verify.py.
"""

import sys, os, io, contextlib, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from verify import LAT, ILAT, L, Linv, sigma, pullback

def parse(s):
    words = [w for w in s.replace('0x', '').split(',')]
    out = []
    for w in words:
        w = w.strip()
        if len(w) == 1:
            w = '0' * 8
        out += [int(w[2 * i:2 * i + 2], 16) for i in range(4)]
    return np.array(out, dtype=np.uint8)

A = parse('0, 0x106aea00, 0, 0x106aea00')
B = parse('0, 0x3f001300, 0, 0x3f001300')
C = parse('0, 0x56061700, 0, 0x56061700')
print('A', bytes(A).hex())
print('B', bytes(B).hex())
print('C', bytes(C).hex())

def tau_log(A, B, C):
    Pf = pullback(int.from_bytes(bytes(B), 'little'), Linv)
    Qf = pullback(int.from_bytes(bytes(C), 'little'), lambda s: Linv(sigma(L(s))))
    p = np.array([(Pf >> (8 * i)) & 0xff for i in range(16)], dtype=np.uint8)
    q = np.array([(Qf >> (8 * i)) & 0xff for i in range(16)], dtype=np.uint8)
    v1 = np.abs(np.array([LAT[int(p[i]), int(q[i])] for i in range(16)], dtype=np.int64))
    cb = (C ^ B).astype(int); a = A.astype(int)
    v2 = np.abs(np.array([ILAT[cb[i], a[i]] for i in range(16)], dtype=np.int64))
    v3 = np.abs(np.array([ILAT[int(C[i]), a[i]] for i in range(16)], dtype=np.int64))
    lg = 0.0
    for i in range(16):
        if v1[i] == 0 or v2[i] == 0 or v3[i] == 0:
            if v1[i] == 0 and v2[i] == 0 and v3[i] == 0:
                return float('-inf')
        lg += math.log2(max(v1[i], 1) / 256) + math.log2(max(v2[i], 1) / 256) + math.log2(max(v3[i], 1) / 256)
    return lg

print('log2 tau =', tau_log(A, B, C), 'expected -54.771')
