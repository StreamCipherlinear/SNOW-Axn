"""Independent recomputation of SEI for k = 2, 3 and of a few elements of the
k = 24 space, using the direct per-triple formula (the one unit-tested against
tau* = 2^-54.770580) instead of the vectorised path.
"""
import sys, os, io, math, contextlib, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import os, sys
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_CODE, 'mask_search', 'correlation') not in sys.path:
    sys.path.insert(0, os.path.join(_CODE, 'mask_search', 'correlation'))
with contextlib.redirect_stdout(io.StringIO()):
    from verify import LAT, ILAT, L, Linv, sigma, pullback

src = open(os.path.join(HERE, 'verify_space.py'), encoding='utf-8').read()
head = src.split("print('top mask")[0]
G = {'__name__': '__main__', '__file__': os.path.join(HERE, 'verify_space.py')}
exec(compile(head, 'verify_space_head', 'exec'), G)
globals().update(G)


def direct_log2_tau(Ab, Bb, Cb):
    Pf = pullback(int.from_bytes(bytes(Bb), 'little'), Linv)
    Qf = pullback(int.from_bytes(bytes(Cb), 'little'), lambda s: Linv(sigma(L(s))))
    lg = 0.0
    for i in range(16):
        p = (Pf >> (8 * i)) & 0xFF
        q = (Qf >> (8 * i)) & 0xFF
        a = int(Ab[i]); b = int(Bb[i]); c = int(Cb[i])
        v1 = abs(int(LAT[p, q])); v2 = abs(int(ILAT[c ^ b, a])); v3 = abs(int(ILAT[c, a]))
        if v1 == 0 or v2 == 0 or v3 == 0:
            return float('-inf')
        lg += math.log2(v1 / 256) + math.log2(v2 / 256) + math.log2(v3 / 256)
    return lg


def sei_direct(k):
    total = 0.0
    for mask in range(1, 1 << k):
        Ab = np.zeros(16, np.uint8); Bb = np.zeros(16, np.uint8); Cb = np.zeros(16, np.uint8)
        for j in range(k):
            if (mask >> j) & 1:
                Ab ^= cands[j][0]; Bb ^= cands[j][1]; Cb ^= cands[j][2]
        if not (Ab.any() or Bb.any() or Cb.any()):
            continue
        lg = direct_log2_tau(Ab, Bb, Cb)
        if lg > float('-inf'):
            total += 2.0 ** (4.0 * lg)
    return math.log2(total)


for k in (1, 2, 3):
    print('k=%d  direct SEI = %.6f' % (k, sei_direct(k)))

random.seed(1)
sub = 0.0
for _ in range(200):
    mask = random.getrandbits(24) or 1
    Ab = np.zeros(16, np.uint8); Bb = np.zeros(16, np.uint8); Cb = np.zeros(16, np.uint8)
    for j in range(24):
        if (mask >> j) & 1:
            Ab ^= cands[j][0]; Bb ^= cands[j][1]; Cb ^= cands[j][2]
    lg = direct_log2_tau(Ab, Bb, Cb)
    if lg > float('-inf'):
        sub += 2.0 ** (4.0 * lg)
print('mean tau^4 over 200 sampled elements of the k=24 space: 2^%.3f' % (math.log2(sub / 200)))
