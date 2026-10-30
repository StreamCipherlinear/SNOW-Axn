"""Extend the nested space search from k = 20 to k = 24 without materialising
2^24 elements: keep the 2^20 elements of V20 and iterate over the 2^t subsets
of the t new basis vectors, accumulating the SEI in log space.
"""
import sys, os, math, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'verify_space.py'), encoding='utf-8').read()
head = src.split("print('top mask")[0]
G = {'__name__': '__main__', '__file__': os.path.join(HERE, 'verify_space.py')}
exec(compile(head, 'verify_space_head', 'exec'), G)
globals().update(G)

def lg_of_arr(Ab, Bb, Cb, p, q):
    cb = (Cb ^ Bb).astype(np.int64); a = Ab.astype(np.int64)
    w1 = absL[p, q]; w2 = absI[cb, a]; w3 = absI[Cb.astype(np.int64), a]
    bad = (w1 == 0).any(1) | (w2 == 0).any(1) | (w3 == 0).any(1)
    lg = (np.log2(np.maximum(w1, 1) / 256).sum(1) + np.log2(np.maximum(w2, 1) / 256).sum(1)
          + np.log2(np.maximum(w3, 1) / 256).sum(1))
    lg[bad] = -np.inf
    ze = ~((Ab.astype(np.int64) | Bb.astype(np.int64) | Cb.astype(np.int64)).any(1))
    lg[ze] = -np.inf
    return lg

# build V20 by doubling, 1 -> 2^20
V = (np.zeros((1, 16), np.uint8),) * 3 + (np.zeros((1, 16), np.uint8), np.zeros((1, 16), np.uint8))
for k in range(1, 21):
    Ab, Bb, Cb, p, q = cands[k - 1]
    V = tuple(np.concatenate([V[i], V[i] ^ x]) for i, x in enumerate((Ab, Bb, Cb, p, q)))
print('V20 elements', len(V[0]))

def log2_sei_chunks(offsets):
    acc_max = float('-inf'); acc_sum = 0.0
    for (oA, oB, oC, op, oq) in offsets:
        lg = lg_of_arr(V[0] ^ oA, V[1] ^ oB, V[2] ^ oC, V[3] ^ op, V[4] ^ oq)
        fin = lg[np.isfinite(lg)]
        if fin.size == 0:
            continue
        m = 4.0 * fin.max()
        if m > acc_max:
            acc_sum = acc_sum * 2.0 ** (acc_max - m) + float(np.exp2(4.0 * fin - m).sum())
            acc_max = m
        else:
            acc_sum += float(np.exp2(4.0 * fin - acc_max).sum())
    return acc_max + math.log2(acc_sum)

out = []
zero = np.zeros(16, np.uint8)
offs = [(zero, zero, zero, zero, zero)]
for t in range(1, 5):
    Ab, Bb, Cb, p, q = cands[20 + t - 1]
    offs = offs + [(oA ^ Ab, oB ^ Bb, oC ^ Cb, op ^ p, oq ^ q) for (oA, oB, oC, op, oq) in offs]
    L = log2_sei_chunks(offs)
    k = 20 + t
    out.append((k, L))
    Nreq = 4 * k * math.log(2) / float(np.exp2(L))
    T = 2 * Nreq + k * 2 ** k
    print('k=%2d  log2 SEI=%13.4f  (cached cost with d1=k: log2 T=%.4f)' % (k, L, math.log2(T)))

with open(os.path.join(HERE, 'sei_nested_24.json'), 'w') as f:
    json.dump([r[1] for r in out], f)
print('written sei_nested_24.json')
