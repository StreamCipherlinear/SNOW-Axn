"""Recompute the GF(2) rank of the 24 strongest complete triples.

Inputs: the first part of verify_space.py, executed to obtain the candidates.
Outputs: one line with the rank of the 24 triples.
Paper object: Section 4.2; a one-off check, verify_list.py reports the same rank.
"""

import sys, os, math, io, contextlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'verify_space.py'), encoding='utf-8').read()
head = src.split("print('top mask")[0]
G = {'__name__': '__main__', '__file__': os.path.join(HERE, 'verify_space.py')}
exec(compile(head, 'verify_space_head', 'exec'), G)
globals().update(G)

vecs = []
for k in range(24):
    Ab, Bb, Cb, p, q = cands[k]
    v = 0
    for i in range(16):
        v |= int(Ab[i]) << (8 * i)
        v |= int(Bb[i]) << (128 + 8 * i)
        v |= int(Cb[i]) << (256 + 8 * i)
    vecs.append(v)

basis = []
for v in vecs:
    x = v
    for b in basis:
        x = min(x, x ^ b)
    if x:
        basis.append(x)
        basis.sort(reverse=True)
print('rank of the 24 strongest masks =', len(basis))
