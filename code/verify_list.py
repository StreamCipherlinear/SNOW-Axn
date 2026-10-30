"""Enumerate the span W of the 24 triples in data/w24.json.

Inputs: ../data/w24.json.
Outputs: a JSON summary and ../data/w24.log with rank, counts and power sums.
Paper object: the span of W and its SEI, Table 6 and Section 5.
"""

import json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify import LAT, ILAT, L, Linv, sigma, pullback

MU = (4, 5, 6, 12, 13, 14)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'w24.json')
LOG = os.path.join(HERE, '..', 'data', 'w24.log')
d = json.load(open(DATA, encoding='utf-8'))
ents = d['entries']
print('entries', len(ents))

def to_bytes(h):
    return [int(h[2 * i:2 * i + 2], 16) for i in range(16)]

Ab_ = np.array([to_bytes(e['A']) for e in ents], dtype=np.uint8)
Bb_ = np.array([to_bytes(e['B']) for e in ents], dtype=np.uint8)
Cb_ = np.array([to_bytes(e['C']) for e in ents], dtype=np.uint8)

# rank over F2 of the complete triples
piv = {}
for k in range(len(ents)):
    v = int.from_bytes(bytes(Ab_[k]) + bytes(Bb_[k]) + bytes(Cb_[k]), 'little')
    while v:
        p = v.bit_length() - 1
        if p in piv:
            v ^= piv[p]
        else:
            piv[p] = v
            break
print('rank of complete triples =', len(piv))

Pb__ = np.array([[pullback(int.from_bytes(bytes(r), 'little'), Linv) >> (8 * i) & 0xff for i in range(16)] for r in Bb_], dtype=np.uint8)
Qb__ = np.array([[pullback(int.from_bytes(bytes(r), 'little'), lambda s: Linv(sigma(L(s)))) >> (8 * i) & 0xff for i in range(16)] for r in Cb_], dtype=np.uint8)

absL = np.abs(LAT.astype(np.int16))
absI = np.abs(ILAT.astype(np.int32))
ILa = np.abs(ILAT.astype(np.int32))
T12a = np.zeros((256, 256), dtype=np.uint8)
T12v = np.zeros((256, 256), dtype=np.int32)
for m1 in range(256):
    pr = ILa[m1][None, :] * ILa
    pr[:, 0] = 0
    T12a[m1] = pr.argmax(axis=1).astype(np.uint8)
    T12v[m1] = pr.max(axis=1)

def expand(basis, k, sel):
    return np.bitwise_xor.reduce(sel[:, :, None] * basis[None, :, :], axis=1)

CH = 1 << 20
nz = 0
strict = []
sum4 = 0.0
sum6 = 0.0
sum8 = 0.0
sum10 = 0.0
sum12 = 0.0
off = [i for i in range(16) if i not in MU]
t0 = time.time()
for start in range(0, 1 << 24, CH):
    idx = np.arange(start, start + CH, dtype=np.int64)
    s = ((idx[:, None] >> np.arange(24)) & 1).astype(np.uint8)
    A = expand(Ab_, 24, s); Bc = expand(Bb_, 24, s); Cc = expand(Cb_, 24, s)
    Pp = expand(Pb__, 24, s); Qq = expand(Qb__, 24, s)
    lg = np.zeros(CH); zr = np.zeros(CH, dtype=bool)
    for l in range(16):
        v1 = absL[Pp[:, l], Qq[:, l]].astype(np.float64)
        v2 = absI[Cc[:, l] ^ Bc[:, l], A[:, l]].astype(np.float64)
        v3 = absI[Cc[:, l], A[:, l]].astype(np.float64)
        zr |= (v1 == 0) | (v2 == 0) | (v3 == 0)
        lg += np.log2(np.maximum(v1, 1) / 256) + np.log2(np.maximum(v2, 1) / 256) + np.log2(np.maximum(v3, 1) / 256)
    if start == 0:
        zr[0] = True
    ok = (Bc[:, off].sum(1) == 0) & (Cc[:, off].sum(1) == 0) & (Pp[:, off].sum(1) == 0) & (Qq[:, off].sum(1) == 0)
    ok &= (A[:, off].sum(1) == 0) & ~zr
    st = np.ones(CH, dtype=bool)
    for i in MU:
        cbv = Cc[:, i] ^ Bc[:, i]
        st &= (A[:, i] == T12a[cbv, Cc[:, i]])
    hit = ok & st
    if hit.any():
        strict.extend(float(x) for x in lg[hit])
    nz += int((~zr).sum())
    sum4 += float(np.exp2(4.0 * lg[~zr]).sum())
    sum6 += float(np.exp2(6.0 * lg[~zr]).sum())
    sum8 += float(np.exp2(8.0 * lg[~zr]).sum())
    sum10 += float(np.exp2(10.0 * lg[~zr]).sum())
    sum12 += float(np.exp2(12.0 * lg[~zr]).sum())

res = {
    'rank': len(piv),
    'masks_with_nonzero_correlation': nz,
    'bytewise_optimal': len(strict),
    'log2_subset_sum6': math.log2(float(np.sum(np.exp2(6.0 * np.array(strict))))) if strict else None,
    'log2_full_sum4': math.log2(sum4),
    'log2_full_sum6': math.log2(sum6),
    'log2_full_sum8': math.log2(sum8),
    'log2_full_sum10': math.log2(sum10),
    'log2_full_sum12': math.log2(sum12),
    'seconds': round(time.time() - t0, 1),
}
print(json.dumps(res, indent=1))
with open(LOG, 'w', encoding='utf-8') as f:
    f.write('input: data/w24.json\n')
    f.write(json.dumps(res, indent=1) + '\n')
