"""Reproduce the strict-support integer search over all four feasible patterns.

Run from any directory with `python -B global_optimum_scan.py`.  The S-box and
linear layers come from verify.py.  Support feasibility is checked separately
by global_support_census.cs and global_support_verify.cs.
"""
import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
with contextlib.redirect_stdout(io.StringIO()):
    import verify as v


def transpose_cols(function):
    images = [v.to_word(function(v.to_bytes(1 << j))) for j in range(128)]
    cols = [0] * 128
    for j, image in enumerate(images):
        while image:
            bit = (image & -image).bit_length() - 1
            image &= image - 1
            cols[bit] |= 1 << j
    return cols


def apply(cols, x):
    result = 0
    while x:
        bit = (x & -x).bit_length() - 1
        x &= x - 1
        result ^= cols[bit]
    return result


def nullspace_for_support(support, cols):
    free_bits = [8 * i + j for i in support for j in range(8)]
    rows = []
    for k in range(128):
        if k // 8 not in support:
            rows.append(sum(((cols[bit] >> k) & 1) << j for j, bit in enumerate(free_bits)))
    pivots = []
    rank = 0
    for j in range(len(free_bits)):
        row = next((i for i in range(rank, len(rows)) if (rows[i] >> j) & 1), None)
        if row is None:
            continue
        rows[rank], rows[row] = rows[row], rows[rank]
        for i in range(len(rows)):
            if i != rank and ((rows[i] >> j) & 1):
                rows[i] ^= rows[rank]
        pivots.append(j)
        rank += 1
    result = []
    outside = sum(255 << (8 * i) for i in range(16) if i not in support)
    for f in range(len(free_bits)):
        if f in pivots:
            continue
        x = 1 << free_bits[f]
        for i, p in enumerate(pivots):
            if (rows[i] >> f) & 1:
                x ^= 1 << free_bits[p]
        assert apply(cols, x) & outside == 0
        result.append(x)
    return result


def expand(basis, support):
    values = np.zeros((1 << len(basis), len(support)), dtype=np.uint8)
    for j, x in enumerate(basis):
        vector = np.array([(x >> (8 * i)) & 255 for i in support], dtype=np.uint8)
        values[1 << j:2 << j] = values[:1 << j] ^ vector
    return values


P = transpose_cols(v.Linv)
Q = transpose_cols(lambda s: v.Linv(v.sigma(v.L(s))))
LAT = (np.abs(v.LAT) // 4).astype(np.uint8)
ILAT = (np.abs(v.ILAT) // 4).astype(np.uint8)
max_two = np.zeros((256, 256), dtype=np.uint8)
number_of_maxima = np.zeros((256, 256), dtype=np.uint8)
for x in range(1, 256):
    products = ILAT[x, 1:].astype(np.uint16)[None, :] * ILAT[1:, 1:].astype(np.uint16)
    max_two[x, 1:] = np.max(products, axis=1).astype(np.uint8)
    number_of_maxima[x, 1:] = np.sum(products == max_two[x, 1:, None], axis=1).astype(np.uint8)
assert LAT.max() == 64 and int(max_two.max()) == 64

expected = {
    0x0707: (11082491, 8084777718513664, 2, 50),
    0x0d0d: (11193038, 8084777718513664, 2, 125),
    0x7070: (11093111, 10559709673160704, 1, 100),
    0xd0d0: (11032069, 8084777718513664, 2, 125),
}
for pattern, expected_row in expected.items():
    support = [i for i in range(16) if (pattern >> i) & 1]
    b_basis = nullspace_for_support(support, P)
    c_basis = nullspace_for_support(support, Q)
    assert (len(b_basis), len(c_basis)) == (16, 8)
    b = expand(b_basis, support)
    pb = expand([apply(P, x) for x in b_basis], support)
    c = expand(c_basis, support)
    qc = expand([apply(Q, x) for x in c_basis], support)
    valid_b = np.all(pb != 0, axis=1)
    valid_c = np.all((c != 0) & (qc != 0), axis=1)
    assert int(valid_b.sum()) == 65025 and int(valid_c.sum()) == 255
    best = best_pairs = best_triples = nonzero = 0
    moments = {k: 0.0 for k in range(2, 7)}
    levels = set()
    for cv, qv in zip(c[valid_c], qc[valid_c]):
        score = valid_b.astype(np.uint64)
        for i in range(6):
            cb = b[:, i] ^ cv[i]
            two = max_two[cb, cv[i]]
            third = LAT[pb[:, i], qv[i]]
            # With strict support, nonzero LAT magnitudes are at most 32.
            assert int(third.max()) <= 8 and int(two.max()) <= 64
            score *= third.astype(np.uint64) * two.astype(np.uint64)
        nonzero += int(np.count_nonzero(score))
        if pattern == 0x7070:
            positive = score[score != 0]
            levels.update(int(x) for x in np.unique(positive))
            correlations = positive.astype(np.float64) * 2.0 ** -108
            for k in moments:
                moments[k] += float(np.sum(correlations ** (2 * k)))
        maximum = int(score.max())
        if maximum > best:
            best = maximum
            best_pairs = best_triples = 0
        if maximum == best:
            indices = np.flatnonzero(score == best)
            best_pairs += len(indices)
            for index in indices:
                ties = 1
                for i in range(6):
                    ties *= int(number_of_maxima[b[index, i] ^ cv[i], cv[i]])
                best_triples += ties
    actual = (nonzero, best, best_pairs, best_triples)
    print(f'mu={pattern:04x} nonzero={nonzero} maxN={best} '
          f'log2tau={math.log2(best)-108:.9f} bestpairs={best_pairs} triples={best_triples}')
    assert actual == expected_row, (pattern, actual, expected_row)
    if pattern == 0x7070:
        assert len(levels) == 3295
        print(f'  exact magnitude levels={len(levels)}')
        expected_logs = {2: -213.38180625962818, 3: -326.18715805988114,
                         4: -437.0841790171351, 5: -547.1403922570111,
                         6: -656.9258006008943}
        for k, total in moments.items():
            value = math.log2(total)
            print(f'  log2 sum tau^{2*k} = {value:.12f}')
            assert abs(value - expected_logs[k]) < 1e-9, (k, value)
print('All four strict-support integer scans passed.')
