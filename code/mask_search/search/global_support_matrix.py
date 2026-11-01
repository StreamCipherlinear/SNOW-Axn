"""Independently validate the byte-field matrices used by the support census.

The field is GF(2^8) modulo 0x11b.  Bit masks are converted to trace-dual
field coordinates before applying field matrices.  Run with Python 3.
"""
import contextlib
import io
import itertools
import random
import re
import sys
from pathlib import Path

SR = (0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11)
SIGMA = (0, 7, 14, 11, 4, 13, 10, 1, 8, 15, 6, 3, 12, 5, 2, 9)
MC = ((2, 3, 1, 1), (1, 2, 3, 1), (1, 1, 2, 3), (3, 1, 1, 2))
INV_MC = ((14, 11, 13, 9), (9, 14, 11, 13), (13, 9, 14, 11), (11, 13, 9, 14))
GROUPS = ((0, 5, 10, 15), (3, 4, 9, 14), (2, 7, 8, 13), (1, 6, 11, 12))


def mul(a, b):
    result = 0
    while b:
        if b & 1:
            result ^= a
        a <<= 1
        if a & 256:
            a ^= 0x11b
        b >>= 1
    return result


def power(a, n):
    result = 1
    while n:
        if n & 1:
            result = mul(result, a)
        a = mul(a, a)
        n >>= 1
    return result


def trace(a):
    result = 0
    for _ in range(8):
        result ^= a
        a = mul(a, a)
    assert result in (0, 1)
    return result


mask_from_field = [sum(trace(mul(a, 1 << j)) << j for j in range(8)) for a in range(256)]
assert len(set(mask_from_field)) == 256
field_from_mask = [0] * 256
for field, mask in enumerate(mask_from_field):
    field_from_mask[mask] = field
assert mask_from_field[1] == 0xa0


def mix(state, matrix):
    result = [0] * 16
    for c in range(4):
        for r in range(4):
            for j in range(4):
                result[4*c+r] ^= mul(matrix[r][j], state[4*c+j])
    return result


def linear(state):
    return mix([state[SR[i]] for i in range(16)], MC)


def inverse(state):
    mixed = mix(state, INV_MC)
    result = [0] * 16
    for i in range(16):
        result[SR[i]] = mixed[i]
    return result


def sigma(state):
    return [state[SIGMA[i]] for i in range(16)]


def to_word(state):
    return int.from_bytes(bytes(state), 'little')


def to_bytes(word):
    return list(word.to_bytes(16, 'little'))


def binary_columns(function):
    return [to_word(function(to_bytes(1 << i))) for i in range(128)]


def pullback(columns, mask):
    return sum(((mask & image).bit_count() & 1) << j for j, image in enumerate(columns))


def field_matrix(columns):
    result = [[0] * 16 for _ in range(16)]
    for j in range(16):
        image = pullback(columns, mask_from_field[1] << (8 * j))
        for i in range(16):
            result[i][j] = field_from_mask[(image >> (8 * i)) & 255]
    return result


def field_pullback(matrix, mask):
    inp = [field_from_mask[(mask >> (8 * j)) & 255] for j in range(16)]
    out = []
    for row in matrix:
        z = 0
        for coefficient, x in zip(row, inp):
            z ^= mul(coefficient, x)
        out.append(mask_from_field[z])
    return to_word(out)


def nonsingular(matrix):
    rows = [list(row) for row in matrix]
    n = len(rows)
    for c in range(n):
        pivot = next((i for i in range(c, n) if rows[i][c]), None)
        if pivot is None:
            return False
        rows[c], rows[pivot] = rows[pivot], rows[c]
        factor = power(rows[c][c], 254)
        rows[c] = [mul(factor, x) for x in rows[c]]
        for i in range(c + 1, n):
            factor = rows[i][c]
            if factor:
                rows[i] = [x ^ mul(factor, y) for x, y in zip(rows[i], rows[c])]
    return True


for j in range(128):
    x = to_bytes(1 << j)
    assert inverse(linear(x)) == x and linear(inverse(x)) == x and sigma(sigma(x)) == x
p_cols = binary_columns(inverse)
q_cols = binary_columns(lambda x: inverse(sigma(linear(x))))
p_matrix = field_matrix(p_cols)
q_matrix = field_matrix(q_cols)
for j in range(128):
    for cols, matrix in ((p_cols, p_matrix), (q_cols, q_matrix)):
        assert pullback(cols, 1 << j) == field_pullback(matrix, 1 << j)
print('Both field matrices agree with all 256 binary basis pullbacks.')

# Independently check the hard-coded first census matrix.
source = (Path(__file__).resolve().parent / 'global_support_census.cs').read_text(encoding='utf-8')
section = source.split('private static readonly byte[,] Q = {', 1)[1].split('};', 1)[0]
printed_matrix = [int(x) for x in re.findall(r'\d+', section)]
assert printed_matrix == [x for row in q_matrix for x in row]
print('The census Q matrix matches the independently derived trace-dual matrix.')

# The four P blocks are independent and every square minor is invertible.
minor_count = 0
for block, columns in enumerate(GROUPS):
    rows = tuple(range(4 * block, 4 * block + 4))
    for i in rows:
        for j in range(16):
            if j not in columns:
                assert p_matrix[i][j] == 0
    for size in range(1, 5):
        for chosen_rows in itertools.combinations(rows, size):
            for chosen_columns in itertools.combinations(columns, size):
                matrix = [[p_matrix[i][j] for j in chosen_columns] for i in chosen_rows]
                assert nonsingular(matrix), (block, chosen_rows, chosen_columns)
                minor_count += 1
assert minor_count == 276
print('All 276 square minors of the four P blocks are nonzero.')

# Cross-check a separate, pre-existing bit-mask implementation on random masks.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import os, sys
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_CODE, 'mask_search', 'correlation') not in sys.path:
    sys.path.insert(0, os.path.join(_CODE, 'mask_search', 'correlation'))
with contextlib.redirect_stdout(io.StringIO()):
    import verify
rng = random.Random(20261030)
for _ in range(50):
    x = rng.getrandbits(128)
    assert pullback(p_cols, x) == verify.pullback(x, verify.Linv)
    assert pullback(q_cols, x) == verify.pullback(x, lambda s: verify.Linv(verify.sigma(verify.L(s))))
print('Independent byte operations agree with verify.py on 100 random pullbacks.')
