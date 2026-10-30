"""Check simultaneous low-bit collisions for the first two fixed W masks.

The 512-bit state uses H0 in bits 0..127 and H3 in bits 384..511.  Within
words byte 0 is least significant.  The input masks come from data/w24.json.
This verifies the reported coordinate split, not arbitrary state bases.
"""
import contextlib
import io
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
with contextlib.redirect_stdout(io.StringIO()):
    from verify import Linv, sigma, pullback

entries = json.loads((HERE.parent / 'data' / 'w24.json').read_text(encoding='utf-8'))['entries'][:2]
assert len(entries) == 2


def state_mask(entry):
    a = int.from_bytes(bytes.fromhex(entry['A']), 'little')
    b = int.from_bytes(bytes.fromhex(entry['B']), 'little')
    c = int.from_bytes(bytes.fromhex(entry['C']), 'little')
    return (a | (pullback(b, Linv) << 128)
            | (pullback(c, lambda x: Linv(sigma(x))) << 256)
            | (pullback(c, Linv) << 384))


masks = [state_mask(e) for e in entries]
assert masks[0] and masks[1] and masks[0] != masks[1]


def forward(state):
    words = list(state.to_bytes(64, 'little'))
    out = words[16:]
    for i in range(16):
        x = words[i]
        modulus = 0x139 if i < 8 else 0x1a5
        product = ((x << 1) ^ (modulus if x & 128 else 0)) & 255
        out.append(product ^ words[i + 7] ^ words[i + 32])
    return int.from_bytes(bytes(out), 'little')


# Derive the dual transition directly by transposing its 512 binary columns.
columns = [forward(1 << j) for j in range(512)]
rows = [0] * 512
for j, column in enumerate(columns):
    while column:
        bit = column & -column
        rows[bit.bit_length() - 1] |= 1 << j
        column ^= bit
byte_maps = []
for pos in range(64):
    table = [0] * 256
    for value in range(1, 256):
        bit = value & -value
        table[value] = table[value ^ bit] ^ rows[8 * pos + bit.bit_length() - 1]
    byte_maps.append(table)


def pull_transition(mask):
    out = 0
    for pos in range(64):
        out ^= byte_maps[pos][(mask >> (8 * pos)) & 255]
    return out


rng = random.Random(20261030)
for _ in range(100):
    x = rng.getrandbits(512)
    y = rng.getrandbits(512)
    assert ((pull_transition(y) & x).bit_count() & 1) == ((y & forward(x)).bit_count() & 1)
print('Forward transition and mask pullback agree on 100 random tests.')

# Express g1 as g0 P(F) in a cyclic Krylov basis.
basis = {}
x = masks[0]
for k in range(512):
    vector, combination = x, 1 << k
    while vector:
        pivot = vector.bit_length()
        if pivot in basis:
            old_vector, old_combination = basis[pivot]
            vector ^= old_vector
            combination ^= old_combination
        else:
            basis[pivot] = (vector, combination)
            break
    x = pull_transition(x)
assert len(basis) == 512
vector = masks[1]
polynomial = 0
while vector:
    old_vector, old_combination = basis[vector.bit_length()]
    vector ^= old_vector
    polynomial ^= old_combination
print('Krylov basis rank=512, relation polynomial weight=', polynomial.bit_count())

# Derive the complete linear action of P(F) on arbitrary 512-bit masks.
action = []
for j in range(512):
    x, acc = 1 << j, 0
    for k in range(512):
        if (polynomial >> k) & 1:
            acc ^= x
        x = pull_transition(x)
    action.append(acc)
actual = 0
for j in range(512):
    if (masks[0] >> j) & 1:
        actual ^= action[j]
assert actual == masks[1]
print('The derived polynomial maps the first mask to the second.')

for bg in (240, 241, 243):
    for orientation in ('high-last', 'high-first'):
        low = range(512-bg) if orientation == 'high-last' else range(bg, 512)
        low = tuple(low)
        constraints = [1 << j for j in low]
        for j in low:
            row = 0
            for k in range(512):
                row |= ((action[k] >> j) & 1) << k
            constraints.append(row)
        pivots = {}
        for vector in constraints:
            while vector:
                pivot = vector.bit_length()
                if pivot in pivots:
                    vector ^= pivots[pivot]
                else:
                    pivots[pivot] = vector
                    break
        dimension = 512 - len(pivots)
        print(f'Bg={bg} orientation={orientation} constraint_rank={len(pivots)} kernel_dimension={dimension}')
        assert dimension == 0
print('No nonzero simultaneous collision difference exists for these parameters.')
