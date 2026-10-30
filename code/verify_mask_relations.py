"""Check separated-time linear masking ranks using only the design recurrence.

Byte 0 is the least significant byte in a 128-bit word.  The test constructs
all matrix rows from forward state transitions, without importing verify.py.
The nonzero three-time kernel is not itself a biased distinguishing attack.
"""
SIGMA = (0, 7, 14, 11, 4, 13, 10, 1, 8, 15, 6, 3, 12, 5, 2, 9)


def forward(state):
    words = list(state.to_bytes(64, 'little'))
    out = words[16:]
    for i in range(16):
        value = words[i]
        modulus = 0x139 if i < 8 else 0x1a5
        product = ((value << 1) ^ (modulus if value & 128 else 0)) & 255
        out.append(product ^ words[i + 7] ^ words[i + 32])
    return int.from_bytes(bytes(out), 'little')


def linear_output(state):
    words = state.to_bytes(64, 'little')
    return int.from_bytes(bytes(words[i] ^ words[16 + SIGMA[i]] for i in range(16)), 'little')


# Rows of the linear keystream part at t=0,...,4 as 512-bit masks.
rows = [[0] * 128 for _ in range(5)]
for bit in range(512):
    state = 1 << bit
    for t in range(5):
        output = linear_output(state)
        while output:
            lowest = output & -output
            rows[t][lowest.bit_length() - 1] |= 1 << bit
            output ^= lowest
        state = forward(state)


def kernel(times):
    pivots = {}
    dependencies = []
    for pos, t in enumerate(times):
        for bit, row in enumerate(rows[t]):
            vector = row
            coefficients = 1 << (128 * pos + bit)
            while vector:
                pivot = vector.bit_length()
                if pivot in pivots:
                    old_vector, old_coefficients = pivots[pivot]
                    vector ^= old_vector
                    coefficients ^= old_coefficients
                else:
                    pivots[pivot] = (vector, coefficients)
                    break
            if not vector:
                dependencies.append(coefficients)
    print(f'T={times} rank={len(pivots)} nullity={128*len(times)-len(pivots)}')
    return dependencies


dependencies = kernel((0, 2, 3))
assert len(dependencies) == 1
masks = [(dependencies[0] >> (128 * i)) & ((1 << 128) - 1) for i in range(3)]
assert all(masks)
print('witness masks:', *(x.to_bytes(16, 'little').hex() for x in masks))
for bit in range(512):
    state = 1 << bit
    value = 0
    for t in range(4):
        if t in (0, 2, 3):
            value ^= (masks[(0, 2, 3).index(t)] & linear_output(state)).bit_count() & 1
        state = forward(state)
    assert value == 0
# The equal-mask polynomial is not an annihilator, even on the first basis state.
assert 1 ^ forward(forward(1)) ^ forward(forward(forward(1)))
assert len(kernel((0, 1, 2, 4))) == 32
print('All 512 initial-state basis checks passed.')
