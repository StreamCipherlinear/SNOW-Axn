"""Numerically audit the single-mask model and statistical constants.

These calculations assume independent noise, the stated collision model, and
an idealised keystream of unbounded length for one key and IV.  They do not
establish the validity of the vectorial attack or the physical data limit.
"""
import math

LOG2 = math.log2
PS = 0.999992
L = 512
TAU_BEST = 2401 / 2 ** 66
TAU_DESIGNER = 1225 / 2 ** 66


def normal_upper_quantile(probability):
    assert 0.0 < probability < 0.5
    low, high = 0.0, 50.0
    for _ in range(120):
        mid = (low + high) / 2.0
        if 0.5 * math.erfc(mid / math.sqrt(2.0)) > probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def logsum(*values):
    largest = max(values)
    return largest + LOG2(sum(2.0 ** (x - largest) for x in values))


def log_equations(tau, bg, folds):
    z = normal_upper_quantile(1 - PS)
    tail = normal_upper_quantile(2.0 ** (-bg - 1))
    # For folds independent observations, the equation correlation is tau^folds.
    return -2 * folds * LOG2(abs(tau)) + 2 * LOG2(z + tail)


def first_stage(tau, bg):
    equations = log_equations(tau, bg, 2)
    data = (equations + L + 1 - bg) / 2
    if equations <= L - bg:
        precomputation = data + 0.5
    else:
        precomputation = data + LOG2(1 + data)
    if bg > equations:
        decoding = bg + LOG2(1 + math.ceil(equations))
    else:
        decoding = logsum(LOG2(bg) + bg, equations)
    time = logsum(precomputation, decoding)
    memory = max(data, float(bg))
    return {'bg': bg, 'equations': equations, 'data': data,
            'precomputation': precomputation, 'decoding': decoding,
            'time': time, 'memory': memory}


def optimum(tau):
    rows = [first_stage(tau, bg) for bg in range(1, L + 1)]
    return min(rows, key=lambda row: max(row['data'], row['time'], row['memory']))


def fmt(row):
    return ' '.join(f'{key}={row[key]:.9f}' if key != 'bg' else f'bg={row[key]}'
                    for key in ('bg', 'equations', 'data', 'precomputation', 'decoding', 'time', 'memory'))


print(f'Phi^-1({PS})={normal_upper_quantile(1-PS):.12f}')
best = optimum(TAU_BEST)
designer = optimum(TAU_DESIGNER)
print('best     ', fmt(best))
print('designer ', fmt(designer))
assert best['bg'] == 241 and designer['bg'] == 243
assert abs(best['equations'] - 228.055877414) < 1e-8
assert abs(best['data'] - 250.027938707) < 1e-8
assert abs(best['time'] - 250.919215250) < 1e-8
print('improvements', *(f'{designer[k]-best[k]:.9f}' for k in ('equations', 'data', 'time')))

# Illustrative disjoint-pair second pass, then direct-observation third pass.
# For the second pass at most 2^135 bins can hold one unmatched observation.
m2 = log_equations(TAU_BEST, 136, 2)
n2 = logsum(m2 + 1, 135.0)
t2 = logsum(n2, m2, 136 + LOG2(136))
n3 = log_equations(TAU_BEST, 135, 1)
t3 = logsum(n3, 135 + LOG2(135))
print(f'later passes: M2={m2:.9f} N2={n2:.9f} T2={t2:.9f} N3={n3:.9f} T3={t3:.9f}')
print(f'combined data={logsum(best["data"], n2, n3):.9f} '
      f'combined time={logsum(best["time"], t2, t3):.9f}')
print(f'three-pass success lower bound by union bound={1 - 3*(1-PS):.9f}')

# Compare YJM20's order-of-magnitude rule with the separately specified
# Gaussian test at alpha=2^-32 and beta=2^-17.  The underlying sample
# independence and normal approximation remain assumptions.
alpha, beta = 2.0 ** -32, 2.0 ** -17
coefficient = (normal_upper_quantile(alpha) + normal_upper_quantile(beta)) ** 2
log_coefficient = LOG2(coefficient)
log_single_reciprocal = -6 * LOG2(TAU_BEST)
print(f'Gaussian coefficient={coefficient:.9f}, log2={log_coefficient:.9f}')
print(f'one-mask reciprocal exponent={log_single_reciprocal:.9f}, '
      f'Gaussian sample exponent={log_single_reciprocal + log_coefficient:.9f}')
