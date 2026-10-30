# Structural check of Section 3.2: for a weight-6 pattern mu the admissible sets
#   B_mu = { B : mu(B) <= mu, (B.L^-1)|_{mu^c} = 0 },
#   C_mu = { C : mu(C) <= mu, (C.L^-1 s^-1 L)|_{mu^c} = 0 }
# have dimensions 16 and 8, hence |Theta_mu| = 2^24.
import sys
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
from verify import Linv, L, sigma, pullback, to_bytes, to_word, act

PATTERNS = {
    'optimal {4,5,6,12,13,14}': (4, 5, 6, 12, 13, 14),
    "designers {0,2,3,8,10,11}": (0, 2, 3, 8, 10, 11),
    '{4,6,7,12,14,15}': (4, 6, 7, 12, 14, 15),
    '{0,1,2,8,9,10}': (0, 1, 2, 8, 9, 10),
}

def kernel_dim(mu, f):
    """dim of { m : supp(m) subset of mu, f(m) vanishes off mu }"""
    free = [(b, t) for b in mu for t in range(8)]
    outc = [b for b in range(16) if b not in mu]
    rows = []
    for (b, t) in free:
        w = f(1 << (8 * b + t))
        v = 0
        k = 0
        for ob in outc:
            for ot in range(8):
                if (w >> (8 * ob + ot)) & 1:
                    v |= 1 << k
                k += 1
        rows.append(v)
    # GF(2) rank by elimination on python ints
    basis = {}
    for r in rows:
        while r:
            p = r.bit_length() - 1
            if p in basis:
                r ^= basis[p]
            else:
                basis[p] = r
                break
    return len(free) - len(basis)

for name, mu in PATTERNS.items():
    dB = kernel_dim(mu, lambda w: pullback(w, Linv))
    dC = kernel_dim(mu, lambda w: pullback(w, lambda s: Linv(sigma(L(s)))))
    print('%-28s  dim B = %2d   dim C = %2d   |Theta| = 2^%d' % (name, dB, dC, dB + dC))

print("\npaper: dim B = 16, dim C = 8, |Theta_mu| = 2^24 for all four patterns")
