"""Test the four AESRound column conditions for four mask assignments.

Inputs: the optimal and designers' triples from verify.py.
Outputs: for each assignment, the column conditions that fail.
Paper object: none; development aid for the mask roles of Section 3.
"""

import sys, io, contextlib
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
with contextlib.redirect_stdout(io.StringIO()):
    from verify import L, Linv, sigma, pullback, to_bytes, to_word, parse, opt, des

def act(w): return [(w >> (8 * i)) & 0xff != 0 for i in range(16)]
def col_ok(inp, outp):
    a, b = act(inp), act(outp)
    bad = []
    for c in range(4):
        idx = [4 * c + 0, 4 * c + 5, 4 * c + 10, 4 * c + 15]
        s = sum(a[i % 16] for i in idx) + sum(b[4 * c + j] for j in range(4))
        if s != 0 and s < 5:
            bad.append((c, s))
    return bad
def S(w): return to_word(sigma(to_bytes(w)))

for label, tri in [('optimal', opt), ("designers'", des)]:
    A = to_word(tri['A']); B = to_word(tri['B']); C = to_word(tri['C'])
    P = pullback(B, Linv)
    Q = pullback(C, lambda s: Linv(sigma(L(s))))
    m = dict(A=A, B=B, C=C, P=P, Q=Q)
    print('=== %s ===' % label)
    for nm, (a, b, g) in [('(A,B,C)', ('A', 'B', 'C')), ('(A,P,C)', ('A', 'P', 'C')),
                          ('(A,B,Q)', ('A', 'B', 'Q')), ('(A,P,Q)', ('A', 'P', 'Q'))]:
        sg = S(m[g])
        A2 = m[a] ^ m[b]
        Bm = m[g] ^ sg
        r = [('col(beta,gamma)', col_ok(m[b], m[g])),
             ('col(alpha,sig gamma)', col_ok(m[a], sg)),
             ('col(alpha,sig gamma^beta)', col_ok(m[a], sg ^ m[b])),
             ('col(A2,B)', col_ok(A2, Bm))]
        fails = [k for k, v in r if v]
        print('  %-9s  fails: %s' % (nm, fails if fails else 'NONE (all four hold)'))
