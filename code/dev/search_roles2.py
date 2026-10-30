# Broader search: can the four meaningful column conditions of SNOW-A.py be satisfied
# by ANY natural assignment of the paper's masks (incl. sigma and L images)?
import sys, io, contextlib, itertools
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
with contextlib.redirect_stdout(io.StringIO()):
    from verify import L, Linv, sigma, pullback, to_bytes, to_word, parse, opt, des

def act(w): return [(w >> (8 * i)) & 0xff != 0 for i in range(16)]
def col_ok(inp, outp):
    a, b = act(inp), act(outp)
    for c in range(4):
        idx = [4 * c + 0, 4 * c + 5, 4 * c + 10, 4 * c + 15]
        s = sum(a[i % 16] for i in idx) + sum(b[4 * c + j] for j in range(4))
        if s != 0 and s < 5:
            return False
    return True
def rev(u, v): return col_ok(v, u)

def S(w): return to_word(sigma(to_bytes(w)))
def Lf(w): return to_word(L(to_bytes(w)))
def Li(w): return to_word(Linv(to_bytes(w)))

def pool(tri):
    A = to_word(tri['A']); B = to_word(tri['B']); C = to_word(tri['C'])
    P = pullback(B, Linv)
    Q = pullback(C, lambda s: Linv(sigma(L(s))))
    base = dict(A=A, B=B, C=C, P=P, Q=Q,
                CB=A ^ B, sigmaC=S(C), sigmaP=S(P), sigmaQ=S(Q), sigmaA=S(A))
    out = {}
    for k, v in base.items():
        out[k] = v
        out['L.' + k] = Lf(v)
        out['Li.' + k] = Li(v)
        out['sL.' + k] = S(Lf(v))
    return out

names = list(pool(opt).keys())
print('candidate masks:', len(names))
for label, tri in [('optimal', opt), ("designers'", des)]:
    m = pool(tri)
    best = []
    for a, b, g in itertools.product(names, repeat=3):
        sg = S(m[g])
        conds = [(m[b], m[g]), (m[a], sg), (m[a], sg ^ m[b]), (m[a] ^ m[b], m[g] ^ sg)]
        for mask_bits in range(16):
            ok = all((rev(*c) if (mask_bits >> i) & 1 else col_ok(*c)) for i, c in enumerate(conds))
            if ok:
                best.append(((a, b, g), mask_bits))
                break
    print('%s: %d assignments work (of %d^3)' % (label, len(best), len(names)))
    for x in best[:12]:
        print('   ', x)
