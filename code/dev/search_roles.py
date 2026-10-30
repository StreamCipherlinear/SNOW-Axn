# Which assignment of the model's masks (alpha, beta, gamma) to the paper's masks
# makes the four meaningful AESRound column conditions hold on the two known triples?
import sys, io, contextlib
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

def masks(tri):
    A = to_word(tri['A']); B = to_word(tri['B']); C = to_word(tri['C'])
    P = pullback(B, Linv)
    Q = pullback(C, lambda s: Linv(sigma(L(s))))
    S = lambda w: to_word(sigma(to_bytes(w)))
    return dict(A=A, B=B, C=C, P=P, Q=Q, sC=S(C), sP=S(P), sQ=S(Q), sA=S(A))

names = ['A', 'B', 'C', 'P', 'Q', 'sA', 'sC', 'sP', 'sQ']
for label, tri in [('optimal', opt), ("designers'", des)]:
    m = masks(tri)
    print('=== %s ===' % label)
    hits = []
    for a in names:
        for b in names:
            for g in names:
                sg = to_word(sigma(to_bytes(m[g])))
                c1 = col_ok(m[b], m[g])
                c2 = col_ok(m[a], sg)
                c3 = col_ok(m[a], sg ^ m[b])
                c5 = col_ok(m[a] ^ m[b], m[g] ^ sg)
                if c1 and c2 and c3 and c5:
                    hits.append((a, b, g))
    print('  assignments satisfying the four conditions:', hits)
    # which A1 makes the fifth condition true
    for (a, b, g) in hits:
        ok = []
        for x in names:
            if col_ok(m[x], m[b]):
                ok.append(x)
        print('    for (a,b,g)=(%s,%s,%s): A1 candidates with col(A1,beta) ok -> %s' % (a, b, g, ok))
    print()
