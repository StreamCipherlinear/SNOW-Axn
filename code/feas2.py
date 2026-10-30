# Structured feasibility: force conditions 1 and 2 by construction,
#   gamma = sigma(L(alpha)),  beta = L^{-1}(gamma),
# then check the remaining two conditions over all 2^16 choices of alpha.
import sys, io, contextlib
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
with contextlib.redirect_stdout(io.StringIO()):
    from verify import L, Linv, sigma, pullback, to_bytes, to_word

sig = [0, 7, 14, 11, 4, 13, 10, 1, 8, 15, 6, 3, 12, 5, 2, 9]
def col_ok(inp, out):
    for c in range(4):
        idx = [4 * c + 0, 4 * c + 5, 4 * c + 10, 4 * c + 15]
        s = sum(inp[i % 16] for i in idx) + sum(out[4 * c + j] for j in range(4))
        if s != 0 and s < 5:
            return False
    return True

def act128(w): return [(w >> (8 * i)) & 0xff != 0 for i in range(16)]
def to_mask(bits):
    w = 0
    for i, v in enumerate(bits):
        if v: w |= 1 << (8 * i)
    return w

hits = []
for pat in range(1 << 16):
    a = [(pat >> i) & 1 for i in range(16)]
    if sum(a) == 0:
        continue
    A = to_mask(a)
    G = to_word(sigma(to_bytes(to_word(L(to_bytes(A))))))   # gamma = sigma(L(alpha))
    Bt = pullback(G, L)                                     # beta = L^{-1}(gamma)
    if Bt == 0:
        continue
    sg = to_word(sigma(to_bytes(G)))
    sgb = sg ^ Bt
    a2 = A ^ Bt
    Bm = G ^ sg
    if col_ok(act128(Bt), act128(G)) and col_ok(act128(A), act128(sg)) \
       and col_ok(act128(A), act128(sgb)) and col_ok(act128(a2), act128(Bm)):
        hits.append((pat, 2 * sum(a) + sum(act128(Bt))))
        if len(hits) >= 6:
            break
print('structured hits:', len(hits))
for pat, obj in hits[:6]:
    a = [(pat >> i) & 1 for i in range(16)]
    A = to_mask(a)
    G = to_word(sigma(to_bytes(to_word(L(to_bytes(A))))))
    Bt = pullback(G, L)
    print('  alpha=%s beta=%s gamma=%s  obj=%d' % (a, act128(Bt), act128(G), obj))
