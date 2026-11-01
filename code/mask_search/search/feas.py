# Is the repaired SNOW-A.py model feasible at all?  Random sparse search over the
# activity patterns (the model is pure 0/1, so a witness is a full certificate).
import random
random.seed(7)
sig = [0, 7, 14, 11, 4, 13, 10, 1, 8, 15, 6, 3, 12, 5, 2, 9]

def col_ok(inp, out):
    for c in range(4):
        idx = [4 * c + 0, 4 * c + 5, 4 * c + 10, 4 * c + 15]
        s = sum(inp[i % 16] for i in idx) + sum(out[4 * c + j] for j in range(4))
        if s != 0 and s < 5:
            return False
    return True

def check(al, be, ga):
    sg = [ga[sig[i]] for i in range(16)]
    sgb = [sg[i] ^ be[i] for i in range(16)]
    a2 = [al[i] ^ be[i] for i in range(16)]
    Bm = [ga[i] ^ sg[i] for i in range(16)]
    return (col_ok(be, ga) and col_ok(al, sg) and col_ok(al, sgb) and col_ok(a2, Bm))

best = None
found = None
for trial in range(400000):
    w = [random.randint(0, 6) for _ in range(3)]
    al = [0] * 16; be = [0] * 16; ga = [0] * 16
    for arr, k in ((al, w[0]), (be, w[1]), (ga, w[2])):
        for i in random.sample(range(16), k):
            arr[i] = 1
    if sum(be) == 0:
        continue
    if check(al, be, ga):
        found = (al, be, ga)
        break
print('feasible witness found:', bool(found))
if found:
    al, be, ga = found
    for nm, v in (('alpha', al), ('beta', be), ('gamma', ga)):
        print('  %-6s' % nm, v, 'w=%d' % sum(v))
    print('  objective 2HW(alpha)+HW(beta) =', 2 * sum(al) + sum(be))
