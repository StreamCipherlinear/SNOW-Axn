# (a) port of aes_p() from SNOW-A.cpp, compared with the verified L = MC o SR
# (b) structural census of the generated snow.py
import sys, os
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from verify import L, SR, SG, to_bytes, to_word

M32 = 0xffffffff
def rotl32(w, o): return ((w << o) | (w >> (32 - o))) & M32

def aes_p(state):
    sb = [(state[i] >> (8 * j)) & 0xff for i in range(4) for j in range(4)]
    result = [0, 0, 0, 0]
    for j in range(4):
        w = (sb[(j * 4 + 0) % 16] << 24) | (sb[(j * 4 + 5) % 16] << 0) | \
            (sb[(j * 4 + 10) % 16] << 8) | (sb[(j * 4 + 15) % 16] << 16)
        t = rotl32(w, 16) ^ ((w << 1) & 0xfefefefe) ^ ((((w >> 7) & 0x01010101) * 0x1b) & M32)
        result[j] = (w ^ t ^ rotl32(t, 8)) & M32
    return result

def aes_p_mask(bitpos):
    """apply aes_p to the 128-bit vector e_bitpos, return 128-bit mask"""
    st = [0, 0, 0, 0]
    st[bitpos // 32] = 1 << (bitpos % 32)
    out = aes_p(st)
    w = 0
    for i in range(4):
        for j in range(4):
            w |= ((out[i] >> (8 * j)) & 0xff) << (8 * (4 * i + j))
    return w

bad = 0
for b in range(128):
    got = aes_p_mask(b)
    want = to_word(L(to_bytes(1 << b)))
    if got != want:
        bad += 1
        if bad <= 3:
            print('MISMATCH at bit %d: aes_p=%032x  L=%032x' % (b, got, want))
print('aes_p == MC o SR on all 128 basis vectors:', bad == 0)

# transpose check: the generator writes  in_i = XOR_{j in supp(aes_p(e_i))} out_j,
# which is the correct pull-back  in = L^T out  iff supp(aes_p(e_i)) = {j : L[j][i]=1}
ok = True
for b in range(0, 128, 7):
    sup = [j for j in range(128) if (aes_p_mask(b) >> j) & 1]
    want = [j for j in range(128) if (to_word(L(to_bytes(1 << j))) >> b) & 1]
    ok &= (sup == want)
print('support(aes_p(e_i)) == {j : L[j][i] = 1}:', ok)

# ---------------- census of snow.py ----------------
path = r'C:\Users\Lenovo\Desktop\Processing\snowa\SNOW_A_Code\SearchingTool\snow.py'
if not os.path.exists(path):
    print('snow.py not found'); sys.exit()
needles = [b's.add(s.ite(', b's.add(s.or_(s.not_(s.and_(', b's.add(AA_', b's.add(BB_', b's.add(CC_',
           b's.add(Total', b's.add(S1 ==', b's.add(S2 ==', b's.add(S3 ==', b'print s.check()',
           b'print s.model()', b'\n']
cnt = {n: 0 for n in needles}
tail = b''
size = os.path.getsize(path)
with open(path, 'rb') as f:
    while True:
        chunk = f.read(1 << 22)
        if not chunk: break
        for n in needles:
            cnt[n] += chunk.count(n)
        tail = (tail + chunk)[-600:]
print('\nsnow.py size = %.1f MB' % (size / 2**20))
for n in needles:
    print('  %-28s %d' % (n.decode('latin-1'), cnt[n]))
print('\nlast bytes:\n' + tail.decode('latin-1', 'replace'))

# expected counts
import glob, re
tot_clauses = 0
for k in range(1, 9):
    t = open(path.replace('snow.py', 'sboxR%d.txt' % (4 * k)), encoding='ascii').read()
    tot_clauses += t[:t.index(';')].count('(')
print('clauses in the 8 class files: %d -> expected clause lines 3*16*%d = %d'
      % (tot_clauses, tot_clauses, 3 * 16 * tot_clauses))
print('expected s.ite lines: 3*16*8 = 384 ; AA/BB/CC lines: 48/16/16')

