# Audit of the 2021 clause files: does (sboxR{4k}.txt) describe exactly the set
#   {(in,out) : |LAT[in][out]| = 4k}   (AES S-box), as the STP generator assumes?
import re, sys, os
import numpy as np

D = r'C:\Users\Lenovo\Desktop\Processing\snowa\SNOW_A_Code\SearchingTool'
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
from verify import LAT, ILAT

pat = re.compile(r'\(([^)]*)\)')
lit = re.compile(r"(in|out)(\d)(\'?)")

def parse(fn):
    t = open(os.path.join(D, fn), encoding='ascii').read()
    assert t.count(';') == 1, 'unexpected terminator'
    body = t[:t.index(';')].strip()
    clauses = []
    for grp in pat.findall(body):
        lits = []
        for kind, bit, prime in lit.findall(grp):
            lits.append((kind, int(bit), 0 if prime else 1))
        clauses.append(lits)
    # every character must be consumed by the pattern (no silent junk)
    recon = ''.join('(' + ''.join(f'{k}{b}' + ("'" if v == 0 else '') + '+' for k, b, v in c)[:-1] + ')' for c in clauses)
    return clauses, (recon == body)

# 16-bit pattern index p: high byte = in-pattern m, low byte = out-pattern u
P = np.arange(65536)
m = (P >> 8) & 0xff
u = P & 0xff
bit = {('in', b): ((m >> b) & 1).astype(bool) for b in range(8)}
bit.update({('out', b): ((u >> b) & 1).astype(bool) for b in range(8)})

print('%-12s %6s %5s %8s %8s %8s' % ('file', '#clauses', 'clean', '#models', '#|LAT|=cor', '#(in,out) swapped'))
for k in range(1, 9):
    fn = f'sboxR{4*k}.txt'
    clauses, clean = parse(fn)
    sat = np.ones(65536, dtype=bool)
    for c in clauses:
        cs = np.zeros(65536, dtype=bool)
        for kind, b, v in c:
            cs |= (bit[(kind, b)] == bool(v))
        sat &= cs
    cor = 4 * k
    A = np.abs(LAT)
    same = int(np.sum(sat & (A.ravel() == cor)))
    swapped = A.T.ravel()
    print('%-12s %6d %5s %8d %8d %8d' % (fn, len(clauses), clean, int(sat.sum()), same,
                                         int(np.sum(sat & (swapped == cor)))))

# global consistency: union of the eight classes must be all nonzero |LAT| pairs
allcls = np.zeros(65536, dtype=bool)
for k in range(1, 9):
    clauses, _ = parse(f'sboxR{4*k}.txt')
    sat = np.ones(65536, dtype=bool)
    for c in clauses:
        cs = np.zeros(65536, dtype=bool)
        for kind, b, v in c:
            cs |= (bit[(kind, b)] == bool(v))
        sat &= cs
    allcls |= sat
A = np.abs(LAT).ravel()
print('\nunion of the 8 classes == {|LAT| != 0} without (0,0):',
      bool(np.array_equal(allcls, (A != 0) & (P != 0))))


# what does the unscaled sbox.txt describe?
clauses, clean = parse('sbox.txt')
sat = np.ones(65536, dtype=bool)
for c in clauses:
    cs = np.zeros(65536, dtype=bool)
    for kind, b, v in c:
        cs |= (bit[(kind, b)] == bool(v))
    sat &= cs
print('\nsbox.txt: %d clauses, clean=%s, models=%d' % (len(clauses), clean, int(sat.sum())))
print('  agrees with {|LAT|>=4}  :', bool(np.array_equal(sat, (A != 0))))
print('  agrees with {|LAT|>=4} minus (0,0):', bool(np.array_equal(sat, (A != 0) & (P != 0))))
print('  agrees with {|LAT|=0}   :', bool(np.array_equal(sat, (A == 0))))
print('  |LAT| distribution of its models:', {int(v): int(np.sum(sat & (A == v))) for v in np.unique(A[sat])})

