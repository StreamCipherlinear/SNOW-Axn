"""Count byte patterns in the generated STP model snow.py and print samples.

Inputs: a hard-coded external snow.py (development copy, not in this repository).
Outputs: occurrence counts for each s.add pattern and the first sample of each.
Paper object: none; development aid for the mask model of Sections 3.2 and 3.3.
"""

import os, sys
p = r'C:\Users\Lenovo\Desktop\Processing\snowa\SNOW_A_Code\SearchingTool\snow.py'
needles = [b's.add(s.ite(', b's.add(s.or_(s.not_(s.and_(', b's.add(AA_', b's.add(BB_', b's.add(CC_',
           b's.add(Total', b's.add(S1 ==', b's.add(S2 ==', b's.add(S3 ==', b's.add(Sigma', b'\n']
cnt = {n: 0 for n in needles}
first = {}
CH = 1 << 22
carry = b''
with open(p, 'rb') as f:
    while True:
        chunk = f.read(CH)
        if not chunk:
            break
        data = carry + chunk
        for n in needles:
            cnt[n] += data.count(n)
            if n not in first and n in data:
                idx = data.index(n)
                first[n] = data[max(0, idx - 40):idx + 220].decode('latin-1', 'replace')
        carry = data[-(max(len(n) for n in needles) - 1):]
print('snow.py %.1f MB' % (os.path.getsize(p) / 2**20))
for n in needles:
    print('%-30s %d' % (repr(n)[1:], cnt[n]))
print()
for n in [b's.add(s.or_(s.not_(s.and_(', b's.add(s.ite(', b's.add(AA_0', b'beta_0 |',
          b's.add(S1 ==', b's.add(Total']:
    print('--- sample for %s ---' % repr(n)[1:])
    print(first.get(n, '(not found)').strip()[:400])
    print()
