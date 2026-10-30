"""Change Proposition references for thm:profile into Section references.

Inputs: every .tex file in the paper directory (absolute path below).
Outputs: rewritten .tex files, a replacement count and any leftover label.
Paper object: none; one-off editing script, not part of the reproduction.
"""

import glob, os
base = r'C:\Users\Lenovo\Desktop\Processing\snowa\SNOW-A'
total = 0
for p in glob.glob(os.path.join(base, '*.tex')):
    s = open(p, encoding='utf-8-sig').read()
    n = s.count(r'Proposition~\ref{thm:profile}')
    if n:
        s = s.replace(r'Proposition~\ref{thm:profile}', r'Section~\ref{sec:profile}')
        open(p, 'w', encoding='utf-8').write(s)
        print(os.path.basename(p), n)
        total += n
print('total replaced:', total)
# any dangling thm:profile left?
for p in glob.glob(os.path.join(base, '*.tex')):
    s = open(p, encoding='utf-8-sig').read()
    if 'thm:profile' in s:
        print('LEFTOVER in', os.path.basename(p), s.count('thm:profile'))
