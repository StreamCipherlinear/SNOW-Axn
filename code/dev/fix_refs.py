"""Change Theorem references for thm:profile into Proposition references.

Inputs: every .tex file in the paper directory (absolute path below).
Outputs: rewritten .tex files and the number of replacements per file.
Paper object: none; one-off editing script, not part of the reproduction.
"""

import glob, os
base = r'C:\Users\Lenovo\Desktop\Processing\snowa\SNOW-A'
total = 0
for p in glob.glob(os.path.join(base, '*.tex')):
    s = open(p, encoding='utf-8-sig').read()
    n = s.count(r'Theorem~\ref{thm:profile}')
    if n:
        s = s.replace(r'Theorem~\ref{thm:profile}', r'Proposition~\ref{thm:profile}')
        open(p, 'w', encoding='utf-8').write(s)
        print(os.path.basename(p), n)
        total += n
print('total replaced:', total)
