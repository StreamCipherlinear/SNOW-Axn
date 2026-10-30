"""Replace two sec:collision references across the paper's .tex files.

Inputs: every .tex file in the paper directory (absolute path below).
Outputs: rewritten .tex files and a per-file count of replacements left.
Paper object: none; one-off editing script, not part of the reproduction.
"""

import glob, os
base = r'C:\Users\Lenovo\Desktop\Processing\snowa\SNOW-A'
for p in glob.glob(os.path.join(base, '*.tex')):
    s = open(p, encoding='utf-8-sig').read()
    if 'sec:collision' not in s:
        continue
    orig = s
    s = s.replace(r'the model of Section~\ref{sec:collision}', r'the model above')
    s = s.replace(r'(Section~\ref{sec:collision})', r'(Section~\ref{sec:multiple})')
    if s != orig:
        open(p, 'w', encoding='utf-8').write(s)
        print('patched', os.path.basename(p), 'left:', s.count('sec:collision'))
    else:
        print('NO CHANGE', os.path.basename(p), 'left:', s.count('sec:collision'))
