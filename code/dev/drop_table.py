"""Remove the tab:designers-boxes table from appendix.tex and fix one sentence.

Inputs: appendix.tex at its absolute path.
Outputs: a rewritten appendix.tex; prints the removed length and leftover count.
Paper object: none; one-off editing script, not part of the reproduction.
"""

import io
p = r'C:\Users\Lenovo\Desktop\Processing\snowa\SNOW-A\appendix.tex'
s = open(p, encoding='utf-8-sig').read()
i = s.find(r'\label{tab:designers-boxes}')
j = s.rfind(r'\begin{table}', 0, i)
k = s.find(r'\end{table}', i) + len(r'\end{table}')
print('removing', k - j, 'chars at', j)
s = s[:j] + s[k:]
s = s.replace(r'Table~\ref{tab:designers-boxes} lists the boxes individually',
              r'The right half of Table~\ref{tab:boxes-opt} lists them individually')
open(p, 'w', encoding='utf-8').write(s)
print('designers-boxes refs left:', s.count('designers-boxes'))
