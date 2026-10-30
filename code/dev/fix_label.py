"""Rename the label sec:model to sec:collision in sec5_multiple.tex.

Inputs: sec5_multiple.tex at its absolute path.
Outputs: the rewritten file and the number of replacements.
Paper object: none; one-off editing script, not part of the reproduction.
"""

p = r'C:\Users\Lenovo\Desktop\Processing\snowa\SNOW-A\sec5_multiple.tex'
s = open(p, encoding='utf-8-sig').read()
n = s.count('sec:model')
s = s.replace('sec:model', 'sec:collision')
open(p, 'w', encoding='utf-8').write(s)
print('replaced in sec5_multiple.tex:', n)
